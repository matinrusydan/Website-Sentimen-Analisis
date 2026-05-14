import os
import torch
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from config import Config
from app import create_app
from extensions import db
from services.database_service import Comment
app = create_app()
from torch.amp import autocast, GradScaler
import time

# Label mapping based on mdhugol/indonesia-bert-sentiment-classification
# It uses 0 for positive, 1 for neutral, 2 for negative (or similar, we will strictly map it)
# Wait, let's look at the original model config if we want to be safe, but usually:
# mdhugol's config: 0: positive, 1: neutral, 2: negative
LABEL_MAP = {'positive': 0, 'neutral': 1, 'negative': 2}
REVERSE_LABEL_MAP = {0: 'positive', 1: 'neutral', 2: 'negative'}

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def get_training_data():
    import pandas as pd
    tsv_path = os.path.join(Config.BASE_DIR, "indonlu", "dataset", "smsa_doc-sentiment-prosa", "train_preprocess.tsv")
    if not os.path.exists(tsv_path):
        print(f"[ERROR] Dataset IndoNLU tidak ditemukan di: {tsv_path}")
        return None, None
        
    print(f"Membaca Dataset Benchmark IndoNLU dari {tsv_path}...")
    df = pd.read_csv(tsv_path, sep='\t', header=None, names=['text', 'label'])
    texts, labels = [], []
    for _, row in df.iterrows():
        text = str(row['text'])
        lbl = str(row['label']).strip().lower()
        if lbl in LABEL_MAP:
            texts.append(text)
            labels.append(LABEL_MAP[lbl])
    return texts, labels

def get_test_data():
    import pandas as pd
    tsv_path = os.path.join(Config.BASE_DIR, "indonlu", "dataset", "smsa_doc-sentiment-prosa", "test_preprocess.tsv")
    if not os.path.exists(tsv_path):
        print(f"[ERROR] Test Dataset IndoNLU tidak ditemukan di: {tsv_path}")
        return None, None
        
    df = pd.read_csv(tsv_path, sep='\t', header=None, names=['text', 'label'])
    texts, labels = [], []
    for _, row in df.iterrows():
        text = str(row['text'])
        lbl = str(row['label']).strip().lower()
        if lbl in LABEL_MAP:
            texts.append(text)
            labels.append(LABEL_MAP[lbl])
    return texts, labels

def train_model(epochs=3, batch_size=8, lr=2e-5):
    print("Mempersiapkan data pelatihan...")
    texts, labels = get_training_data()
    
    if texts is None or len(texts) < 10:
        print("[ERROR] Data terlalu sedikit (< 10). Tolong tambah data scraping agar Fine-Tuning tidak error.")
        return

    print(f"[OK] Ditemukan {len(texts)} data pelatihan.")
    
    # Split Data (80% Train, 20% Val)
    X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.2, random_state=42)
    
    print("Mempersiapkan Tokenizer dan Model IndoBERT Lokal...")
    model_dir = Config.MODEL_DIR
    if not os.path.exists(model_dir):
        print("[ERROR] Folder model tidak ditemukan. Harap pastikan model sudah di-download.")
        return
        
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model = BertForSequenceClassification.from_pretrained(model_dir, num_labels=3)

    train_dataset = SentimentDataset(X_train, y_train, tokenizer, max_length=128)
    val_dataset = SentimentDataset(X_val, y_val, tokenizer, max_length=128)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # Setup GPU (CUDA)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    import sys
    print(f"\n=============================================")
    print(f"DIAGNOSTIC INFO:")
    print(f"Python EXE: {sys.executable}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f">>> DEVICE TRAINING: {device.type.upper()}")
    if device.type == 'cuda':
        print(f"VRAM TERDETEKSI: {torch.cuda.get_device_name(0)}")
        print("Menggunakan Mixed Precision FP16 (Anti OOM)")
    else:
        print("PERINGATAN: GPU CUDA tidak terdeteksi! Proses akan sangat lama (Hitungan Jam/Hari).")
    print(f"=============================================\n")

    model = model.to(device)
    
    # Optimasi Hyperparameter berdasarkan IndoNLU Benchmark
    epochs = 5
    lr = 5e-6
    optimizer = AdamW(model.parameters(), lr=lr)
    
    # LR Scheduler (Warmup) untuk stabilitas (Terinspirasi dari IndoNLU)
    from transformers import get_linear_schedule_with_warmup
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)
    
    scaler = GradScaler('cuda') if device.type == 'cuda' else None

    print(f"Mulai Fine-Tuning selama {epochs} Epochs dengan LR={lr} dan MaxLen=128...")
    
    for epoch in range(epochs):
        print(f"\n[Epoch {epoch + 1} / {epochs}]")
        start_time = time.time()
        
        # Training Phase
        model.train()
        total_loss = 0
        correct_train = 0
        total_train = 0
        
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['labels'].to(device)
            
            # Forward pass dengan Mixed Precision jika pakai GPU
            if device.type == 'cuda':
                with autocast('cuda'):
                    outputs = model(input_ids, attention_mask=attention_mask, labels=targets)
                    loss = outputs.loss
                    logits = outputs.logits
                
                # Backward pass
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step() # Update learning rate
            else:
                # CPU biasa
                outputs = model(input_ids, attention_mask=attention_mask, labels=targets)
                loss = outputs.loss
                logits = outputs.logits
                loss.backward()
                optimizer.step()
                scheduler.step() # Update learning rate

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct_train += (preds == targets).sum().item()
            total_train += targets.size(0)
            
            if (step + 1) % 5 == 0:
                print(f"   Batch {step+1}/{len(train_loader)} - Loss: {loss.item():.4f}")

        train_acc = correct_train / total_train
        
        # Validation Phase
        model.eval()
        val_loss = 0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                targets = batch['labels'].to(device)
                
                if device.type == 'cuda':
                    with autocast('cuda'):
                        outputs = model(input_ids, attention_mask=attention_mask, labels=targets)
                        loss = outputs.loss
                        logits = outputs.logits
                else:
                    outputs = model(input_ids, attention_mask=attention_mask, labels=targets)
                    loss = outputs.loss
                    logits = outputs.logits
                    
                val_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                correct_val += (preds == targets).sum().item()
                total_val += targets.size(0)

        val_acc = correct_val / total_val if total_val > 0 else 0
        elapsed_time = time.time() - start_time
        
        print(f"[DONE] Selesai Epoch {epoch + 1} dalam {elapsed_time:.2f} detik")
        print(f"   Train Loss: {total_loss/len(train_loader):.4f} | Train Acc: {train_acc:.4f}")
        print(f"   Val Loss: {val_loss/max(1, len(val_loader)):.4f} | Val Acc: {val_acc:.4f}")

    # Save Fine-Tuned Model
    output_dir = os.path.join(os.path.dirname(__file__), 'models', 'indobert_finetuned')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"\nMenyimpan model hasil Fine-Tuning ke folder: {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("[OK] Fine-Tuning selesai! Model telah disimpan.")

    # ==========================================
    # EVALUASI MODEL MENGGUNAKAN TEST DATASET
    # ==========================================
    print("\n=============================================")
    print("MELAKUKAN EVALUASI PADA TEST DATASET INDONLU...")
    test_texts, test_labels = get_test_data()
    if test_texts:
        test_dataset = SentimentDataset(test_texts, test_labels, tokenizer, max_length=128)
        test_loader = DataLoader(test_dataset, batch_size=32)
        
        model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                targets = batch['labels'].to(device)
                
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
        # Calculate Metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        import json
        
        acc = accuracy_score(all_targets, all_preds)
        prec = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
        rec = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
        cm = confusion_matrix(all_targets, all_preds).tolist()
        
        metrics = {
            "accuracy": round(acc * 100, 2),
            "precision": round(prec * 100, 2),
            "recall": round(rec * 100, 2),
            "f1_score": round(f1 * 100, 2),
            "confusion_matrix": cm,
            "labels": ["Positive", "Neutral", "Negative"] # 0: positive, 1: neutral, 2: negative (Sesuai LABEL_MAP)
        }
        
        results_dir = os.path.join(Config.BASE_DIR, 'results')
        os.makedirs(results_dir, exist_ok=True)
        with open(os.path.join(results_dir, 'model_metrics.json'), 'w') as f:
            json.dump(metrics, f)
            
        print(f"[OK] Evaluasi selesai. Accuracy Test: {metrics['accuracy']}%")
        print("[OK] Metrik berhasil disimpan ke 'results/model_metrics.json'.")
    else:
        print("[WARNING] Test Dataset tidak ditemukan, melewati tahap evaluasi akhir.")

if __name__ == '__main__':
    # Hapus warning transformers
    os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
    # Panggil fungsi training sesuai optimasi IndoNLU
    train_model()
