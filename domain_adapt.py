"""
Domain Adaptation Fine-Tuning untuk IndoBERT.

Strategi:
1. Load model yang sudah fine-tuned (dari IndoNLU SmSA)
2. Pseudo-label komentar YouTube dengan confidence >= 0.95
3. Gabungkan dataset IndoNLU SmSA + pseudo-labeled YouTube
4. Continue fine-tuning dengan learning rate kecil (2e-6)
5. Simpan model baru ke models/indobert_adapted/

Penggunaan:
    python domain_adapt.py
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import json
import time
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from config import Config

# ============================================================================
# KONFIGURASI
# ============================================================================
LABEL_MAP = {'positive': 0, 'neutral': 1, 'negative': 2}
REVERSE_LABEL_MAP = {0: 'positive', 1: 'neutral', 2: 'negative'}

# Hyperparameters untuk domain adaptation (lebih konservatif)
EPOCHS = 3
LEARNING_RATE = 2e-6       # Lebih kecil dari fine-tuning awal (5e-6)
BATCH_SIZE = 16
MAX_LENGTH = 128
WARMUP_RATIO = 0.1
CONFIDENCE_THRESHOLD = 0.90  # Minimum confidence untuk pseudo-label

# Paths
SOURCE_MODEL_DIR = Config.MODEL_DIR  # models/indobert_finetuned
OUTPUT_MODEL_DIR = os.path.join(Config.BASE_DIR, 'models', 'indobert_adapted')
SMSA_TRAIN_PATH = os.path.join(Config.BASE_DIR, "indonlu", "dataset", "smsa_doc-sentiment-prosa", "train_preprocess.tsv")
SMSA_TEST_PATH = os.path.join(Config.BASE_DIR, "indonlu", "dataset", "smsa_doc-sentiment-prosa", "test_preprocess.tsv")
METRICS_PATH = os.path.join(Config.BASE_DIR, 'results', 'model_metrics_adapted.json')


class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=MAX_LENGTH):
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
            text, add_special_tokens=True, max_length=self.max_length,
            return_token_type_ids=False, padding='max_length',
            truncation=True, return_attention_mask=True, return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def load_smsa_data(path):
    """Load dataset IndoNLU SmSA."""
    if not os.path.exists(path):
        print(f"[ERROR] Dataset tidak ditemukan: {path}")
        return [], []
    df = pd.read_csv(path, sep='\t', header=None, names=['text', 'label'])
    texts, labels = [], []
    for _, row in df.iterrows():
        text = str(row['text'])
        lbl = str(row['label']).strip().lower()
        if lbl in LABEL_MAP:
            texts.append(text)
            labels.append(LABEL_MAP[lbl])
    return texts, labels


def generate_pseudo_labels(model, tokenizer, device):
    """
    Generate pseudo-labels dari komentar YouTube di database.
    Hanya ambil komentar dengan confidence >= threshold.
    """
    from services.preprocessing import preprocess_for_model
    
    print(f"\n{'='*60}")
    print(f"[PSEUDO-LABELING] Threshold confidence: {CONFIDENCE_THRESHOLD}")
    print(f"{'='*60}")
    
    # Baca dari database
    from app import create_app
    app = create_app()
    with app.app_context():
        from services.database_service import Comment
        comments = Comment.query.filter(Comment.text_clean.isnot(None)).all()
        
        if not comments:
            print("[WARNING] Tidak ada komentar YouTube di database.")
            return [], []
        
        print(f"Total komentar di database: {len(comments)}")
        
        texts, labels = [], []
        model.eval()
        
        for c in comments:
            # Gunakan preprocessing model (ringan)
            processed = preprocess_for_model(c.text_original, video_title=c.video_title)
            if not processed or len(processed.strip()) < 3:
                continue
            
            inputs = tokenizer(processed, return_tensors="pt", truncation=True, 
                             max_length=MAX_LENGTH, padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            confidence = probs.max().item()
            pred_class = probs.argmax().item()
            
            if confidence >= CONFIDENCE_THRESHOLD:
                texts.append(processed)
                labels.append(pred_class)
        
        print(f"Pseudo-labeled data (confidence >= {CONFIDENCE_THRESHOLD}): {len(texts)}")
        
        if texts:
            label_dist = {REVERSE_LABEL_MAP[l]: labels.count(l) for l in set(labels)}
            print(f"Distribusi pseudo-labels: {label_dist}")
    
    return texts, labels


def domain_adapt():
    """Main domain adaptation pipeline."""
    print("\n" + "="*60)
    print("  DOMAIN ADAPTATION FINE-TUNING IndoBERT")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[INFO] Device: {device}")
    
    # ================================================================
    # Step 1: Load model yang sudah fine-tuned
    # ================================================================
    print(f"\n[Step 1] Loading fine-tuned model dari: {SOURCE_MODEL_DIR}")
    
    if not os.path.exists(os.path.join(SOURCE_MODEL_DIR, "model.safetensors")):
        print("[ERROR] Model fine-tuned tidak ditemukan! Jalankan fine-tuning dulu.")
        sys.exit(1)
    
    tokenizer = BertTokenizer.from_pretrained(SOURCE_MODEL_DIR)
    model = BertForSequenceClassification.from_pretrained(SOURCE_MODEL_DIR, num_labels=3)
    model.to(device)
    
    print(f"[OK] Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # ================================================================
    # Step 2: Load IndoNLU SmSA dataset
    # ================================================================
    print(f"\n[Step 2] Loading IndoNLU SmSA dataset...")
    smsa_texts, smsa_labels = load_smsa_data(SMSA_TRAIN_PATH)
    print(f"[OK] SmSA data: {len(smsa_texts)} samples")
    
    # ================================================================
    # Step 3: Generate pseudo-labels dari YouTube data
    # ================================================================
    print(f"\n[Step 3] Generating pseudo-labels dari YouTube comments...")
    yt_texts, yt_labels = generate_pseudo_labels(model, tokenizer, device)
    
    # ================================================================
    # Step 4: Gabungkan datasets
    # ================================================================
    print(f"\n[Step 4] Menggabungkan datasets...")
    
    # Tambahkan YouTube data dengan bobot lebih (oversampling 3x)
    # agar model lebih terpengaruh domain YouTube
    oversample_factor = 3
    all_texts = smsa_texts + (yt_texts * oversample_factor)
    all_labels = smsa_labels + (yt_labels * oversample_factor)
    
    print(f"  SmSA: {len(smsa_texts)} samples")
    print(f"  YouTube (pseudo-labeled x{oversample_factor}): {len(yt_texts) * oversample_factor} samples")
    print(f"  Total combined: {len(all_texts)} samples")
    
    # Shuffle
    combined = list(zip(all_texts, all_labels))
    np.random.seed(42)
    np.random.shuffle(combined)
    all_texts, all_labels = zip(*combined)
    all_texts, all_labels = list(all_texts), list(all_labels)
    
    # ================================================================
    # Step 5: Domain Adaptation Fine-Tuning
    # ================================================================
    print(f"\n[Step 5] Memulai Domain Adaptation Fine-Tuning...")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Batch Size: {BATCH_SIZE}")
    
    train_dataset = SentimentDataset(all_texts, all_labels, tokenizer, MAX_LENGTH)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    
    from transformers import get_linear_schedule_with_warmup
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    scaler = GradScaler('cuda') if device.type == 'cuda' else None
    
    model.train()
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            if device.type == 'cuda':
                with autocast('cuda'):
                    outputs = model(input_ids, attention_mask=attention_mask, labels=targets)
                    loss = outputs.loss
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(input_ids, attention_mask=attention_mask, labels=targets)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
            
            scheduler.step()
            
            total_loss += loss.item()
            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            
            if (batch_idx + 1) % 50 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                acc = correct / total * 100
                print(f"  Epoch {epoch+1}/{EPOCHS} - Batch {batch_idx+1}/{len(train_loader)} - Loss: {avg_loss:.4f} - Acc: {acc:.1f}%")
        
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = correct / total * 100
        elapsed = time.time() - start_time
        print(f"\n  [Epoch {epoch+1}] Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.1f}% | Time: {elapsed:.0f}s")
    
    # ================================================================
    # Step 6: Simpan model
    # ================================================================
    print(f"\n[Step 6] Menyimpan model ke: {OUTPUT_MODEL_DIR}")
    os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_MODEL_DIR)
    tokenizer.save_pretrained(OUTPUT_MODEL_DIR)
    print("[OK] Model domain-adapted berhasil disimpan!")
    
    # ================================================================
    # Step 7: Evaluasi pada test set IndoNLU
    # ================================================================
    print(f"\n[Step 7] Evaluasi pada test set IndoNLU...")
    test_texts, test_labels = load_smsa_data(SMSA_TEST_PATH)
    
    if test_texts:
        test_dataset = SentimentDataset(test_texts, test_labels, tokenizer, MAX_LENGTH)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        
        model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                targets = batch['labels'].to(device)
                
                if device.type == 'cuda':
                    with autocast('cuda'):
                        outputs = model(input_ids, attention_mask=attention_mask)
                else:
                    outputs = model(input_ids, attention_mask=attention_mask)
                
                preds = torch.argmax(outputs.logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        acc = accuracy_score(all_targets, all_preds) * 100
        precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='weighted')
        cm = confusion_matrix(all_targets, all_preds)
        
        metrics = {
            'accuracy': round(acc, 2),
            'precision': round(precision * 100, 2),
            'recall': round(recall * 100, 2),
            'f1_score': round(f1 * 100, 2),
            'confusion_matrix': cm.tolist(),
            'labels': ['Positive', 'Neutral', 'Negative'],
            'adaptation_info': {
                'source_model': SOURCE_MODEL_DIR,
                'smsa_samples': len(smsa_texts),
                'youtube_samples': len(yt_texts),
                'oversample_factor': oversample_factor,
                'epochs': EPOCHS,
                'learning_rate': LEARNING_RATE,
                'confidence_threshold': CONFIDENCE_THRESHOLD,
            }
        }
        
        os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
        with open(METRICS_PATH, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"  HASIL EVALUASI DOMAIN ADAPTATION")
        print(f"{'='*60}")
        print(f"  Accuracy:  {acc:.2f}%")
        print(f"  Precision: {precision*100:.2f}%")
        print(f"  Recall:    {recall*100:.2f}%")
        print(f"  F1-Score:  {f1*100:.2f}%")
        print(f"\n  Confusion Matrix:")
        print(f"              Pred_Pos  Pred_Neu  Pred_Neg")
        print(f"  Act_Pos     {cm[0][0]:>8}  {cm[0][1]:>8}  {cm[0][2]:>8}")
        print(f"  Act_Neu     {cm[1][0]:>8}  {cm[1][1]:>8}  {cm[1][2]:>8}")
        print(f"  Act_Neg     {cm[2][0]:>8}  {cm[2][1]:>8}  {cm[2][2]:>8}")
        print(f"\n  Metrik disimpan ke: {METRICS_PATH}")
    
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  DOMAIN ADAPTATION SELESAI! ({total_time:.0f} detik)")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    domain_adapt()
