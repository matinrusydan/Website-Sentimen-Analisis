"""
Modul Sentiment Analysis dengan IndoBERT.
Mendukung context-aware prediction menggunakan judul video + komentar.
"""
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils.logger import setup_logger
from config import Config
from services.preprocessing import preprocess_for_model

logger = setup_logger(__name__)

# Gunakan model dari konfigurasi (sekarang menunjuk ke indobert_finetuned)
LOCAL_MODEL_DIR = Config.MODEL_DIR

# Cek apakah ada model domain-adapted (prioritas lebih tinggi)
ADAPTED_MODEL_DIR = os.path.join(Config.BASE_DIR, 'models', 'indobert_adapted')

if os.path.exists(os.path.join(ADAPTED_MODEL_DIR, "model.safetensors")):
    MODEL_NAME = ADAPTED_MODEL_DIR
    logger.info("Menggunakan model domain-adapted.")
elif os.path.exists(os.path.join(LOCAL_MODEL_DIR, "model.safetensors")):
    MODEL_NAME = LOCAL_MODEL_DIR
else:
    MODEL_NAME = "mdhugol/indonesia-bert-sentiment-classification"

tokenizer = None
model = None
is_model_loaded = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_model():
    global tokenizer, model, is_model_loaded
    if is_model_loaded:
        return True
        
    try:
        logger.info(f"Loading IndoBERT model from: {MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True if MODEL_NAME != "mdhugol/indonesia-bert-sentiment-classification" else False)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, local_files_only=True if MODEL_NAME != "mdhugol/indonesia-bert-sentiment-classification" else False)
        model.to(device)
        model.eval()
        is_model_loaded = True
        logger.info(f"Model loaded successfully on {device}.")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False

def reload_model():
    """Force reload model (digunakan setelah domain adaptation selesai)."""
    global is_model_loaded, MODEL_NAME
    is_model_loaded = False
    
    # Re-check model priority
    if os.path.exists(os.path.join(ADAPTED_MODEL_DIR, "model.safetensors")):
        MODEL_NAME = ADAPTED_MODEL_DIR
    elif os.path.exists(os.path.join(LOCAL_MODEL_DIR, "model.safetensors")):
        MODEL_NAME = LOCAL_MODEL_DIR
    else:
        MODEL_NAME = "mdhugol/indonesia-bert-sentiment-classification"
    
    return init_model()

def predict_sentiment(text, video_title=None):
    """
    Predict sentiment untuk teks menggunakan IndoBERT.
    Mendukung context-aware prediction dengan judul video.
    
    Args:
        text: Teks komentar (bisa sudah cleaned atau raw)
        video_title: Judul video YouTube (opsional)
    
    Returns:
        Tuple (label, confidence_score, probabilities_dict)
        label: 'positive', 'negative', atau 'neutral'
        confidence_score: float 0-1
        probabilities_dict: dict dengan probabilitas untuk setiap kelas
    """
    if not is_model_loaded:
        init_model()
        
    if not text or model is None or tokenizer is None:
        return "neutral", 0.0, {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34}
        
    try:
        # Preprocessing khusus model (ringan, preserve semantik)
        processed_text = preprocess_for_model(text, video_title=video_title)
        
        inputs = tokenizer(processed_text, return_tensors="pt", truncation=True, max_length=128, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        predicted_class_id = torch.argmax(logits, dim=-1).item()
        
        # Probabilitas per kelas
        probs = {
            'positive': round(probabilities[0][0].item(), 4),
            'neutral': round(probabilities[0][1].item(), 4),
            'negative': round(probabilities[0][2].item(), 4),
        }
        
        # Confidence score
        score = probabilities[0][predicted_class_id].item()
        
        # Label mapping: 0=positive, 1=neutral, 2=negative
        label_map = {0: 'positive', 1: 'neutral', 2: 'negative'}
        label = label_map.get(predicted_class_id, 'neutral')
            
        return label, score, probs

    except Exception as e:
        logger.error(f"Error predicting sentiment: {e}")
        return "neutral", 0.0, {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34}
