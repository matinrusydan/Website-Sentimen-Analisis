import os

class Config:
    # Basic Config
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'mbg-sentiment-secret-key-2026'
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    
    # Database Config
    SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(BASE_DIR, 'database', 'comments.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Directory Config
    DATASETS_DIR = os.path.join(BASE_DIR, 'datasets')
    RAW_DATA_DIR = os.path.join(DATASETS_DIR, 'raw')
    PROCESSED_DATA_DIR = os.path.join(DATASETS_DIR, 'processed')
    LOGS_DIR = os.path.join(BASE_DIR, 'logs')
    STATIC_DIR = os.path.join(BASE_DIR, 'static')
    
    # Dataset Label Config
    DATASET_LABEL_DIR = os.path.join(BASE_DIR, 'smsa_doc-sentiment-prosa')
    
    # Model Config
    PRETRAINED_MODEL_NAME = "indobenchmark/indobert-base-p1"
    MODEL_DIR = os.path.join(BASE_DIR, 'models', 'indobert_finetuned')
