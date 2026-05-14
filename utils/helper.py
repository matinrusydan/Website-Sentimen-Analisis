import os
import pandas as pd
from config import Config
from utils.logger import setup_logger

logger = setup_logger(__name__)

def load_dataset_label():
    """
    Load the dataset label from the root folder (smsa_doc-sentiment-prosa).
    Returns a pandas DataFrame or None if not found.
    """
    dataset_path = os.path.join(Config.DATASET_LABEL_DIR, 'train_preprocess.tsv')
    if os.path.exists(dataset_path):
        try:
            df = pd.read_csv(dataset_path, sep='\t', header=None, names=['text', 'label'])
            logger.info(f"Successfully loaded dataset label with {len(df)} rows.")
            return df
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            return None
    else:
        logger.warning(f"Dataset not found at {dataset_path}")
        return None
