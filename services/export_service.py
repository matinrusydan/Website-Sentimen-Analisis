import pandas as pd
import os
from config import Config
from services.database_service import get_all_comments_by_url
from utils.logger import setup_logger

logger = setup_logger(__name__)

def export_comments_to_csv(video_url, filename="comments_export.csv"):
    """
    Export comments for a specific video URL to a CSV file.
    """
    try:
        comments = get_all_comments_by_url(video_url)
        if not comments:
            return None
            
        data = []
        for c in comments:
            data.append({
                'author': c.author,
                'text_original': c.text_original,
                'text_clean': c.text_clean,
                'sentiment': c.sentiment_label,
                'score': c.sentiment_score,
                'created_at': c.created_at
            })
            
        df = pd.DataFrame(data)
        os.makedirs(Config.PROCESSED_DATA_DIR, exist_ok=True)
        filepath = os.path.join(Config.PROCESSED_DATA_DIR, filename)
        df.to_csv(filepath, index=False)
        
        logger.info(f"Successfully exported data to {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Error exporting to CSV: {e}")
        return None
