from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from wordcloud import WordCloud
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from config import Config
from utils.logger import setup_logger

logger = setup_logger(__name__)

def generate_wordcloud(texts, filename="wordcloud.png"):
    """
    Generate and save a wordcloud from a list of texts.
    """
    if not texts:
        return None
        
    try:
        combined_text = " ".join(texts)
        wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=100).generate(combined_text)
        
        filepath = os.path.join(Config.STATIC_DIR, 'wordcloud', filename)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(filepath)
        plt.close()
        
        return f"static/wordcloud/{filename}"
    except Exception as e:
        logger.error(f"Error generating wordcloud: {e}")
        return None

def perform_topic_modeling(texts, num_topics=3, num_words=5):
    """
    Perform simple LDA topic modeling.
    Returns a list of topics with their top words.
    """
    if not texts or len(texts) < 5:
        return []
        
    try:
        vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
        dtm = vectorizer.fit_transform(texts)
        
        lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
        lda.fit(dtm)
        
        feature_names = vectorizer.get_feature_names_out()
        topics = []
        
        for topic_idx, topic in enumerate(lda.components_):
            top_words = [feature_names[i] for i in topic.argsort()[:-num_words - 1:-1]]
            topics.append({
                'topic': f"Topik {topic_idx + 1}",
                'words': top_words
            })
            
        return topics
    except Exception as e:
        logger.error(f"Error in topic modeling: {e}")
        return []
