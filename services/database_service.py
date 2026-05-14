from extensions import db
from sqlalchemy import Column, Integer, String, Text, Float, DateTime
from datetime import datetime

class Comment(db.Model):
    __tablename__ = 'comments'
    
    id = Column(Integer, primary_key=True)
    video_url = Column(String(255), nullable=False)
    video_title = Column(String(500), nullable=True)
    author = Column(String(100), nullable=True)
    text_original = Column(Text, nullable=False)
    text_clean = Column(Text, nullable=True)
    lexicon_label = Column(String(20), nullable=True)
    sentiment_label = Column(String(20), nullable=True)
    sentiment_score = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

def save_comment(video_url, author, text_original, video_title=None):
    comment = Comment(video_url=video_url, author=author, text_original=text_original, video_title=video_title)
    db.session.add(comment)
    db.session.commit()
    return comment

def get_all_comments_by_url(video_url):
    return Comment.query.filter_by(video_url=video_url).all()
