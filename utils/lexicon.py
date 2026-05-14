import re

# Kamus kata positif sederhana (bisa diperluas)
POSITIVE_WORDS = {
    "bagus", "baik", "setuju", "mendukung", "dukung", "keren", "mantap", "hebat", "luar biasa", 
    "bantu", "membantu", "solusi", "tepat", "setuju", "gratis", "bergizi", "bermanfaat", "manfaat",
    "maju", "sehat", "cerdas", "pintar", "enak", "kenyang", "top", "sukses", "lanjutkan", "pro",
    "terbaik", "gizi", "peduli", "suka", "senang", "harapan", "alhamdulillah", "amin", "aamiin"
}

# Kamus kata negatif sederhana (bisa diperluas)
NEGATIVE_WORDS = {
    "jelek", "buruk", "tolak", "menolak", "korupsi", "rugi", "merugikan", "utang", "hutang",
    "beban", "hancur", "gagal", "bohong", "hoax", "pencitraan", "buang", "pemborosan", "boros",
    "pajak", "mahal", "susah", "menderita", "lapar", "gizi buruk", "stunting", "kacau", "aneh",
    "lucu", "parah", "kecewa", "bodoh", "lawak", "ngawur", "kurang", "batal", "distop", "stop",
    "badokan", "badokkan", "badok", "zohseng"
}

def get_lexicon_label(text):
    """
    Memberikan label berdasarkan perhitungan kata positif vs negatif (Lexicon).
    Mengembalikan: 'positive', 'negative', atau 'neutral'
    """
    if not text:
        return "neutral"
        
    # Tokenisasi sederhana
    words = re.findall(r'\b\w+\b', text.lower())
    
    pos_score = sum(1 for word in words if word in POSITIVE_WORDS)
    neg_score = sum(1 for word in words if word in NEGATIVE_WORDS)
    
    if pos_score > neg_score:
        return "positive"
    elif neg_score > pos_score:
        return "negative"
    else:
        return "neutral"
