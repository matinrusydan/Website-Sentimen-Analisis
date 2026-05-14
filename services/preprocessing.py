"""
Modul Preprocessing untuk Analisis Sentimen YouTube Indonesia.

Menyediakan dua pipeline:
1. preprocess_for_storage(text) — Cleaning agresif untuk penyimpanan di database
2. preprocess_for_model(text) — Cleaning ringan khusus input IndoBERT (preserve semantik)
"""

import re
from utils.logger import setup_logger

logger = setup_logger(__name__)

# ============================================================================
# KAMUS NORMALISASI SLANG INDONESIA (200+ entries)
# Mencakup: slang medsos, singkatan umum, bahasa netizen YouTube, 
# negasi informal, intensifier, dan konteks politik/MBG
# ============================================================================
SLANG_DICT = {
    # --- Negasi ---
    'gk': 'tidak', 'ga': 'tidak', 'gak': 'tidak', 'kagak': 'tidak',
    'kaga': 'tidak', 'tdk': 'tidak', 'ngga': 'tidak', 'nggak': 'tidak',
    'enggak': 'tidak', 'engga': 'tidak', 'ogah': 'tidak mau', 'gx': 'tidak',
    'g': 'tidak', 'kgk': 'tidak', 'kagk': 'tidak', 'ndak': 'tidak',
    'tak': 'tidak', 'gak': 'tidak', 'gaada': 'tidak ada',

    # --- Kata ganti orang ---
    'sy': 'saya', 'ak': 'aku', 'gw': 'saya', 'gue': 'saya', 'w': 'saya',
    'gua': 'saya', 'ane': 'saya', 'lu': 'kamu', 'lo': 'kamu', 'elu': 'kamu',
    'u': 'kamu', 'ente': 'kamu', 'km': 'kamu', 'kmu': 'kamu',
    'mrk': 'mereka', 'mreka': 'mereka', 'kt': 'kita', 'kta': 'kita',

    # --- Kata hubung & preposisi ---
    'yg': 'yang', 'dgn': 'dengan', 'utk': 'untuk', 'krn': 'karena',
    'krna': 'karena', 'karna': 'karena', 'tp': 'tapi', 'tpi': 'tapi',
    'dr': 'dari', 'dri': 'dari', 'pd': 'pada', 'sm': 'sama',
    'sma': 'sama', 'dlm': 'dalam', 'ttg': 'tentang', 'spy': 'supaya',
    'spya': 'supaya', 'spt': 'seperti', 'kyk': 'kayak', 'kyak': 'kayak',
    'kek': 'seperti', 'drpd': 'daripada',

    # --- Kata kerja ---
    'hrs': 'harus', 'ksh': 'kasih', 'bs': 'bisa', 'bsa': 'bisa',
    'dpt': 'dapat', 'dpat': 'dapat', 'dapet': 'dapat', 'bikin': 'membuat',
    'bkin': 'membuat', 'pake': 'pakai', 'pk': 'pakai', 'pki': 'pakai',
    'mkn': 'makan', 'mkan': 'makan', 'tau': 'tahu', 'tw': 'tahu',
    'liat': 'lihat', 'lht': 'lihat', 'ngeliat': 'melihat',
    'blg': 'bilang', 'ngmng': 'ngomong', 'ngomong': 'berbicara',
    'nyari': 'mencari', 'nanya': 'bertanya', 'ngasih': 'memberikan',
    'ngerti': 'mengerti', 'ntn': 'nonton', 'nntn': 'nonton',
    'trbuang': 'terbuang', 'trbang': 'terbang',

    # --- Kata sifat & keterangan ---
    'bgt': 'banget', 'bngt': 'banget', 'bner': 'benar', 'bnr': 'benar',
    'bgus': 'bagus', 'gede': 'besar', 'dikit': 'sedikit', 'sdkit': 'sedikit',
    'cpt': 'cepat', 'lmbt': 'lambat', 'lbh': 'lebih', 'plg': 'paling',
    'bnyak': 'banyak', 'byk': 'banyak', 'bnyk': 'banyak',

    # --- Kata benda ---
    'org': 'orang', 'mknan': 'makanan', 'pmerintah': 'pemerintah',
    'mslh': 'masalah', 'mslhnya': 'masalahnya', 'brg': 'barang',
    'thun': 'tahun', 'thn': 'tahun', 'bln': 'bulan',
    'ank': 'anak', 'anaknya': 'anaknya', 'ortu': 'orang tua',
    'nyokap': 'ibu', 'bokap': 'ayah',

    # --- Waktu & keadaan ---
    'skrg': 'sekarang', 'skr': 'sekarang', 'dlu': 'dulu',
    'ntar': 'nanti', 'ntr': 'nanti', 'msh': 'masih', 'msih': 'masih',
    'udh': 'sudah', 'sdh': 'sudah', 'uda': 'sudah', 'dah': 'sudah',
    'blm': 'belum', 'blom': 'belum', 'lg': 'lagi', 'sdg': 'sedang',
    'sdng': 'sedang',

    # --- Pertanyaan ---
    'knp': 'kenapa', 'gmn': 'gimana', 'gmna': 'gimana',
    'bgmn': 'bagaimana', 'dmn': 'dimana', 'kpn': 'kapan',
    'brp': 'berapa', 'spa': 'siapa',

    # --- Intensifier & modifier ---
    'bgt': 'banget', 'pol': 'sekali', 'parah': 'parah',
    'bener': 'benar', 'emg': 'memang', 'emng': 'memang', 'mmg': 'memang',
    'emang': 'memang', 'aja': 'saja', 'aj': 'saja', 'doang': 'saja',
    'dong': 'dong', 'deh': 'deh', 'sih': 'sih', 'kok': 'kok',
    'trs': 'terus', 'trus': 'terus',

    # --- Afirmasi ---
    'iy': 'iya', 'iyaa': 'iya', 'yoi': 'iya', 'yup': 'iya',
    'yap': 'iya', 'sip': 'oke', 'oce': 'oke', 'okee': 'oke',
    'wokeh': 'oke',

    # --- Ekspresi / slang medsos ---
    'mantul': 'mantap betul', 'mantab': 'mantap', 'mantep': 'mantap',
    'keren': 'keren', 'anjay': 'anjir', 'anjir': 'gila',
    'bjir': 'gila', 'jir': 'gila', 'wkwk': 'haha',
    'wkwkwk': 'haha', 'kwkw': 'haha', 'hehe': 'haha',
    'xixi': 'haha', 'awkwk': 'haha',
    'bangke': 'bangsat', 'bngst': 'bangsat', 'kampret': 'bangsat',
    'goblog': 'bodoh', 'gblk': 'bodoh', 'tolol': 'bodoh',
    'gpp': 'tidak apa apa', 'gpapa': 'tidak apa apa',
    'smoga': 'semoga', 'mudmud': 'mudah mudahan',

    # --- Konteks Politik / MBG ---
    'koruptor': 'koruptor', 'korup': 'korupsi',
    'rakyat': 'rakyat', 'negara': 'negara',
    'presiden': 'presiden', 'pres': 'presiden',
    'mentri': 'menteri', 'mentrinya': 'menterinya',
    'pemimpin': 'pemimpin', 'penguasa': 'penguasa',
    'demo': 'demonstrasi', 'pilpres': 'pemilihan presiden',
    'pilkada': 'pemilihan kepala daerah',
    'nyaleg': 'calon legislatif', 'caleg': 'calon legislatif',

    # --- Lainnya ---
    'jd': 'jadi', 'jdi': 'jadi', 'jg': 'juga', 'jgn': 'jangan',
    'smua': 'semua', 'bkn': 'bukan', 'thd': 'terhadap',
    'mksd': 'maksud', 'mksdnya': 'maksudnya', 'klo': 'kalau',
    'kl': 'kalau', 'kalo': 'kalau', 'hnya': 'hanya', 'hy': 'hanya',
    'stlh': 'setelah', 'stlah': 'setelah', 'sblm': 'sebelum',
    'sblmnya': 'sebelumnya', 'sbnrnya': 'sebenarnya',
    'sbnernya': 'sebenarnya', 'krj': 'kerja', 'kerjaan': 'pekerjaan',
    'dlm': 'dalam', 'dep': 'depan', 'blkg': 'belakang',
    'atas': 'atas', 'bawah': 'bawah',
    'males': 'malas', 'mager': 'malas gerak',
    'gabut': 'tidak ada kerjaan', 'ngab': 'bro',
    'warga': 'warga', 'masy': 'masyarakat',
    'masyrkt': 'masyarakat', 'msyrkt': 'masyarakat',
}


# ============================================================================
# PIPELINE 1: PREPROCESSING UNTUK STORAGE (agresif, untuk penyimpanan DB)
# ============================================================================

def _clean_urls_mentions(text):
    """Hapus URL, mention, hashtag, dan mojibake."""
    if not text:
        return ""
    # Mojibake
    text = re.sub(r'â€[^\s]*', '', text)
    text = re.sub(r'Ã[^\s]*', '', text)
    text = re.sub(r'ð[^\s]*', '', text)
    # URL
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Mentions (termasuk tanda hubung)
    text = re.sub(r'@[\w\-]+', '', text)
    # Hashtags
    text = re.sub(r'#\w+', '', text)
    return text

def _normalize_informal_repetition(text):
    """Normalisasi kata ulang informal: antek2 → antek antek, pelan" → pelan pelan."""
    if not text:
        return ""
    text = re.sub(r'\b(\w+)2\b', r'\1 \1', text)
    text = re.sub(r'\b(\w+)"(\w+)', r'\1 \1\2', text)
    text = re.sub(r'\b(\w+)"\b', r'\1 \1', text)
    text = re.sub(r'\b(\w+)"\s', r'\1 \1 ', text)
    return text

def _normalize_slang(text):
    """Normalisasi slang/singkatan menggunakan kamus."""
    if not text:
        return ""
    words = text.split()
    return ' '.join(SLANG_DICT.get(w, w) for w in words)

def _remove_repeated_chars(text):
    """Normalkan huruf berulang: baguuus → bagus."""
    if not text:
        return ""
    return re.sub(r'(.)\1{2,}', r'\1', text)

def _whitespace_clean(text):
    """Hapus spasi ganda dan whitespace."""
    if not text:
        return ""
    return re.sub(r'\s+', ' ', text).strip()

def preprocess_for_storage(text):
    """
    Pipeline AGRESIF untuk penyimpanan di database.
    Digunakan saat upload CSV / preprocessing data.
    Menghapus semua noise, tanda baca, karakter non-ASCII.
    """
    try:
        text = _clean_urls_mentions(text)
        text = text.lower()
        text = _normalize_informal_repetition(text)
        # Hapus karakter non-ASCII (emoji, simbol)
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        # Hapus tanda baca
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        text = _normalize_slang(text)
        text = _remove_repeated_chars(text)
        # Hapus kata 1 huruf
        words = text.split()
        text = ' '.join(w for w in words if len(w) > 1)
        text = _whitespace_clean(text)
        return text
    except Exception as e:
        logger.error(f"Error in storage preprocessing: {e}")
        return text


# ============================================================================
# PIPELINE 2: PREPROCESSING UNTUK MODEL (ringan, untuk input IndoBERT)
# ============================================================================

def preprocess_for_model(text, video_title=None):
    """
    Pipeline RINGAN khusus input ke IndoBERT.
    Preserve makna semantik — TIDAK menghapus tanda baca penting,
    TIDAK stemming, TIDAK hapus stopword.

    IndoBERT sudah di-pretrain dengan teks natural,
    jadi input harus tetap se-natural mungkin.

    Args:
        text: Teks komentar
        video_title: Judul video YouTube (opsional, untuk context-aware)

    Returns:
        Teks yang sudah dinormalisasi untuk input IndoBERT
    """
    try:
        if not text:
            return ""

        # 1. Hapus URL dan mention saja
        text = _clean_urls_mentions(text)

        # 2. Lowercase
        text = text.lower()

        # 3. Normalisasi kata ulang informal
        text = _normalize_informal_repetition(text)

        # 4. Normalisasi emoji penting ke teks (preserve sentiment signal)
        emoji_map = {
            '😂': ' lucu ', '🤣': ' lucu ', '😭': ' sedih ', '😢': ' sedih ',
            '😡': ' marah ', '🤬': ' marah ', '👍': ' bagus ', '👎': ' jelek ',
            '❤️': ' suka ', '💔': ' kecewa ', '🔥': ' mantap ', '💩': ' jelek ',
            '👏': ' tepuk tangan ', '🙏': ' terima kasih ', '😤': ' kesal ',
            '🤮': ' menjijikkan ', '😍': ' suka sekali ', '🤡': ' badut ',
            '💪': ' semangat ', '😊': ' senang ', '😔': ' sedih ', '🥰': ' sayang ',
        }
        for emoji, replacement in emoji_map.items():
            text = text.replace(emoji, replacement)

        # 5. Hapus sisa emoji/non-ascii SETELAH konversi penting
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)

        # 6. Normalisasi slang
        text = _normalize_slang(text)

        # 7. Normalkan karakter berulang
        text = _remove_repeated_chars(text)

        # 8. Whitespace cleaning
        text = _whitespace_clean(text)

        # 9. Context-aware: gabungkan dengan judul video jika ada
        if video_title:
            video_title = video_title.lower().strip()
            text = f"judul: {video_title}. komentar: {text}"

        return text
    except Exception as e:
        logger.error(f"Error in model preprocessing: {e}")
        return text


# ============================================================================
# BACKWARD COMPATIBILITY — fungsi lama tetap ada
# ============================================================================

def preprocess_comment(text):
    """Alias untuk preprocess_for_storage (backward compatibility)."""
    return preprocess_for_storage(text)
