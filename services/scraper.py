"""
YouTube Comment Scraper menggunakan Innertube API.
Tidak memerlukan library pihak ketiga selain `requests`.
Menggunakan endpoint internal YouTube yang sama dengan browser.
"""
import requests
import re
import json
from utils.logger import setup_logger

logger = setup_logger(__name__)

# YouTube Innertube API endpoint & client config
INNERTUBE_API_URL = 'https://www.youtube.com/youtubei/v1/next'
INNERTUBE_CLIENT = {
    'clientName': 'WEB',
    'clientVersion': '2.20240101.00.00',
    'hl': 'id',
    'gl': 'ID',
}

USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36'

def _extract_video_id(url):
    """Extract video ID dari berbagai format URL YouTube."""
    patterns = [
        r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/|youtube\.com/v/)([a-zA-Z0-9_-]{11})',
        r'([a-zA-Z0-9_-]{11})',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def _get_session_data(video_id):
    """Ambil session token, continuation token, dan judul video dari halaman YouTube."""
    session = requests.Session()
    session.headers.update({
        'User-Agent': USER_AGENT,
        'Accept-Language': 'id-ID,id;q=0.9,en-US;q=0.8,en;q=0.7',
    })
    
    url = f'https://www.youtube.com/watch?v={video_id}'
    response = session.get(url)
    
    if response.status_code != 200:
        raise Exception(f"Gagal mengakses halaman YouTube (HTTP {response.status_code})")
    
    html = response.text
    
    # Cari API key
    api_key_match = re.search(r'"INNERTUBE_API_KEY":"([^"]+)"', html)
    if not api_key_match:
        api_key_match = re.search(r'"innertubeApiKey":"([^"]+)"', html)
    
    api_key = api_key_match.group(1) if api_key_match else 'AIzaSyAO_FJ2SlqU8Q4STEHLGCilw_Y9_11qcW8'
    
    # Cari judul video
    video_title = None
    title_match = re.search(r'<title>(.*?)(?:\s*-\s*YouTube)?</title>', html)
    if title_match:
        video_title = title_match.group(1).strip()
    
    # Cari continuation token untuk komentar dari ytInitialData
    initial_data_match = re.search(r'var ytInitialData = ({.*?});</script>', html, re.DOTALL)
    if not initial_data_match:
        initial_data_match = re.search(r'window\["ytInitialData"\] = ({.*?});', html, re.DOTALL)
    
    continuation_token = None
    if initial_data_match:
        try:
            data = json.loads(initial_data_match.group(1))
            continuation_token = _find_comment_continuation(data)
            # Fallback judul dari ytInitialData
            if not video_title:
                video_title = (data.get('contents', {})
                    .get('twoColumnWatchNextResults', {})
                    .get('results', {}).get('results', {})
                    .get('contents', [{}])[0]
                    .get('videoPrimaryInfoRenderer', {})
                    .get('title', {}).get('runs', [{}])[0]
                    .get('text', ''))
        except json.JSONDecodeError:
            logger.warning("Gagal parse ytInitialData")
    
    logger.info(f"Judul video: {video_title}")
    return session, api_key, continuation_token, video_title

def _find_comment_continuation(data):
    """Cari continuation token untuk komentar dari ytInitialData."""
    try:
        # Path: contents -> twoColumnWatchNextResults -> results -> results -> contents
        contents = data.get('contents', {}).get('twoColumnWatchNextResults', {}).get('results', {}).get('results', {}).get('contents', [])
        
        for item in contents:
            # Cari itemSectionRenderer yang berisi komentar
            section = item.get('itemSectionRenderer', {})
            section_contents = section.get('contents', [])
            
            for content in section_contents:
                continuation_item = content.get('continuationItemRenderer', {})
                if continuation_item:
                    token = continuation_item.get('continuationEndpoint', {}).get('continuationCommand', {}).get('token')
                    if token:
                        return token
            
            # Alternatif: cek di sectionListRenderer
            section_list = item.get('itemSectionRenderer', {})
            continuations = section_list.get('continuations', [])
            for cont in continuations:
                next_cont = cont.get('nextContinuationData', {})
                if next_cont.get('continuation'):
                    return next_cont['continuation']
    except Exception as e:
        logger.warning(f"Error navigating data structure: {e}")
    
    return None

def _fetch_comments(session, api_key, continuation_token, limit):
    """Fetch komentar menggunakan Innertube API."""
    comments = []
    
    url = f'{INNERTUBE_API_URL}?key={api_key}'
    
    is_first_request = True
    
    while continuation_token and len(comments) < limit:
        payload = {
            'context': {
                'client': INNERTUBE_CLIENT,
            },
            'continuation': continuation_token,
        }
        
        try:
            response = session.post(
                url,
                json=payload,
                headers={
                    'Content-Type': 'application/json',
                    'User-Agent': USER_AGENT,
                }
            )
            
            if response.status_code != 200:
                logger.warning(f"Innertube API error: HTTP {response.status_code}")
                break
                
            data = response.json()
            
            # Parse response untuk komentar
            new_comments, next_token = _parse_comment_response(data)
            comments.extend(new_comments)
            
            # Jika ini request pertama dan tidak ada komentar, coba ambil sort token
            if is_first_request and not new_comments:
                sort_token = _get_sort_token(data)
                if sort_token:
                    logger.info("Menggunakan sort token untuk memuat komentar...")
                    continuation_token = sort_token
                    is_first_request = False
                    continue
                else:
                    break
            
            is_first_request = False
            
            if not next_token or not new_comments:
                break
                
            continuation_token = next_token
            logger.info(f"Terkumpul {len(comments)} komentar...")
            
        except Exception as e:
            logger.warning(f"Error fetching comments batch: {e}")
            break
    
    return comments[:limit]

def _get_sort_token(data):
    """Ambil sort token 'Teratas' dari header komentar."""
    endpoints = data.get('onResponseReceivedEndpoints', [])
    for endpoint in endpoints:
        for key in ['appendContinuationItemsAction', 'reloadContinuationItemsCommand']:
            items = endpoint.get(key, {})
            if isinstance(items, dict) and 'continuationItems' in items:
                for item in items['continuationItems']:
                    header = item.get('commentsHeaderRenderer', {})
                    if header:
                        sort_menu = header.get('sortMenu', {})
                        sub_items = sort_menu.get('sortFilterSubMenuRenderer', {}).get('subMenuItems', [])
                        # Ambil token pertama (biasanya "Teratas")
                        if sub_items:
                            token = (sub_items[0].get('serviceEndpoint', {})
                                    .get('continuationCommand', {})
                                    .get('token'))
                            if token:
                                return token
    return None

def _parse_comment_response(data):
    """Parse response dari Innertube API untuk mengekstrak komentar."""
    comments = []
    next_token = None
    
    # === METODE BARU (2025+): Komentar ada di frameworkUpdates.entityBatchUpdate.mutations ===
    mutations = (
        data.get('frameworkUpdates', {})
        .get('entityBatchUpdate', {})
        .get('mutations', [])
    )
    
    for mutation in mutations:
        payload = mutation.get('payload', {})
        comment_entity = payload.get('commentEntityPayload', {})
        
        if not comment_entity:
            continue
            
        props = comment_entity.get('properties', {})
        author_data = comment_entity.get('author', {})
        toolbar_data = comment_entity.get('toolbar', {})
        
        # Ambil teks komentar
        content_obj = props.get('content', {})
        text = content_obj.get('content', '') if isinstance(content_obj, dict) else str(content_obj)
        
        # Ambil author
        author = author_data.get('displayName', 'Unknown')
        
        # Ambil jumlah like
        like_count = toolbar_data.get('likeCountNotliked', '0')
        
        # Ambil waktu
        published_time = props.get('publishedTime', '')
        
        if text.strip():
            comments.append({
                'author': author.lstrip('@'),
                'text': text.strip(),
                'time': published_time,
                'votes': like_count,
            })
    
    # === Cari next continuation token dari onResponseReceivedEndpoints ===
    endpoints = data.get('onResponseReceivedEndpoints', [])
    for endpoint in endpoints:
        for key in ['appendContinuationItemsAction', 'reloadContinuationItemsCommand']:
            items = endpoint.get(key, {})
            if isinstance(items, dict) and 'continuationItems' in items:
                for item in items['continuationItems']:
                    cont_item = item.get('continuationItemRenderer', {})
                    if cont_item:
                        # Check continuationEndpoint
                        token = (cont_item.get('continuationEndpoint', {})
                                .get('continuationCommand', {})
                                .get('token'))
                        if token:
                            next_token = token
                        
                        # Check button renderer
                        btn_token = (cont_item.get('button', {})
                                    .get('buttonRenderer', {})
                                    .get('command', {})
                                    .get('continuationCommand', {})
                                    .get('token'))
                        if btn_token:
                            next_token = btn_token
    
    return comments, next_token


def scrape_youtube_comments(video_url, limit=100):
    """
    Scrape komentar dari video YouTube.
    Menggunakan YouTube Innertube API secara langsung (tanpa library pihak ketiga).
    
    Args:
        video_url: URL video YouTube (mendukung berbagai format)
        limit: Jumlah maksimal komentar yang diambil
    
    Returns:
        Tuple (comments_list, video_title)
        comments_list: List of dict dengan keys: author, text, time, votes
        video_title: Judul video YouTube
    """
    video_id = _extract_video_id(video_url)
    if not video_id:
        raise Exception(f"Tidak dapat mengekstrak Video ID dari URL: {video_url}")
    
    logger.info(f"Memulai scraping komentar untuk video: {video_id}")
    
    # Step 1: Ambil session, continuation token, dan judul video
    session, api_key, continuation_token, video_title = _get_session_data(video_id)
    
    if not continuation_token:
        logger.warning("Tidak dapat menemukan token komentar. Video mungkin menonaktifkan komentar.")
        return [], video_title
    
    logger.info(f"Berhasil mendapatkan session. Mengambil komentar...")
    
    # Step 2: Fetch komentar menggunakan token
    comments = _fetch_comments(session, api_key, continuation_token, limit)
    
    logger.info(f"Berhasil scraping {len(comments)} komentar dari {video_url}")
    return comments, video_title
