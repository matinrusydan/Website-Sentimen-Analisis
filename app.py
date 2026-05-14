from flask import Flask, render_template, request, jsonify, send_file
from extensions import db
from config import Config
from utils.logger import setup_logger
import os

logger = setup_logger(__name__)

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    
    db.init_app(app)
    
    with app.app_context():
        # Register Routes
        from services import database_service # Load models
        db.create_all()
        
        @app.route('/')
        def index():
            return render_template('index.html')
            
        @app.route('/dashboard')
        def dashboard():
            return render_template('dashboard.html')
            
        @app.route('/evaluasi')
        def evaluasi():
            return render_template('evaluasi.html')

        @app.route('/api/videos', methods=['GET'])
        def api_videos():
            """List semua video yang sudah di-scraping, dengan status preprocessing."""
            try:
                from services.database_service import Comment
                from sqlalchemy import func
                
                videos = db.session.query(
                    Comment.video_url,
                    Comment.video_title,
                    func.count(Comment.id).label('total'),
                    func.count(Comment.text_clean).label('cleaned'),
                ).filter(
                    Comment.video_url != 'uploaded_csv'
                ).group_by(Comment.video_url).all()
                
                result = []
                for v in videos:
                    result.append({
                        'video_url': v.video_url,
                        'video_title': v.video_title or 'Tidak diketahui',
                        'total_comments': v.total,
                        'cleaned_comments': v.cleaned,
                        'is_preprocessed': v.cleaned == v.total,
                    })
                
                return jsonify({'success': True, 'videos': result})
            except Exception as e:
                return jsonify({'success': False, 'message': str(e)})
            
        import threading
        import uuid
        
        # Dictionary global untuk menyimpan status task
        TASKS = {}
        
        def background_scrape_task(app_instance, task_id, video_url, limit):
            with app_instance.app_context():
                try:
                    def add_log(msg):
                        TASKS[task_id]['logs'].append(msg)
                        logger.info(f"[Task {task_id}] {msg}")

                    add_log("Mempersiapkan modul dan library...")
                    from services.scraper import scrape_youtube_comments
                    from services.database_service import save_comment
                    add_log(f"Mulai scraping YouTube untuk URL: {video_url} (Limit: {limit} komentar)...")
                    raw_comments, video_title = scrape_youtube_comments(video_url, limit=limit)
                    
                    if not raw_comments:
                        TASKS[task_id]['status'] = 'error'
                        add_log("Gagal: Tidak ada komentar yang ditemukan atau video tidak valid.")
                        return
                    
                    if video_title:
                        add_log(f"Judul video: {video_title}")
                    
                    total = len(raw_comments)
                    add_log(f"Berhasil mendapatkan {total} komentar. Menyimpan ke database...")
                    
                    saved_count = 0
                    
                    for i, c in enumerate(raw_comments):
                        if i % max(1, total // 10) == 0 or i == total - 1:
                            add_log(f"Menyimpan komentar {i+1} dari {total}...")
                            
                        text_ori = c['text']
                        author = c['author']
                        
                        # Save RAW to DB with video title for context-aware prediction
                        save_comment(video_url, author, text_ori, video_title=video_title)
                        saved_count += 1
                    
                    add_log("Proses Scraping selesai 100%! Data mentah berhasil disimpan.")
                    TASKS[task_id]['result'] = {
                        'success': True, 
                        'message': f'Berhasil memproses {saved_count} komentar mentah.',
                        'sentiments': None,
                        'wordcloud': None,
                        'topics': None,
                        'video_url': video_url,
                        'video_title': video_title
                    }
                    TASKS[task_id]['status'] = 'completed'
                    
                except Exception as e:
                    logger.error(f"Error in background task {task_id}: {e}")
                    TASKS[task_id]['status'] = 'error'
                    TASKS[task_id]['logs'].append(f"Terjadi kesalahan sistem: {str(e)}")

        @app.route('/api/scrape', methods=['POST'])
        def api_scrape():
            data = request.get_json()
            video_url = data.get('url')
            limit = int(data.get('limit', 100))
            
            if not video_url:
                return jsonify({'success': False, 'message': 'URL YouTube diperlukan'})
                
            task_id = str(uuid.uuid4())
            TASKS[task_id] = {
                'status': 'running',
                'logs': ['Menerima permintaan analisis...'],
                'result': None
            }
            
            # Start background thread
            thread = threading.Thread(target=background_scrape_task, args=(app, task_id, video_url, limit))
            thread.daemon = True
            thread.start()
            
            return jsonify({'success': True, 'task_id': task_id})

        @app.route('/api/task_status/<task_id>', methods=['GET'])
        def task_status(task_id):
            task = TASKS.get(task_id)
            if not task:
                return jsonify({'status': 'not_found'})
            return jsonify(task)

        @app.route('/api/export', methods=['GET'])
        def api_export():
            video_url = request.args.get('url')
            if not video_url:
                return jsonify({'success': False, 'message': 'URL diperlukan'})
                
            from services.export_service import export_comments_to_csv
            filepath = export_comments_to_csv(video_url)
            
            if filepath and os.path.exists(filepath):
                return send_file(filepath, as_attachment=True)
            return jsonify({'success': False, 'message': 'Gagal mengekspor data'})

        @app.route('/preprocessing')
        def preprocessing_page():
            return render_template('preprocessing.html')

        @app.route('/api/upload_clean', methods=['POST'])
        def api_upload_clean():
            if 'file' not in request.files:
                return jsonify({'success': False, 'error': 'Tidak ada file diupload'})
            
            file = request.files['file']
            if file.filename == '':
                return jsonify({'success': False, 'error': 'Tidak ada file dipilih'})
                
            if not file.filename.endswith('.csv'):
                return jsonify({'success': False, 'error': 'File harus format CSV'})

            try:
                import pandas as pd
                from services.preprocessing import preprocess_comment
                from services.database_service import Comment
                
                # Baca CSV
                df = pd.read_csv(file)
                
                # Coba cari nama kolom teks yang cocok
                text_col = None
                for col in ['text_original', 'Text', 'text', 'Komentar']:
                    if col in df.columns:
                        text_col = col
                        break
                        
                if not text_col:
                    return jsonify({'success': False, 'error': f'Kolom teks asli tidak ditemukan di CSV. Kolom yang tersedia: {", ".join(df.columns)}'})

                # Hapus hanya data CSV lama (bukan data scraping)
                Comment.query.filter_by(video_url='uploaded_csv').delete()
                db.session.commit()

                processed_count = 0
                for index, row in df.iterrows():
                    text_ori = str(row[text_col])
                    author = str(row.get('author', row.get('Author', 'Unknown')))
                    video_url = str(row.get('Video URL', 'uploaded_csv'))
                    
                    # Preprocessing: hanya membersihkan teks (tanpa pelabelan)
                    text_clean = preprocess_comment(text_ori)
                    if not text_clean or len(text_clean) < 2:
                        continue
                    
                    # Simpan ke DB tanpa label (label akan diberikan oleh IndoBERT nanti)
                    comment = Comment(video_url=video_url, author=author, text_original=text_ori)
                    comment.text_clean = text_clean
                    db.session.add(comment)
                    processed_count += 1
                
                db.session.commit()
                
                task_id = "clean_export"
                
                return jsonify({
                    'success': True, 
                    'message': f'Berhasil memproses dan membersihkan {processed_count} data! (Tanpa pelabelan)',
                    'task_id': task_id
                })
                
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})

        @app.route('/api/preprocess_db', methods=['POST'])
        def api_preprocess_db():
            """Preprocessing data scraping per video."""
            try:
                from services.preprocessing import preprocess_for_storage
                from services.database_service import Comment
                
                data = request.get_json() or {}
                video_url = data.get('video_url')
                
                # Filter per video jika diberikan
                query = Comment.query.filter(Comment.text_clean.is_(None))
                if video_url:
                    query = query.filter_by(video_url=video_url)
                
                comments = query.all()
                
                if not comments:
                    return jsonify({'success': False, 'message': 'Semua data sudah di-preprocessing.'})
                
                processed_count = 0
                skipped = 0
                for c in comments:
                    text_clean = preprocess_for_storage(c.text_original)
                    if text_clean and len(text_clean) >= 2:
                        c.text_clean = text_clean
                        c.sentiment_label = None
                        c.sentiment_score = None
                        processed_count += 1
                    else:
                        skipped += 1
                
                db.session.commit()
                
                return jsonify({
                    'success': True,
                    'message': f'Berhasil preprocessing {processed_count} komentar. ({skipped} dilewati)',
                    'processed': processed_count,
                    'skipped': skipped,
                })
            except Exception as e:
                logger.error(f"Error preprocessing DB: {e}")
                return jsonify({'success': False, 'message': str(e)})

        @app.route('/finetuning')
        def finetuning_page():
            return render_template('finetuning.html')

        @app.route('/api/train_start', methods=['POST'])
        def api_train_start():
            task_id = "training_task"
            
            # Jika sudah berjalan, jangan ditimpa (kecuali error/completed)
            if task_id in TASKS and TASKS[task_id]['status'] == 'running':
                return jsonify({'success': False, 'message': 'Pelatihan sedang berjalan di latar belakang.'})
                
            TASKS[task_id] = {
                'status': 'running',
                'logs': ['Menerima instruksi pelatihan...'],
                'result': None
            }
            
            def background_train_task(app_instance):
                import subprocess
                import sys
                import os
                
                with app_instance.app_context():
                    try:
                        def add_log(msg):
                            TASKS[task_id]['logs'].append(msg)
                            logger.info(f"[Train] {msg}")

                        add_log("> Memulai PyTorch DataLoader...")
                        
                        python_exe = sys.executable
                        script_path = os.path.join(Config.BASE_DIR, 'train_model.py')
                        
                        env = os.environ.copy()
                        env['PYTHONIOENCODING'] = 'utf-8'
                        env['PYTHONUNBUFFERED'] = '1'
                        
                        # Use Popen to stream output (-u = unbuffered)
                        process = subprocess.Popen(
                            [python_exe, '-u', script_path],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True,
                            encoding='utf-8',
                            cwd=Config.BASE_DIR,
                            env=env,
                            bufsize=1, # Line buffered
                            universal_newlines=True
                        )
                        
                        for line in process.stdout:
                            add_log(line.strip())
                            
                        process.wait()
                        
                        if process.returncode == 0:
                            TASKS[task_id]['status'] = 'completed'
                            TASKS[task_id]['result'] = {'success': True, 'message': 'Fine-Tuning selesai 100%!'}
                        else:
                            TASKS[task_id]['status'] = 'error'
                            add_log(f"Proses berhenti dengan kode error: {process.returncode}")
                            
                    except Exception as e:
                        logger.error(f"Error in training background task: {e}")
                        TASKS[task_id]['status'] = 'error'
                        TASKS[task_id]['logs'].append(f"Terjadi kesalahan sistem: {str(e)}")

            import threading
            thread = threading.Thread(target=background_train_task, args=(app,))
            thread.daemon = True
            thread.start()
            
            return jsonify({'success': True, 'task_id': task_id})

        @app.route('/api/export_clean', methods=['GET'])
        def api_export_clean():
            from services.database_service import Comment
            import pandas as pd
            import os
            
            # Export ONLY cleaned comments (yang sudah melalui preprocessing)
            comments = Comment.query.filter(Comment.text_clean.isnot(None)).all()
            if not comments:
                return "Belum ada data bersih", 404
                
            data = []
            for c in comments:
                data.append({
                    'Original_Text': c.text_original,
                    'Cleaned_Text': c.text_clean,
                })
                
            df = pd.DataFrame(data)
            filepath = os.path.join(Config.PROCESSED_DATA_DIR, 'cleaned_dataset.csv')
            os.makedirs(Config.PROCESSED_DATA_DIR, exist_ok=True)
            df.to_csv(filepath, index=False, sep=';')
            
            return send_file(filepath, as_attachment=True)

        @app.route('/api/evaluate', methods=['POST'])
        def api_evaluate():
            """Evaluasi sentimen per video. Wajib kirim video_url."""
            try:
                from services.database_service import Comment
                from services.sentiment import predict_sentiment, init_model
                
                data = request.get_json() or {}
                video_url = data.get('video_url')
                
                if not video_url:
                    return jsonify({'success': False, 'message': 'Pilih video yang ingin dievaluasi terlebih dahulu.'})
                
                # Ambil komentar HANYA dari video yang dipilih
                comments = Comment.query.filter(
                    Comment.video_url == video_url,
                    Comment.text_clean.isnot(None)
                ).all()

                if not comments:
                    return jsonify({'success': False, 'message': 'Data bersih belum ada untuk video ini. Jalankan Preprocessing terlebih dahulu.'})
                    
                init_model()
                
                results = []
                y_pred = []
                confidence_scores = []
                ambiguous_cases = []
                video_title = None
                
                for c in comments:
                    text = c.text_original if c.text_original else c.text_clean
                    vt = getattr(c, 'video_title', None)
                    if vt and not video_title:
                        video_title = vt
                    
                    pred_label, pred_score, probs = predict_sentiment(text, video_title=vt)
                    c.sentiment_label = pred_label
                    c.sentiment_score = pred_score
                    
                    predicted = pred_label.strip().lower()
                    y_pred.append(predicted)
                    confidence_scores.append(pred_score)
                    
                    result_item = {
                        'text': c.text_clean or text,
                        'original': text[:100],
                        'predicted': predicted,
                        'score': round(pred_score, 4),
                        'probs': probs,
                    }
                    results.append(result_item)
                    
                    if pred_score < 0.7:
                        ambiguous_cases.append(result_item)
                    
                db.session.commit()
                
                pos_count = y_pred.count('positive')
                neg_count = y_pred.count('negative')
                neu_count = y_pred.count('neutral')
                total = len(y_pred)
                
                distribution = {
                    'positive': round((pos_count/total)*100, 1) if total > 0 else 0,
                    'negative': round((neg_count/total)*100, 1) if total > 0 else 0,
                    'neutral': round((neu_count/total)*100, 1) if total > 0 else 0
                }
                
                avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
                high_conf = sum(1 for s in confidence_scores if s >= 0.9)
                med_conf = sum(1 for s in confidence_scores if 0.7 <= s < 0.9)
                low_conf = sum(1 for s in confidence_scores if s < 0.7)
                
                confidence_analysis = {
                    'average': round(avg_confidence, 4),
                    'high_confidence': high_conf,
                    'medium_confidence': med_conf,
                    'low_confidence': low_conf,
                    'total': total,
                }
                
                return jsonify({
                    'success': True,
                    'distribution': distribution,
                    'total_sample': total,
                    'confidence_analysis': confidence_analysis,
                    'ambiguous_count': len(ambiguous_cases),
                    'ambiguous_cases': ambiguous_cases[:20],
                    'video_title': video_title or 'Tidak diketahui',
                    'video_url': video_url,
                    'results': results
                })
            except Exception as e:
                logger.error(f"Error evaluation: {e}")
                return jsonify({'success': False, 'message': str(e)})

        @app.route('/api/export_video', methods=['GET'])
        def api_export_video():
            """Export CSV per video."""
            try:
                import csv, io
                from services.database_service import Comment
                
                video_url = request.args.get('video_url')
                if not video_url:
                    return jsonify({'success': False, 'message': 'video_url diperlukan'})
                
                comments = Comment.query.filter_by(video_url=video_url).all()
                if not comments:
                    return jsonify({'success': False, 'message': 'Tidak ada data'})
                
                output = io.StringIO()
                writer = csv.writer(output)
                writer.writerow(['No', 'Author', 'Text Original', 'Text Clean', 'Video Title', 'Sentiment', 'Confidence'])
                
                for i, c in enumerate(comments, 1):
                    writer.writerow([
                        i, c.author, c.text_original, c.text_clean or '',
                        c.video_title or '', c.sentiment_label or '', c.sentiment_score or ''
                    ])
                
                output.seek(0)
                
                from flask import Response
                return Response(
                    output.getvalue(),
                    mimetype='text/csv',
                    headers={'Content-Disposition': 'attachment; filename=hasil_evaluasi.csv'}
                )
            except Exception as e:
                return jsonify({'success': False, 'message': str(e)})

        @app.route('/api/metrics', methods=['GET'])
        def api_metrics():
            try:
                metrics_path = os.path.join(Config.BASE_DIR, 'results', 'model_metrics.json')
                if not os.path.exists(metrics_path):
                    return jsonify({'success': False, 'message': 'Belum ada model yang dilatih atau metrik belum tersedia.'})
                    
                import json
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
                    
                return jsonify({
                    'success': True,
                    'metrics': metrics
                })
            except Exception as e:
                logger.error(f"Error fetching metrics: {e}")
                return jsonify({'success': False, 'message': str(e)})

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)
