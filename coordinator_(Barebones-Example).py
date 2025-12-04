import socket
import threading
import subprocess
import os
import sys
import time
import json
import shutil
import uuid
import hashlib
import sqlite3
import jwt
import secrets
from pathlib import Path
from datetime import datetime, timedelta
from flask import Flask, render_template_string, request, jsonify, send_file
from werkzeug.utils import secure_filename
from werkzeug.security import check_password_hash, generate_password_hash
import zipfile
import mimetypes
from functools import wraps

# ===== Configuration =====
COORDINATOR_HOST = '0.0.0.0'
COORDINATOR_NETWORK_PORT = 5555
COORDINATOR_WEB_PORT = 5000
COORDINATOR_UPLOAD_PORT = 5556  # New port for frame uploads
UPLOAD_FOLDER = Path('./coordinator_uploads')
OUTPUT_BASE = Path('./coordinator_output')
DATABASE_FILE = Path('./render_farm.db')
SECRET_KEY = secrets.token_urlsafe(32)  # Change this in production!

# Ensure directories exist
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

# ===== Flask App =====
app = Flask(__name__)
app.config['SECRET_KEY'] = SECRET_KEY
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024 * 1024  # 10GB max

# ===== Database Connection Pool =====
def get_db():
    """Get database connection from pool"""
    conn = sqlite3.connect(DATABASE_FILE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize database if needed"""
    if not DATABASE_FILE.exists():
        print("❌ Database not found. Run database_setup.py first!")
        sys.exit(1)

# ===== Authentication Decorators =====
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        
        if 'X-API-Token' in request.headers:
            token = request.headers['X-API-Token']
        
        if not token:
            return jsonify({'error': 'Token is missing'}), 401
        
        try:
            conn = get_db()
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM users WHERE api_token = ?', (token,))
            user = cursor.fetchone()
            conn.close()
            
            if not user:
                return jsonify({'error': 'Invalid token'}), 401
        except:
            return jsonify({'error': 'Token verification failed'}), 401
        
        return f(*args, **kwargs)
    return decorated

def worker_token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        
        if 'X-Worker-Token' in request.headers:
            token = request.headers['X-Worker-Token']
        
        if not token:
            return jsonify({'error': 'Worker token is missing'}), 401
        
        try:
            conn = get_db()
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM workers WHERE api_token = ?', (token,))
            worker = cursor.fetchone()
            conn.close()
            
            if not worker:
                return jsonify({'error': 'Invalid worker token'}), 401
            
            # Add worker info to request context
            request.worker_id = worker['id']
            request.worker_uuid = worker['worker_id']
        except:
            return jsonify({'error': 'Worker token verification failed'}), 401
        
        return f(*args, **kwargs)
    return decorated

# ===== Database Models =====
class DatabaseModel:
    @staticmethod
    def log_render(job_id, frame_id, worker_id, level, message):
        """Log render activity"""
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO render_logs (job_id, frame_id, worker_id, log_level, message)
            VALUES (?, ?, ?, ?, ?)
        ''', (job_id, frame_id, worker_id, level, message))
        conn.commit()
        conn.close()

    @staticmethod
    def update_worker_heartbeat(worker_id):
        """Update worker last heartbeat"""
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE workers 
            SET last_heartbeat = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP
            WHERE worker_id = ?
        ''', (worker_id,))
        conn.commit()
        conn.close()

    @staticmethod
    def get_available_workers():
        """Get all available workers sorted by load and capability"""
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT w.*, 
                   (w.render_speed_score * (1.0 - w.current_load)) as priority_score
            FROM workers w
            WHERE w.status IN ('idle', 'busy')
            AND (w.last_heartbeat IS NULL OR 
                 datetime(w.last_heartbeat) > datetime('now', '-5 minutes'))
            ORDER BY priority_score DESC
        ''')
        workers = cursor.fetchall()
        conn.close()
        
        # Parse GPU info from JSON
        result = []
        for worker in workers:
            worker_dict = dict(worker)
            if worker_dict['gpu_info']:
                worker_dict['gpu_info'] = json.loads(worker_dict['gpu_info'])
            result.append(worker_dict)
        
        return result

    @staticmethod
    def assign_frame_to_worker(job_id, frame_number, worker_id):
        """Assign a frame to a worker"""
        conn = get_db()
        cursor = conn.cursor()
        
        try:
            # Start transaction
            cursor.execute('BEGIN TRANSACTION')
            
            # Get frame details
            cursor.execute('''
                SELECT f.id, j.render_settings
                FROM frames f
                JOIN jobs j ON f.job_id = j.id
                WHERE f.job_id = ? AND f.frame_number = ?
            ''', (job_id, frame_number))
            frame = cursor.fetchone()
            
            if not frame:
                raise ValueError("Frame not found")
            
            frame_id = frame['id']
            
            # Update frame status
            cursor.execute('''
                UPDATE frames 
                SET worker_id = ?, 
                    status = 'assigned',
                    assigned_at = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (worker_id, frame_id))
            
            # Update worker status and load
            cursor.execute('''
                UPDATE workers 
                SET status = 'busy',
                    current_load = current_load + 0.1
                WHERE id = ?
            ''', (worker_id,))
            
            # Log the assignment
            cursor.execute('''
                INSERT INTO render_logs (job_id, frame_id, worker_id, log_level, message)
                VALUES (?, ?, ?, 'info', ?)
            ''', (job_id, frame_id, worker_id, f"Frame {frame_number} assigned to worker {worker_id}"))
            
            conn.commit()
            
            # Get blend file info
            cursor.execute('''
                SELECT j.job_uuid, j.blend_filename, j.output_format
                FROM jobs j
                WHERE j.id = ?
            ''', (job_id,))
            job = cursor.fetchone()
            
            return {
                'frame_id': frame_id,
                'job_uuid': job['job_uuid'],
                'blend_filename': job['blend_filename'],
                'output_format': job['output_format']
            }
            
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    @staticmethod
    def complete_frame(frame_id, worker_id, file_size=None, render_time=None, gpu_used=None):
        """Mark frame as completed"""
        conn = get_db()
        cursor = conn.cursor()
        
        try:
            cursor.execute('BEGIN TRANSACTION')
            
            # Update frame status
            cursor.execute('''
                UPDATE frames 
                SET status = 'completed',
                    completed_at = CURRENT_TIMESTAMP,
                    render_time_seconds = ?,
                    file_size = ?,
                    gpu_used = ?
                WHERE id = ?
            ''', (render_time, file_size, gpu_used, frame_id))
            
            # Update worker load
            cursor.execute('''
                UPDATE workers 
                SET current_load = MAX(0, current_load - 0.1),
                    status = CASE 
                        WHEN current_load <= 0.1 THEN 'idle'
                        ELSE 'busy'
                    END
                WHERE id = ?
            ''', (worker_id,))
            
            # Check if job is complete
            cursor.execute('''
                SELECT j.id, 
                       COUNT(CASE WHEN f.status = 'completed' THEN 1 END) as completed,
                       COUNT(*) as total
                FROM jobs j
                JOIN frames f ON j.id = f.job_id
                WHERE j.id = (SELECT job_id FROM frames WHERE id = ?)
                GROUP BY j.id
            ''', (frame_id,))
            job_status = cursor.fetchone()
            
            if job_status and job_status['completed'] == job_status['total']:
                cursor.execute('''
                    UPDATE jobs 
                    SET status = 'completed',
                        completed_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (job_status['id'],))
            
            cursor.execute('''
                INSERT INTO render_logs (frame_id, worker_id, log_level, message)
                VALUES (?, ?, 'info', 'Frame completed successfully')
            ''', (frame_id, worker_id))
            
            conn.commit()
            return True
            
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

# ===== Load Balancer =====
class LoadBalancer:
    def __init__(self):
        self.worker_scores = {}
    
    def calculate_worker_score(self, worker):
        """Calculate score for a worker based on multiple factors"""
        score = 0
        
        # GPU score (most important)
        if worker.get('gpu_info'):
            gpu_info = worker['gpu_info']
            
            # VRAM score
            vram_gb = gpu_info.get('vram_gb', 0)
            score += vram_gb * 10
            
            # CUDA/OptiX support
            if gpu_info.get('cuda', False):
                score += 50
            if gpu_info.get('optix', False):
                score += 100
            if gpu_info.get('metal', False):
                score += 75
            
            # Number of GPUs
            score += gpu_info.get('gpu_count', 1) * 25
        
        # CPU cores
        score += worker.get('cpu_cores', 4) * 2
        
        # RAM
        score += worker.get('total_ram_gb', 8) * 1
        
        # Current load penalty
        load = worker.get('current_load', 0)
        score *= (1.0 - load * 0.5)  # 50% penalty at full load
        
        # Render speed multiplier
        score *= worker.get('render_speed_score', 1.0)
        
        return score
    
    def select_best_worker(self, gpu_requirements=None):
        """Select the best worker for a task"""
        workers = DatabaseModel.get_available_workers()
        
        if not workers:
            return None
        
        # Filter by GPU requirements if specified
        if gpu_requirements:
            filtered_workers = []
            for worker in workers:
                if self.worker_meets_requirements(worker, gpu_requirements):
                    filtered_workers.append(worker)
            
            if filtered_workers:
                workers = filtered_workers
        
        # Calculate scores and select best
        best_worker = None
        best_score = -1
        
        for worker in workers:
            score = self.calculate_worker_score(worker)
            if score > best_score:
                best_score = score
                best_worker = worker
        
        return best_worker
    
    def worker_meets_requirements(self, worker, requirements):
        """Check if worker meets GPU requirements"""
        if not worker.get('gpu_info'):
            return False
        
        gpu_info = worker['gpu_info']
        
        # Minimum VRAM
        min_vram = requirements.get('min_vram_gb', 0)
        if gpu_info.get('vram_gb', 0) < min_vram:
            return False
        
        # GPU type preference
        prefer_cuda = requirements.get('prefer_cuda', False)
        prefer_optix = requirements.get('prefer_optix', False)
        prefer_metal = requirements.get('prefer_metal', False)
        
        if prefer_cuda and not gpu_info.get('cuda', False):
            return False
        if prefer_optix and not gpu_info.get('optix', False):
            return False
        if prefer_metal and not gpu_info.get('metal', False):
            return False
        
        return True

# ===== Worker Network Server =====
class WorkerServer(threading.Thread):
    def __init__(self, load_balancer):
        super().__init__()
        self.daemon = True
        self.load_balancer = load_balancer
        self.stop_event = threading.Event()
    
    def handle_worker_connection(self, sock_conn, addr):
        """Handle authenticated worker connection"""
        worker_id = None
        worker_db_id = None
        worker_token = None
        
        try:
            sock_conn.settimeout(30.0)
            
            # Worker must authenticate first
            auth_data = sock_conn.recv(1024).decode('utf-8').strip()
            if auth_data.startswith('AUTH:'):
                worker_token = auth_data.split(':', 1)[1]
                
                # Verify worker token
                db_conn = get_db()
                cursor = db_conn.cursor()
                cursor.execute('SELECT * FROM workers WHERE api_token = ?', (worker_token,))
                worker = cursor.fetchone()
                db_conn.close()
                
                if worker:
                    worker_db_id = worker['id']
                    worker_id = worker['worker_id']
                    print(f"✓ Worker authenticated: {worker_id} from {addr[0]}")
                    sock_conn.send(b'AUTH_OK\n')
                    
                    # Update worker IP and heartbeat
                    db_conn = get_db()
                    cursor = db_conn.cursor()
                    cursor.execute('''
                        UPDATE workers 
                        SET ip_address = ?, 
                            last_heartbeat = CURRENT_TIMESTAMP,
                            status = 'idle'
                        WHERE worker_id = ?
                    ''', (addr[0], worker_id))
                    db_conn.commit()
                    db_conn.close()
                else:
                    print(f"✗ Authentication failed for token: {worker_token[:10]}...")
                    sock_conn.send(b'AUTH_FAILED\n')
                    return
            else:
                print(f"✗ No AUTH header received: {auth_data}")
                sock_conn.send(b'NO_AUTH\n')
                return
            
            # Main communication loop - SIMPLIFIED PROTOCOL
            while not self.stop_event.is_set():
                try:
                    # Send READY? to worker
                    sock_conn.send(b'READY?\n')
                    
                    # Wait for worker response
                    response = sock_conn.recv(1024).decode('utf-8').strip()
                    
                    if response == 'REQUEST_TASK':
                        # Get next task for this worker
                        task = self.get_next_task_for_worker(worker_id, worker_db_id)
                        
                        if task:
                            # Send task to worker
                            if task.get('use_local_path'):
                                msg = f"TASK|{task['job_uuid']}|{task['frame']}|{task['output_format']}|{task['blend_filename']}|LOCAL|{task['blend_file_path']}|{task['frame_id']}\n"
                            else:
                                msg = f"TASK|{task['job_uuid']}|{task['frame']}|{task['output_format']}|{task['blend_filename']}|REMOTE|{task['frame_id']}\n"
                            
                            print(f"Sending task: {msg.strip()}")
                            sock_conn.send(msg.encode('utf-8'))
                            
                            # Wait for result with a long timeout for rendering
                            sock_conn.settimeout(14400)  # 4 hours for rendering
                            result = sock_conn.recv(1024).decode('utf-8').strip()
                            sock_conn.settimeout(30.0)  # Reset to normal timeout
                            
                            if result.startswith('COMPLETE|'):
                                parts = result.split('|')
                                frame_id = int(parts[1])
                                render_time = float(parts[2]) if len(parts) > 2 else None
                                gpu_used = parts[3] if len(parts) > 3 else None
                                
                                # Mark frame as completed
                                DatabaseModel.complete_frame(
                                    frame_id, 
                                    worker_db_id, 
                                    render_time=render_time,
                                    gpu_used=gpu_used
                                )
                                print(f"✓ Worker {worker_id} completed frame {task['frame']}")
                                
                                # Tell worker to upload the frame
                                sock_conn.send(b'UPLOAD_NOW\n')
                                
                            elif result.startswith('FAILED|'):
                                parts = result.split('|', 2)
                                frame_id = int(parts[1])
                                error = parts[2]
                                
                                # Update frame as failed
                                db_conn = get_db()
                                cursor = db_conn.cursor()
                                cursor.execute('''
                                    UPDATE frames 
                                    SET status = 'failed',
                                        error_message = ?,
                                        retry_count = retry_count + 1
                                    WHERE id = ?
                                ''', (error[:500], frame_id))
                                db_conn.commit()
                                db_conn.close()
                                
                                print(f"✗ Worker {worker_id} failed frame: {error[:100]}")
                        else:
                            sock_conn.send(b'NO_TASK\n')
                            # Wait a bit before asking again
                            time.sleep(2)
                            
                    elif response == 'HEARTBEAT':
                        DatabaseModel.update_worker_heartbeat(worker_id)
                        sock_conn.send(b'ALIVE\n')
                        
                    elif response.startswith('UPDATE_STATS'):
                        # Worker is sending stats - ignore or handle it
                        print(f"Worker {worker_id} sent stats update - ignoring in new protocol")
                        sock_conn.send(b'STATS_UPDATED\n')  # Or just ignore
    
                    elif response == 'DISCONNECT':
                        print(f"Worker {worker_id} disconnecting")
                        break
                        
                    else:
                        print(f"Unknown message from {worker_id}: {response}")
                        break
                        
                except socket.timeout:
                    print(f"Worker {worker_id} timeout - sending heartbeat check")
                    try:
                        sock_conn.send(b'HEARTBEAT_CHECK\n')
                        alive_response = sock_conn.recv(1024).decode('utf-8').strip()
                        if alive_response != 'ALIVE':
                            print(f"Worker {worker_id} failed heartbeat check")
                            break
                    except:
                        print(f"Worker {worker_id} connection lost")
                        break
                        
                except Exception as e:
                    print(f"Error with worker {worker_id}: {e}")
                    break
        
        except Exception as e:
            print(f"Connection error: {e}")
        finally:
            # Mark worker as offline
            if worker_id:
                db_conn = get_db()
                cursor = db_conn.cursor()
                cursor.execute('UPDATE workers SET status = "offline" WHERE worker_id = ?', (worker_id,))
                db_conn.commit()
                db_conn.close()
            
            try:
                sock_conn.close()
            except:
                pass
            
    def get_next_task_for_worker(self, worker_uuid, worker_db_id):
        """Get next pending frame for a worker using database"""
        db_conn = get_db()
        cursor = db_conn.cursor()
        
        try:
            # Find a pending frame that needs to be rendered
            cursor.execute('''
                SELECT f.id as frame_id, f.frame_number, j.job_uuid, j.blend_filename, 
                       j.output_format, j.id as job_id
                FROM frames f
                JOIN jobs j ON f.job_id = j.id
                WHERE f.status = 'pending' 
                AND j.status = 'processing'
                AND f.retry_count < 3
                ORDER BY j.priority DESC, f.frame_number
                LIMIT 1
            ''')
            
            frame = cursor.fetchone()
            
            if not frame:
                return None
            
            frame_id = frame['frame_id']
            frame_number = frame['frame_number']
            job_uuid = frame['job_uuid']
            
            # Assign this frame to the worker
            cursor.execute('''
                UPDATE frames 
                SET worker_id = ?, 
                    status = 'assigned',
                    assigned_at = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (worker_db_id, frame_id))
            
            # Update worker status
            cursor.execute('''
                UPDATE workers 
                SET status = 'busy',
                    current_load = current_load + 0.1
                WHERE id = ?
            ''', (worker_db_id,))
            
            # Log the assignment
            cursor.execute('''
                INSERT INTO render_logs (job_id, frame_id, worker_id, log_level, message)
                VALUES (?, ?, ?, 'info', ?)
            ''', (frame['job_id'], frame_id, worker_db_id, 
                  f"Frame {frame_number} assigned to worker {worker_uuid}"))
            
            db_conn.commit()
            
            # Check if worker is on same machine (local path optimization)
            use_local_path = False
            blend_file_path = str((UPLOAD_FOLDER / frame['blend_filename']).absolute())  # Get ABSOLUTE path
            
            # Get worker IP
            cursor.execute('SELECT ip_address FROM workers WHERE id = ?', (worker_db_id,))
            worker_info = cursor.fetchone()
            
            if worker_info and worker_info['ip_address']:
                worker_ip = worker_info['ip_address']
                
                # Check if worker is on same machine as coordinator
                if worker_ip in ['127.0.0.1', 'localhost']:
                    # Check if file exists
                    blend_path = Path(blend_file_path)
                    if blend_path.exists():
                        use_local_path = True
                        print(f"✓ Worker {worker_uuid} is local, using direct file path: {blend_file_path}")
                    else:
                        print(f"✗ Local blend file not found: {blend_file_path}")
            
            return {
                'frame_id': frame_id,
                'frame': frame_number,
                'job_uuid': job_uuid,
                'blend_filename': frame['blend_filename'],
                'output_format': frame['output_format'],
                'use_local_path': use_local_path,
                'blend_file_path': blend_file_path
            }
            
        except Exception as e:
            db_conn.rollback()
            print(f"Error getting task: {e}")
            return None
        finally:
            db_conn.close()
        
    def update_worker_stats(self, worker_id, stats_json):
        """Update worker statistics"""
        try:
            stats = json.loads(stats_json)
            conn = get_db()
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE workers 
                SET gpu_info = ?,
                    cpu_cores = ?,
                    total_ram_gb = ?,
                    render_speed_score = ?,
                    current_load = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE worker_id = ?
            ''', (
                json.dumps(stats.get('gpu_info', {})),
                stats.get('cpu_cores', 4),
                stats.get('total_ram_gb', 8),
                stats.get('render_speed_score', 1.0),
                stats.get('current_load', 0),
                worker_id
            ))
            
            conn.commit()
            conn.close()
            print(f"✓ Updated stats for worker {worker_id}")
            
        except Exception as e:
            print(f"Error updating worker stats: {e}")
    
    def run(self):
        """Main server loop"""
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            server.bind((COORDINATOR_HOST, COORDINATOR_NETWORK_PORT))
            server.listen(10)
            server.settimeout(1.0)
            
            print(f"✓ Worker server listening on {COORDINATOR_HOST}:{COORDINATOR_NETWORK_PORT}")
            
            while not self.stop_event.is_set():
                try:
                    sock_conn, addr = server.accept()  # Changed variable name to sock_conn
                    client_thread = threading.Thread(
                        target=self.handle_worker_connection,
                        args=(sock_conn, addr)  # Pass sock_conn instead of conn
                    )
                    client_thread.daemon = True
                    client_thread.start()
                except socket.timeout:
                    continue
                except Exception as e:
                    if not self.stop_event.is_set():
                        print(f"Error accepting connection: {e}")
                    
        except Exception as e:
            print(f"✗ Worker server failed: {e}")
        finally:
            server.close()

# ===== Frame Upload Server =====
class FrameUploadServer(threading.Thread):
    def __init__(self):
        super().__init__()
        self.daemon = True
        self.stop_event = threading.Event()
    
    def run(self):
        """HTTP server for frame uploads"""
        from http.server import HTTPServer, BaseHTTPRequestHandler
        import cgi
        
        class FrameUploadHandler(BaseHTTPRequestHandler):
            def do_POST(self):
                # Verify worker token
                worker_token = self.headers.get('X-Worker-Token')
                if not worker_token:
                    self.send_error(401, 'Worker token required')
                    return
                
                # Verify token
                conn = get_db()
                cursor = conn.cursor()
                cursor.execute('SELECT id FROM workers WHERE api_token = ?', (worker_token,))
                worker = cursor.fetchone()
                conn.close()
                
                if not worker:
                    self.send_error(401, 'Invalid worker token')
                    return
                
                # Parse multipart form
                content_type = self.headers['content-type']
                if not content_type.startswith('multipart/form-data'):
                    self.send_error(400, 'Only multipart/form-data supported')
                    return
                
                try:
                    form = cgi.FieldStorage(
                        fp=self.rfile,
                        headers=self.headers,
                        environ={'REQUEST_METHOD': 'POST',
                                'CONTENT_TYPE': content_type}
                    )
                    
                    # Get frame ID - CONVERT TO INTEGER
                    frame_id_str = form.getvalue('frame_id')
                    if not frame_id_str:
                        self.send_error(400, 'frame_id required')
                        return
                    
                    try:
                        frame_id = int(frame_id_str)  # Convert to integer
                    except ValueError:
                        self.send_error(400, 'frame_id must be an integer')
                        return
                    
                    # Get uploaded file
                    file_item = form['frame_file']
                    if not file_item.filename:
                        self.send_error(400, 'No file uploaded')
                        return
                    
                    # Verify frame belongs to this worker
                    conn = get_db()
                    cursor = conn.cursor()
                    cursor.execute('''
                        SELECT f.id, j.job_uuid, j.output_format, f.worker_id
                        FROM frames f
                        JOIN jobs j ON f.job_id = j.id
                        WHERE f.id = ?
                    ''', (frame_id,))
                    frame = cursor.fetchone()
                    
                    if not frame or frame['worker_id'] != worker['id']:
                        self.send_error(403, 'Frame not assigned to this worker')
                        conn.close()
                        return
                    
                    # Save the file
                    job_output_dir = OUTPUT_BASE / frame['job_uuid'] / 'frames'
                    job_output_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Determine file extension
                    ext_map = {'PNG': '.png', 'JPEG': '.jpg', 'OPEN_EXR': '.exr', 'TIFF': '.tif'}
                    ext = ext_map.get(frame['output_format'], '.png')
                    
                    # Now frame_id is an integer, so the format specifier will work
                    filename = f"frame_{frame_id:04d}{ext}"
                    filepath = job_output_dir / filename
                    
                    # Save file
                    with open(filepath, 'wb') as f:
                        f.write(file_item.file.read())
                    
                    # Update frame record with file info
                    file_size = filepath.stat().st_size
                    cursor.execute('''
                        UPDATE frames 
                        SET file_path = ?, 
                            file_size = ?,
                            checksum = ?
                        WHERE id = ?
                    ''', (str(filepath), file_size, hashlib.md5(open(filepath, 'rb').read()).hexdigest(), frame_id))
                    
                    conn.commit()
                    conn.close()
                    
                    # Send success response
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({
                        'success': True,
                        'message': 'Frame uploaded successfully',
                        'file_size': file_size
                    }).encode('utf-8'))
                    
                    print(f"✓ Frame {frame_id} uploaded ({file_size/1024:.1f} KB)")
                    
                except Exception as e:
                    print(f"✗ Upload error: {e}")
                    self.send_error(500, str(e))
            
            def log_message(self, format, *args):
                # Suppress default logging
                pass
        
        try:
            server = HTTPServer((COORDINATOR_HOST, COORDINATOR_UPLOAD_PORT), FrameUploadHandler)
            print(f"✓ Frame upload server listening on {COORDINATOR_HOST}:{COORDINATOR_UPLOAD_PORT}")
            
            while not self.stop_event.is_set():
                server.handle_request()
                
        except Exception as e:
            print(f"✗ Upload server failed: {e}")

# HTML Template 
INDEX_HTML = '''
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Blender Render Farm Coordinator</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.0/font/bootstrap-icons.css">
  <style>
    body {
      background-color: #f5f7fb;
      font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    }
    
    .navbar {
      background: linear-gradient(135deg, #4361ee 0%, #3a56d4 100%);
      box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    .card {
      border: none;
      border-radius: 10px;
      box-shadow: 0 4px 12px rgba(0,0,0,0.05);
      margin-bottom: 1.5rem;
    }
    
    .card-header {
      background: linear-gradient(135deg, #2b2d42 0%, #1a1b2e 100%);
      color: white;
      border-radius: 10px 10px 0 0 !important;
      padding: 1rem 1.5rem;
    }
    
    .login-card {
      max-width: 500px;
      margin: 100px auto;
    }
    
    .job-card {
      border-left: 4px solid #4361ee;
      border-radius: 8px;
      margin-bottom: 1rem;
      padding: 1rem;
      background: white;
    }
    
    .node-card {
      border-radius: 8px;
      margin-bottom: 1rem;
      padding: 1rem;
      border: 1px solid #e9ecef;
    }
    
    .node-idle {
      border-left: 4px solid #28a745;
    }
    
    .node-busy {
      border-left: 4px solid #ffc107;
    }
    
    .node-offline {
      border-left: 4px solid #dc3545;
    }
    
    .btn-login {
      width: 100%;
      padding: 10px;
      font-size: 1.1rem;
    }
    
    .stats-card {
      background: white;
      border-radius: 10px;
      padding: 1rem;
      text-align: center;
      border-top: 4px solid #4361ee;
      margin-bottom: 1rem;
    }
    
    .stats-number {
      font-size: 2rem;
      font-weight: 700;
      color: #4361ee;
    }
  </style>
</head>
<body>
<nav class="navbar navbar-dark">
  <div class="container-fluid">
    <a class="navbar-brand" href="#">
      <i class="bi bi-cpu-fill me-2"></i>
      <span class="fw-bold">Blender Render Farm</span>
    </a>
    <div class="text-white">
      <span id="userInfo">Not logged in</span>
      <span id="nodeCount" class="ms-3">0 Nodes</span>
    </div>
  </div>
</nav>

<div class="container-fluid py-4">
  
  <!-- Login Section -->
  <div class="card login-card" id="loginSection">
    <div class="card-header text-center">
      <h4 class="mb-0"><i class="bi bi-shield-lock me-2"></i>Login Required</h4>
    </div>
    <div class="card-body p-4">
      <div class="text-center mb-4">
        <i class="bi bi-cpu-fill display-4 text-primary mb-3"></i>
        <h3>Render Farm Coordinator</h3>
        <p class="text-muted">Enter your credentials to access the dashboard</p>
      </div>
      
      <div class="mb-3">
        <label class="form-label">Username</label>
        <input type="text" class="form-control" id="loginUsername" value="admin" placeholder="Enter username">
      </div>
      
      <div class="mb-4">
        <label class="form-label">Password</label>
        <input type="password" class="form-control" id="loginPassword" value="admin" placeholder="Enter password">
      </div>
      
      <button class="btn btn-primary btn-login" onclick="login()" id="loginButton">
        <i class="bi bi-box-arrow-in-right me-2"></i>Login to Dashboard
      </button>
      
      <div class="mt-4 text-center">
        <div class="alert alert-info p-2">
          <small>
            <i class="bi bi-info-circle me-1"></i>
            <strong>Default credentials:</strong> admin / admin
          </small>
        </div>
      </div>
    </div>
  </div>

  <!-- Main Dashboard -->
  <div id="dashboardSection" style="display: none;">
    
    <!-- Stats Overview -->
    <div class="row mb-4">
      <div class="col-md-3 col-sm-6">
        <div class="stats-card">
          <div class="stats-number" id="totalJobs">0</div>
          <div class="stats-label">Total Jobs</div>
        </div>
      </div>
      <div class="col-md-3 col-sm-6">
        <div class="stats-card">
          <div class="stats-number" id="activeWorkers">0</div>
          <div class="stats-label">Active Workers</div>
        </div>
      </div>
      <div class="col-md-3 col-sm-6">
        <div class="stats-card">
          <div class="stats-number" id="completedFrames">0</div>
          <div class="stats-label">Completed Frames</div>
        </div>
      </div>
      <div class="col-md-3 col-sm-6">
        <div class="stats-card">
          <div class="stats-number" id="renderingSpeed">0</div>
          <div class="stats-label">FPS</div>
        </div>
      </div>
    </div>

    <div class="row">
      <!-- Left Column: Job Management -->
      <div class="col-lg-8">
        <!-- Create Job Card -->
        <div class="card mb-4">
          <div class="card-header">
            <h5 class="mb-0"><i class="bi bi-cloud-upload me-2"></i>Create New Render Job</h5>
          </div>
          <div class="card-body">
            <form id="uploadForm">
              <div class="row">
                <div class="col-md-6 mb-3">
                  <label class="form-label">Job Name</label>
                  <input type="text" class="form-control" id="jobName" placeholder="My Animation Render" required>
                </div>
                <div class="col-md-6 mb-3">
                  <label class="form-label">Blend File</label>
                  <input type="file" class="form-control" id="blendFile" accept=".blend" required>
                </div>
              </div>
              
              <div class="row">
                <div class="col-md-3 mb-3">
                  <label class="form-label">Start Frame</label>
                  <input type="number" class="form-control" id="startFrame" value="1" min="1" required>
                </div>
                <div class="col-md-3 mb-3">
                  <label class="form-label">End Frame</label>
                  <input type="number" class="form-control" id="endFrame" value="250" min="1" required>
                </div>
                <div class="col-md-3 mb-3">
                  <label class="form-label">Output Format</label>
                  <select class="form-select" id="outputFormat">
                    <option value="PNG">PNG</option>
                    <option value="JPEG">JPEG</option>
                    <option value="OPEN_EXR">OpenEXR</option>
                    <option value="TIFF">TIFF</option>
                  </select>
                </div>
                <div class="col-md-3 mb-3">
                  <label class="form-label">Priority</label>
                  <select class="form-select" id="priority">
                    <option value="1">Normal</option>
                    <option value="2">High</option>
                    <option value="3">Urgent</option>
                  </select>
                </div>
              </div>
              
              <button type="submit" class="btn btn-primary">
                <i class="bi bi-play-circle me-2"></i>Start Render Job
              </button>
            </form>
          </div>
        </div>

        <!-- Active Jobs Card -->
        <div class="card">
          <div class="card-header">
            <h5 class="mb-0"><i class="bi bi-list-task me-2"></i>Render Jobs</h5>
          </div>
          <div class="card-body">
            <div id="jobsList">
              <div class="text-center py-5">
                <i class="bi bi-inboxes display-4 text-muted mb-3"></i>
                <h5 class="text-muted">No render jobs yet</h5>
                <p class="text-muted">Upload a .blend file to start rendering</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Right Column: System Info -->
      <div class="col-lg-4">
        <!-- Worker Nodes Card -->
        <div class="card mb-4">
          <div class="card-header">
            <h5 class="mb-0"><i class="bi bi-pc-display me-2"></i>Worker Nodes</h5>
          </div>
          <div class="card-body">
            <div id="nodesList">
              <div class="text-center py-4">
                <i class="bi bi-hdd-network display-4 text-muted mb-3"></i>
                <h6 class="text-muted">No workers connected</h6>
                <p class="text-muted small">Start worker nodes to begin rendering</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>

<script>
  // Simple state management
  let apiToken = localStorage.getItem('apiToken') || '';
  let currentUser = null;
  
  // Initialize on page load
  document.addEventListener('DOMContentLoaded', function() {
    console.log("Page loaded, checking token...");
    if (apiToken) {
      checkToken();
    } else {
      showLogin();
    }
  });
  
  // Show login section
  function showLogin() {
    console.log("Showing login form");
    document.getElementById('loginSection').style.display = 'block';
    document.getElementById('dashboardSection').style.display = 'none';
    document.getElementById('userInfo').textContent = 'Not logged in';
  }
  
  // Show dashboard
  function showDashboard() {
    console.log("Showing dashboard");
    document.getElementById('loginSection').style.display = 'none';
    document.getElementById('dashboardSection').style.display = 'block';
    if (currentUser) {
      document.getElementById('userInfo').textContent = 'Logged in as: ' + currentUser.username;
    }
    updateDashboard();
  }
  
  // Check if token is valid
  async function checkToken() {
    console.log("Checking token...");
    try {
      const response = await fetch('/api/current_user?token=' + encodeURIComponent(apiToken));
      if (response.ok) {
        const user = await response.json();
        console.log("Token valid, user:", user);
        currentUser = user;
        showDashboard();
      } else {
        console.log("Token invalid");
        showLogin();
      }
    } catch (error) {
      console.error("Token check failed:", error);
      showLogin();
    }
  }
  
  // Login function - SIMPLE AND WORKING
  async function login() {
    console.log("Login button clicked!");
    
    const username = document.getElementById('loginUsername').value;
    const password = document.getElementById('loginPassword').value;
    
    console.log("Username:", username, "Password:", password ? "***" : "empty");
    
    if (!username || !password) {
      alert('Please enter both username and password');
      return;
    }
    
    // Disable button and show loading
    const loginBtn = document.getElementById('loginButton');
    const originalText = loginBtn.innerHTML;
    loginBtn.innerHTML = '<i class="bi bi-hourglass-split me-2"></i>Logging in...';
    loginBtn.disabled = true;
    
    try {
      console.log("Sending login request to /api/login");
      
      const response = await fetch('/api/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          username: username,
          password: password
        })
      });
      
      console.log("Response status:", response.status);
      
      const result = await response.json();
      console.log("Response data:", result);
      
      if (response.ok) {
        console.log("Login successful, token:", result.token ? "received" : "missing");
        apiToken = result.token;
        localStorage.setItem('apiToken', apiToken);
        currentUser = {
          username: result.username,
          is_admin: result.is_admin
        };
        
        // Verify the token works
        const verifyResponse = await fetch('/api/current_user?token=' + encodeURIComponent(apiToken));
        if (verifyResponse.ok) {
          showDashboard();
        } else {
          alert('Login failed: Token verification failed');
          showLogin();
        }
      } else {
        alert('Login failed: ' + (result.error || 'Unknown error'));
      }
    } catch (error) {
      console.error("Login error details:", error);
      alert('Login error: ' + error.message);
    } finally {
      // Restore button
      loginBtn.innerHTML = originalText;
      loginBtn.disabled = false;
    }
  }
  
  // Logout
  function logout() {
    if (confirm('Are you sure you want to logout?')) {
      apiToken = '';
      localStorage.removeItem('apiToken');
      currentUser = null;
      showLogin();
    }
  }
  
  // Dashboard updates
  let dashboardInterval = null;
  
  function updateDashboard() {
    console.log("Starting dashboard updates");
    if (!apiToken) return;
    
    // Clear any existing interval
    if (dashboardInterval) {
      clearInterval(dashboardInterval);
    }
    
    // Fetch immediately
    fetchDashboardData();
    
    // Set up auto-refresh every 3 seconds
    dashboardInterval = setInterval(fetchDashboardData, 3000);
  }
  
  async function fetchDashboardData() {
    if (!apiToken) return;
    
    try {
      // Fetch jobs
      const jobsRes = await fetch('/api/jobs', {
        headers: {
          'X-API-Token': apiToken
        }
      });
      
      if (jobsRes.status === 401) {
        // Token expired
        showLogin();
        return;
      }
      
      if (jobsRes.ok) {
        const jobs = await jobsRes.json();
        updateJobsDisplay(jobs);
      }
      
      // Fetch workers
      const workersRes = await fetch('/api/workers', {
        headers: {
          'X-API-Token': apiToken
        }
      });
      
      if (workersRes.ok) {
        const workers = await workersRes.json();
        updateWorkersDisplay(workers);
      }
      
    } catch (error) {
      console.error('Dashboard update failed:', error);
    }
  }
  
  function updateJobsDisplay(jobs) {
    const jobsList = document.getElementById('jobsList');
    
    if (!jobs || jobs.length === 0) {
      jobsList.innerHTML = '<div class="text-center py-5">' +
        '<i class="bi bi-inboxes display-4 text-muted mb-3"></i>' +
        '<h5 class="text-muted">No render jobs yet</h5>' +
        '<p class="text-muted">Upload a .blend file to start rendering</p>' +
        '</div>';
      return;
    }
    
    // Update stats
    const totalJobs = jobs.length;
    const completedFrames = jobs.reduce((sum, job) => sum + (job.completed_frames || 0), 0);
    
    // Calculate FPS (frames per second) from active jobs
    let fps = 0;
    const activeJobs = jobs.filter(job => job.status === 'processing');
    if (activeJobs.length > 0) {
        fps = Math.round(completedFrames / Math.max(1, totalJobs) * 10);
    }
    
    document.getElementById('totalJobs').textContent = totalJobs;
    document.getElementById('completedFrames').textContent = completedFrames;
    document.getElementById('renderingSpeed').textContent = fps;
    
    // Render job cards
    let jobsHTML = '';
    for (const job of jobs) {
      const totalFrames = job.end_frame - job.start_frame + 1;
      const completed = job.completed_frames || 0;
      const progress = totalFrames > 0 ? (completed / totalFrames * 100).toFixed(1) : 0;
      
      let statusColor = 'primary';
      switch(job.status) {
        case 'completed': statusColor = 'success'; break;
        case 'cancelled': statusColor = 'warning'; break;
        case 'failed': statusColor = 'danger'; break;
        default: statusColor = 'primary';
      }
      
      const jobName = job.job_name || 'this job';
      const escapedJobName = jobName.replace(/'/g, "\\'");
      
      jobsHTML += `
        <div class="job-card">
          <div class="d-flex justify-content-between align-items-start mb-2">
            <div>
              <h6 class="mb-1">${job.job_name || 'Unnamed Job'}</h6>
              <small class="text-muted">
                ID: ${job.job_id ? job.job_id.substring(0, 8) : 'N/A'} | 
                Frames: ${job.start_frame}-${job.end_frame}
              </small>
            </div>
            <div>
              <span class="badge bg-${statusColor}">${job.status}</span>
              <div class="text-end mt-1 small">${completed}/${totalFrames}</div>
            </div>
          </div>
          
          <div class="progress mb-3" style="height: 8px;">
            <div class="progress-bar bg-${statusColor}" style="width: ${progress}%"></div>
          </div>
          
          <div class="d-flex gap-2">`;
      
      if (completed === totalFrames && job.job_id && job.status === 'completed') {
        jobsHTML += `<button class="btn btn-sm btn-success" onclick="downloadFrames('${job.job_id}')">
                <i class="bi bi-download"></i> Download
              </button>`;
      }
      
      if (job.status === 'processing') {
        jobsHTML += `<button class="btn btn-sm btn-danger" onclick="cancelJob('${job.job_id}', '${escapedJobName}')">
                <i class="bi bi-x-circle"></i> Cancel
              </button>`;
      }
      
      if (job.status === 'cancelled') {
        jobsHTML += `<button class="btn btn-sm btn-warning" onclick="restartJob('${job.job_id}', '${escapedJobName}')">
                <i class="bi bi-arrow-clockwise"></i> Restart
              </button>`;
      }
      
      jobsHTML += `<button class="btn btn-sm btn-dark" onclick="deleteJob('${job.job_id}', '${escapedJobName}')">
              <i class="bi bi-trash"></i> Delete
            </button>
          </div>
        </div>`;
    }
    
    jobsList.innerHTML = jobsHTML;
  }
  
  function updateWorkersDisplay(workers) {
    const nodesList = document.getElementById('nodesList');
    
    if (!workers || workers.length === 0) {
      nodesList.innerHTML = '<div class="text-center py-4">' +
        '<i class="bi bi-hdd-network display-4 text-muted mb-3"></i>' +
        '<h6 class="text-muted">No workers connected</h6>' +
        '<p class="text-muted small">Start worker nodes to begin rendering</p>' +
        '</div>';
      document.getElementById('activeWorkers').textContent = '0';
      document.getElementById('nodeCount').textContent = '0 Nodes';
      return;
    }
    
    // Count active workers
    const activeWorkers = workers.filter(w => w.status === 'idle' || w.status === 'busy').length;
    document.getElementById('activeWorkers').textContent = activeWorkers;
    document.getElementById('nodeCount').textContent = activeWorkers + ' Nodes';
    
    // Render worker cards
    let workersHTML = '';
    for (const worker of workers) {
      const statusClass = worker.status === 'idle' ? 'node-idle' : 
                         worker.status === 'busy' ? 'node-busy' : 'node-offline';
      const statusColor = worker.status === 'idle' ? 'success' : 
                         worker.status === 'busy' ? 'warning' : 'danger';
      
      // Parse GPU info if available
      let gpuInfo = '';
      try {
        if (worker.gpu_info) {
          const gpu = typeof worker.gpu_info === 'string' ? JSON.parse(worker.gpu_info) : worker.gpu_info;
          if (gpu.vram_gb) {
            gpuInfo = gpu.vram_gb + 'GB VRAM';
          }
        }
      } catch (e) {
        // Ignore parsing errors
      }
      
      const workerName = worker.hostname || worker.worker_id;
      const escapedWorkerName = workerName.replace(/'/g, "\\'");
      
      workersHTML += `
        <div class="node-card ${statusClass}">
          <div class="d-flex justify-content-between align-items-start">
            <div>
              <strong>${workerName}</strong>
              <br>
              <small class="text-muted">
                <i class="bi bi-pc"></i> ${worker.worker_id || 'Unknown'}
                <br>
                <i class="bi bi-geo-alt"></i> ${worker.ip_address || 'Unknown'}
                ${gpuInfo ? '<br><i class="bi bi-gpu-card"></i> ' + gpuInfo : ''}
              </small>
            </div>
            <div class="text-end">
              <span class="badge bg-${statusColor}">
                ${worker.status || 'offline'}
              </span>
              <br>
              <small class="text-muted">Load: ${(worker.current_load * 100 || 0).toFixed(0)}%</small>
            </div>
          </div>`;
      
      if (currentUser && currentUser.is_admin) {
        workersHTML += `
          <div class="mt-2 text-end">
            <button class="btn btn-sm btn-outline-danger" onclick="deleteWorker('${worker.worker_id}', '${escapedWorkerName}')">
              <i class="bi bi-trash"></i> Remove
            </button>
          </div>`;
      }
      
      workersHTML += `</div>`;
    }
    
    nodesList.innerHTML = workersHTML;
  }
  
  // Job actions
  async function cancelJob(jobId, jobName) {
    if (!confirm('Are you sure you want to cancel "' + jobName + '"?')) return;
    
    try {
      const response = await fetch('/api/job/' + jobId + '/cancel', {
        method: 'POST',
        headers: {
          'X-API-Token': apiToken
        }
      });
      
      if (response.ok) {
        const result = await response.json();
        alert('✅ Job cancelled: ' + result.message);
        updateDashboard();
      } else {
        const error = await response.json();
        alert('❌ Error: ' + error.error);
      }
    } catch (error) {
      alert('❌ Cancel failed: ' + error);
    }
  }
  
  async function restartJob(jobId, jobName) {
    if (!confirm('Restart "' + jobName + '"? Uncompleted frames will be re-rendered.')) return;
    
    try {
      const response = await fetch('/api/job/' + jobId + '/restart', {
        method: 'POST',
        headers: {
          'X-API-Token': apiToken
        }
      });
      
      if (response.ok) {
        const result = await response.json();
        alert('✅ Job restarted: ' + result.message);
        updateDashboard();
      } else {
        const error = await response.json();
        alert('❌ Error: ' + error.error);
      }
    } catch (error) {
      alert('❌ Restart failed: ' + error);
    }
  }
  
  async function deleteJob(jobId, jobName) {
    if (!confirm('Permanently delete "' + jobName + '"?\\n\\nThis will remove:\\n• Job record\\n• Uploaded blend file\\n• All rendered frames\\n\\nThis action cannot be undone!')) return;
    
    try {
      const response = await fetch('/api/job/' + jobId + '/delete', {
        method: 'POST',
        headers: {
          'X-API-Token': apiToken
        }
      });
      
      if (response.ok) {
        const result = await response.json();
        alert('✅ Job deleted: ' + result.message);
        updateDashboard();
      } else {
        const error = await response.json();
        alert('❌ Error: ' + error.error);
      }
    } catch (error) {
      alert('❌ Delete failed: ' + error);
    }
  }
  
  // Delete worker node
  async function deleteWorker(workerId, workerName) {
    if (!confirm('Remove worker "' + workerName + '"?\\n\\nThis will:\\n• Delete the worker from the system\\n• Reassign any pending frames to other workers\\n\\nAre you sure?')) return;
    
    try {
      const response = await fetch('/api/worker/' + workerId + '/delete', {
        method: 'POST',
        headers: {
          'X-API-Token': apiToken
        }
      });
      
      if (response.ok) {
        const result = await response.json();
        alert('✅ Worker removed: ' + result.message);
        updateDashboard();
      } else {
        const error = await response.json();
        alert('❌ Error: ' + error.error);
      }
    } catch (error) {
      alert('❌ Failed to remove worker: ' + error);
    }
  }
  
  function downloadFrames(jobId) {
    window.location.href = '/api/download_frames/' + jobId + '?token=' + encodeURIComponent(apiToken);
  }
  
  // Upload form handler
  document.getElementById('uploadForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    if (!apiToken) {
      alert('Please login first');
      return;
    }
    
    const formData = new FormData();
    formData.append('job_name', document.getElementById('jobName').value);
    formData.append('blend_file', document.getElementById('blendFile').files[0]);
    formData.append('start_frame', document.getElementById('startFrame').value);
    formData.append('end_frame', document.getElementById('endFrame').value);
    formData.append('output_format', document.getElementById('outputFormat').value);
    formData.append('priority', document.getElementById('priority').value);
    
    try {
      const response = await fetch('/api/upload_job', {
        method: 'POST',
        headers: {
          'X-API-Token': apiToken
        },
        body: formData
      });
      
      const result = await response.json();
      
      if (response.ok) {
        alert('Job created! ID: ' + result.job_id);
        document.getElementById('uploadForm').reset();
        updateDashboard();
      } else {
        alert('Error: ' + result.error);
      }
    } catch (error) {
      alert('Upload failed: ' + error);
    }
  });
</script>
</body>
</html>
'''

# ===== Flask Routes =====

@app.route('/')
def index():
    return render_template_string(INDEX_HTML)

@app.route('/api/login', methods=['POST'])
def login():
    """User login endpoint"""
    data = request.get_json()
    if not data or 'username' not in data or 'password' not in data:
        return jsonify({'error': 'Username and password required'}), 400
    
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE username = ?', (data['username'],))
    user = cursor.fetchone()
    conn.close()
    
    if not user:
        return jsonify({'error': 'Invalid credentials'}), 401
    
    # Verify password (in production, use proper password hashing)
    password_hash = hashlib.sha256(data['password'].encode()).hexdigest()
    if password_hash != user['password_hash']:
        return jsonify({'error': 'Invalid credentials'}), 401
    
    # Update last login
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?', (user['id'],))
    conn.commit()
    conn.close()
    
    return jsonify({
        'success': True,
        'token': user['api_token'],
        'username': user['username'],
        'is_admin': bool(user['is_admin'])
    })

@app.route('/api/register_worker', methods=['POST'])
@token_required
def register_worker():
    """Register a new worker node"""
    data = request.get_json()
    if not data or 'hostname' not in data:
        return jsonify({'error': 'Hostname required'}), 400
    
    # Generate worker ID and token
    worker_id = f"worker_{data['hostname']}_{uuid.uuid4().hex[:8]}"
    api_token = secrets.token_urlsafe(32)
    
    # Extract system info
    system_info = data.get('system_info', {})
    gpu_info = system_info.get('gpu_info', {})
    
    conn = get_db()
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            INSERT INTO workers (worker_id, hostname, ip_address, port, api_token, gpu_info, 
                                cpu_cores, total_ram_gb, render_speed_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            worker_id,
            data['hostname'],
            request.remote_addr,
            data.get('port', 0),
            api_token,
            json.dumps(gpu_info),
            system_info.get('cpu_cores', 4),
            system_info.get('total_ram_gb', 8),
            data.get('render_speed_score', 1.0)
        ))
        
        conn.commit()
        
        return jsonify({
            'success': True,
            'worker_id': worker_id,
            'api_token': api_token,
            'message': 'Worker registered successfully'
        })
        
    except sqlite3.IntegrityError as e:
        print(f"❌ Database error: {e}")
        return jsonify({'error': 'Worker already exists'}), 400
    except Exception as e:
        print(f"❌ Registration error: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        conn.close()

@app.route('/api/upload_job', methods=['POST'])
@token_required
def upload_job():
    """Upload a new render job"""
    try:
        if 'blend_file' not in request.files:
            return jsonify({'error': 'No blend file uploaded'}), 400
        
        blend_file = request.files['blend_file']
        if blend_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not blend_file.filename.lower().endswith('.blend'):
            return jsonify({'error': 'File must be a .blend file'}), 400
        
        # Get user from token
        token = request.headers['X-API-Token']
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute('SELECT id FROM users WHERE api_token = ?', (token,))
        user = cursor.fetchone()
        
        if not user:
            return jsonify({'error': 'User not found'}), 401
        
        user_id = user['id']
        
        # Save blend file
        original_filename = secure_filename(blend_file.filename)
        unique_filename = f"{uuid.uuid4().hex}_{original_filename}"
        blend_path = UPLOAD_FOLDER / unique_filename
        blend_file.save(blend_path)
        
        # Calculate file hash
        file_hash = hashlib.md5(open(blend_path, 'rb').read()).hexdigest()
        file_size = blend_path.stat().st_size
        
        # Get job parameters
        job_name = request.form.get('job_name', 'Unnamed Job')
        start_frame = int(request.form.get('start_frame', 1))
        end_frame = int(request.form.get('end_frame', 250))
        output_format = request.form.get('output_format', 'PNG')
        priority = int(request.form.get('priority', 1))
        
        # Parse render settings
        render_settings = {
            'samples': int(request.form.get('samples', 128)),
            'resolution': request.form.get('resolution', '1920x1080'),
            'engine': request.form.get('engine', 'CYCLES')
        }
        
        # Create job record
        job_uuid = str(uuid.uuid4())
        cursor.execute('''
            INSERT INTO jobs (
                job_uuid, user_id, job_name, blend_filename, blend_file_hash, 
                blend_file_size, start_frame, end_frame, output_format, 
                render_settings, priority, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'processing')
        ''', (
            job_uuid, user_id, job_name, unique_filename, file_hash,
            file_size, start_frame, end_frame, output_format,
            json.dumps(render_settings), priority
        ))
        
        job_id = cursor.lastrowid
        
        # Create frame records
        frames = []
        for frame_num in range(start_frame, end_frame + 1):
            cursor.execute('''
                INSERT INTO frames (job_id, frame_number)
                VALUES (?, ?)
            ''', (job_id, frame_num))
            frames.append({'frame_number': frame_num, 'id': cursor.lastrowid})
        
        conn.commit()
        conn.close()
        
        print(f"✓ Job created: {job_uuid} - {job_name} ({start_frame}-{end_frame})")
        
        return jsonify({
            'success': True,
            'job_id': job_uuid,
            'job_name': job_name,
            'frames': len(frames),
            'message': 'Job created successfully'
        })
        
    except Exception as e:
        print(f"✗ Error creating job: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/upload_frame', methods=['POST'])
@worker_token_required
def upload_frame():
    """Endpoint for workers to upload rendered frames"""
    try:
        if 'frame_file' not in request.files or 'frame_id' not in request.form:
            return jsonify({'error': 'Frame file and ID required'}), 400
        
        frame_id_str = request.form['frame_id']
        try:
            frame_id = int(frame_id_str)  # Convert to integer
        except ValueError:
            return jsonify({'error': 'frame_id must be an integer'}), 400
            
        frame_file = request.files['frame_file']
        
        # Verify frame belongs to this worker
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT f.id, j.job_uuid, j.output_format, f.worker_id, f.frame_number
            FROM frames f
            JOIN jobs j ON f.job_id = j.id
            WHERE f.id = ? AND f.worker_id = ?
        ''', (frame_id, request.worker_id))
        
        frame = cursor.fetchone()
        if not frame:
            return jsonify({'error': 'Frame not assigned to this worker'}), 403
        
        # Save the file
        job_output_dir = OUTPUT_BASE / frame['job_uuid'] / 'frames'
        job_output_dir.mkdir(parents=True, exist_ok=True)
        
        ext_map = {'PNG': '.png', 'JPEG': '.jpg', 'OPEN_EXR': '.exr', 'TIFF': '.tif'}
        ext = ext_map.get(frame['output_format'], '.png')
        
        # Use frame number for filename
        frame_number = frame['frame_number']
        filename = f"frame_{frame_number:04d}{ext}"
        filepath = job_output_dir / filename
        
        frame_file.save(filepath)
        
        # Update frame record
        file_size = filepath.stat().st_size
        checksum = hashlib.md5(open(filepath, 'rb').read()).hexdigest()
        
        cursor.execute('''
            UPDATE frames 
            SET file_path = ?, file_size = ?, checksum = ?
            WHERE id = ?
        ''', (str(filepath), file_size, checksum, frame_id))
        
        conn.commit()
        conn.close()
        
        return jsonify({
            'success': True,
            'message': 'Frame uploaded successfully',
            'file_size': file_size,
            'checksum': checksum
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/job/<job_uuid>/delete', methods=['POST'])
@token_required
def delete_job(job_uuid):
    """Permanently delete a job and all associated files"""
    try:
        token = request.headers['X-API-Token']
        conn = get_db()
        cursor = conn.cursor()
        
        # Get user ID from token
        cursor.execute('SELECT id FROM users WHERE api_token = ?', (token,))
        user = cursor.fetchone()
        
        if not user:
            return jsonify({'error': 'User not found'}), 401
        
        # Get the job details
        cursor.execute('''
            SELECT j.id, j.status, j.user_id, j.blend_filename
            FROM jobs j
            WHERE j.job_uuid = ?
        ''', (job_uuid,))
        
        job = cursor.fetchone()
        
        if not job:
            return jsonify({'error': 'Job not found'}), 404
        
        # Check if user owns the job or is admin
        cursor.execute('SELECT is_admin FROM users WHERE id = ?', (user['id'],))
        user_info = cursor.fetchone()
        is_admin = user_info['is_admin'] if user_info else False
        
        if not is_admin and job['user_id'] != user['id']:
            return jsonify({'error': 'You do not have permission to delete this job'}), 403
        
        # First, update workers that were assigned to this job (reduce their load)
        cursor.execute('''
            UPDATE workers 
            SET current_load = MAX(0, current_load - 0.1),
                status = CASE 
                    WHEN current_load <= 0.1 THEN 'idle'
                    ELSE 'busy'
                END
            WHERE id IN (
                SELECT DISTINCT worker_id 
                FROM frames 
                WHERE job_id = ? 
                AND status = 'assigned'
                AND worker_id IS NOT NULL
            )
        ''', (job['id'],))
        
        # Store file paths before deleting database records
        blend_file_path = UPLOAD_FOLDER / job['blend_filename'] if job['blend_filename'] else None
        
        # Delete all frames first (due to foreign key constraints)
        cursor.execute('DELETE FROM frames WHERE job_id = ?', (job['id'],))
        
        # Delete all logs for this job
        cursor.execute('DELETE FROM render_logs WHERE job_id = ?', (job['id'],))
        
        # Now delete the job
        cursor.execute('DELETE FROM jobs WHERE id = ?', (job['id'],))
        
        conn.commit()
        conn.close()
        
        # Delete physical files (outside of transaction)
        deleted_files = []
        
        # Delete blend file if it exists
        if blend_file_path and blend_file_path.exists():
            try:
                blend_file_path.unlink()
                deleted_files.append(f"Blend file: {job['blend_filename']}")
            except Exception as e:
                print(f"Warning: Could not delete blend file {blend_file_path}: {e}")
        
        # Delete rendered frames directory
        frames_dir = OUTPUT_BASE / job_uuid
        if frames_dir.exists():
            try:
                import shutil
                shutil.rmtree(frames_dir)
                deleted_files.append(f"Rendered frames directory")
            except Exception as e:
                print(f"Warning: Could not delete frames directory {frames_dir}: {e}")
        
        message = f'Job permanently deleted'
        if deleted_files:
            message += f'. Removed: {", ".join(deleted_files)}'
        
        print(f"✓ Job {job_uuid} deleted by user {user['id']}")
        
        return jsonify({
            'success': True,
            'message': message
        })
        
    except Exception as e:
        print(f"✗ Error deleting job: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        conn.close()

@app.route('/api/jobs')
@token_required
def api_jobs():
    """Get all jobs for the current user"""
    token = request.headers['X-API-Token']
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT j.*, u.username,
               (SELECT COUNT(*) FROM frames f WHERE f.job_id = j.id AND f.status = 'completed') as completed_frames,
               (SELECT COUNT(*) FROM frames f WHERE f.job_id = j.id) as total_frames
        FROM jobs j
        JOIN users u ON j.user_id = u.id
        WHERE u.api_token = ?
        ORDER BY j.created_at DESC
    ''', (token,))
    
    jobs = cursor.fetchall()
    conn.close()
    
    job_list = []
    for job in jobs:
        job_list.append({
            'job_id': job['job_uuid'],
            'job_name': job['job_name'],
            'username': job['username'],
            'start_frame': job['start_frame'],
            'end_frame': job['end_frame'],
            'output_format': job['output_format'],
            'status': job['status'],
            'completed_frames': job['completed_frames'],
            'total_frames': job['total_frames'],
            'created_at': job['created_at']
        })
    
    return jsonify(job_list)

@app.route('/api/workers')
@token_required
def api_workers():
    """Get all worker nodes"""
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT worker_id, hostname, ip_address, status, 
               gpu_info, cpu_cores, total_ram_gb, render_speed_score,
               current_load, last_heartbeat
        FROM workers
        ORDER BY status, last_heartbeat DESC
    ''')
    
    workers = cursor.fetchall()
    conn.close()
    
    worker_list = []
    for worker in workers:
        worker_dict = dict(worker)
        if worker_dict['gpu_info']:
            worker_dict['gpu_info'] = json.loads(worker_dict['gpu_info'])
        worker_list.append(worker_dict)
    
    return jsonify(worker_list)

# ===== Additional API Endpoints for Web UI =====

@app.route('/api/nodes')
def api_nodes():
    """Get all worker nodes (alias for /api/workers)"""
    return api_workers()  # Use the same logic

@app.route('/api/current_user')
def api_current_user():
    """Get current user info from token"""
    token = request.args.get('token') or request.headers.get('X-API-Token')
    if not token:
        return jsonify({'error': 'No token provided'}), 401
    
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT username, is_admin FROM users WHERE api_token = ?', (token,))
    user = cursor.fetchone()
    conn.close()
    
    if not user:
        return jsonify({'error': 'Invalid token'}), 401
    
    return jsonify({
        'username': user['username'],
        'is_admin': bool(user['is_admin'])
    })

@app.route('/api/get_blend_file/<job_uuid>', methods=['GET'])
@worker_token_required
def get_blend_file(job_uuid):
    """Serve blend file to workers"""
    try:
        conn = get_db()
        cursor = conn.cursor()
        
        # Get job info
        cursor.execute('''
            SELECT blend_filename FROM jobs WHERE job_uuid = ?
        ''', (job_uuid,))
        job = cursor.fetchone()
        
        if not job:
            return jsonify({'error': 'Job not found'}), 404
        
        blend_filename = job['blend_filename']
        blend_path = UPLOAD_FOLDER / blend_filename
        
        if not blend_path.exists():
            return jsonify({'error': 'Blend file not found'}), 404
        
        # Send file
        return send_file(
            blend_path,
            as_attachment=True,
            download_name=blend_filename,
            mimetype='application/octet-stream'
        )
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        conn.close()

@app.route('/api/debug/workers')
def debug_workers():
    """Debug endpoint to check worker registration"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT worker_id, api_token, status, hostname FROM workers')
    workers = cursor.fetchall()
    conn.close()
    
    result = []
    for worker in workers:
        result.append({
            'worker_id': worker['worker_id'],
            'api_token_prefix': worker['api_token'][:20] + '...' if worker['api_token'] else None,
            'status': worker['status'],
            'hostname': worker['hostname']
        })
    
    return jsonify(result)

@app.route('/api/job/<job_uuid>/cancel', methods=['POST'])
@token_required
def cancel_job(job_uuid):
    """Cancel a running job"""
    try:
        token = request.headers['X-API-Token']
        conn = get_db()
        cursor = conn.cursor()
        
        # Get user ID from token
        cursor.execute('SELECT id FROM users WHERE api_token = ?', (token,))
        user = cursor.fetchone()
        
        if not user:
            return jsonify({'error': 'User not found'}), 401
        
        # Get the job
        cursor.execute('''
            SELECT j.id, j.status, j.user_id, 
                   COUNT(CASE WHEN f.status = 'completed' THEN 1 END) as completed_frames,
                   COUNT(*) as total_frames
            FROM jobs j
            LEFT JOIN frames f ON j.id = f.job_id
            WHERE j.job_uuid = ?
            GROUP BY j.id
        ''', (job_uuid,))
        
        job = cursor.fetchone()
        
        if not job:
            return jsonify({'error': 'Job not found'}), 404
        
        # Check if user owns the job or is admin
        cursor.execute('SELECT is_admin FROM users WHERE id = ?', (user['id'],))
        user_info = cursor.fetchone()
        is_admin = user_info['is_admin'] if user_info else False
        
        if not is_admin and job['user_id'] != user['id']:
            return jsonify({'error': 'You do not have permission to cancel this job'}), 403
        
        # Check if job can be cancelled
        if job['status'] in ['completed', 'cancelled', 'failed']:
            return jsonify({'error': f'Job is already {job["status"]}'}), 400
        
        # Update job status to cancelled
        cursor.execute('''
            UPDATE jobs 
            SET status = 'cancelled',
                cancelled_at = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (job['id'],))
        
        # Update all non-completed frames to cancelled
        cursor.execute('''
            UPDATE frames 
            SET status = 'cancelled',
                updated_at = CURRENT_TIMESTAMP
            WHERE job_id = ? AND status NOT IN ('completed', 'failed')
        ''', (job['id'],))
        
        # Update workers that were assigned to this job
        cursor.execute('''
            UPDATE workers 
            SET current_load = MAX(0, current_load - 0.1),
                status = CASE 
                    WHEN current_load <= 0.1 THEN 'idle'
                    ELSE 'busy'
                END
            WHERE id IN (
                SELECT DISTINCT worker_id 
                FROM frames 
                WHERE job_id = ? 
                AND status = 'assigned'
                AND worker_id IS NOT NULL
            )
        ''', (job['id'],))
        
        conn.commit()
        
        return jsonify({
            'success': True,
            'message': f'Job cancelled. {job["completed_frames"]}/{job["total_frames"]} frames were completed.'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        conn.close()

@app.route('/api/job/<job_uuid>/restart', methods=['POST'])
@token_required
def restart_job(job_uuid):
    """Restart a cancelled job"""
    try:
        token = request.headers['X-API-Token']
        conn = get_db()
        cursor = conn.cursor()
        
        # Get user ID from token
        cursor.execute('SELECT id FROM users WHERE api_token = ?', (token,))
        user = cursor.fetchone()
        
        if not user:
            return jsonify({'error': 'User not found'}), 401
        
        # Get the job
        cursor.execute('''
            SELECT j.id, j.status, j.user_id, 
                   COUNT(CASE WHEN f.status = 'completed' THEN 1 END) as completed_frames,
                   COUNT(CASE WHEN f.status = 'cancelled' THEN 1 END) as cancelled_frames,
                   COUNT(*) as total_frames
            FROM jobs j
            LEFT JOIN frames f ON j.id = f.job_id
            WHERE j.job_uuid = ?
            GROUP BY j.id
        ''', (job_uuid,))
        
        job = cursor.fetchone()
        
        if not job:
            return jsonify({'error': 'Job not found'}), 404
        
        # Check if user owns the job or is admin
        cursor.execute('SELECT is_admin FROM users WHERE id = ?', (user['id'],))
        user_info = cursor.fetchone()
        is_admin = user_info['is_admin'] if user_info else False
        
        if not is_admin and job['user_id'] != user['id']:
            return jsonify({'error': 'You do not have permission to restart this job'}), 403
        
        # Check if job can be restarted
        if job['status'] not in ['cancelled', 'failed']:
            return jsonify({'error': f'Job is {job["status"]}, cannot restart'}), 400
        
        # Update job status to processing
        cursor.execute('''
            UPDATE jobs 
            SET status = 'processing',
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (job['id'],))
        
        # Update cancelled frames to pending
        cursor.execute('''
            UPDATE frames 
            SET status = 'pending',
                worker_id = NULL,
                assigned_at = NULL,
                completed_at = NULL,
                error_message = NULL,
                retry_count = 0,
                updated_at = CURRENT_TIMESTAMP
            WHERE job_id = ? AND status = 'cancelled'
        ''', (job['id'],))
        
        conn.commit()
        
        return jsonify({
            'success': True,
            'message': f'Job restarted. {job["completed_frames"]}/{job["total_frames"]} frames were already completed.'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        conn.close()

@app.route('/api/download_frames/<job_uuid>')
# REMOVED the @token_required decorator here
def download_frames(job_uuid):
    """Download all frames for a job as a zip file"""
    try:
        # Get token from URL parameters (sent by the web UI)
        token = request.args.get('token')
        if not token:
            # Fallback: also check the headers
            token = request.headers.get('X-API-Token')
        
        if not token:
            return jsonify({'error': 'Token is missing'}), 401
        
        conn = get_db()
        cursor = conn.cursor()
        
        # 1. Get user from token
        cursor.execute('SELECT id, is_admin FROM users WHERE api_token = ?', (token,))
        user = cursor.fetchone()
        if not user:
            conn.close()
            return jsonify({'error': 'Invalid token'}), 401
        
        user_id = user['id']
        is_admin = user['is_admin']
        
        # 2. Get job
        cursor.execute('''
            SELECT j.id, j.job_name, j.user_id
            FROM jobs j
            WHERE j.job_uuid = ?
        ''', (job_uuid,))
        job = cursor.fetchone()
        conn.close()
        
        if not job:
            return jsonify({'error': 'Job not found'}), 404
        
        # 3. Check permission: user owns the job OR is admin
        if not is_admin and job['user_id'] != user_id:
            return jsonify({'error': 'You do not have permission to download these frames'}), 403
        
        # 4. Check if frames directory exists
        frames_dir = OUTPUT_BASE / job_uuid / 'frames'
        if not frames_dir.exists():
            return jsonify({'error': 'No frames rendered yet'}), 404
        
        # 5. Create a zip file in memory
        import io
        import zipfile
        
        data = io.BytesIO()
        with zipfile.ZipFile(data, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for frame_file in frames_dir.iterdir():
                if frame_file.is_file():
                    # Use arcname to avoid full paths in the zip
                    zipf.write(frame_file, frame_file.name)
        
        data.seek(0)
        
        # 6. Send the zip file
        return send_file(
            data,
            mimetype='application/zip',
            as_attachment=True,
            download_name=f'render_frames_{job_uuid}.zip'
        )
        
    except Exception as e:
        print(f"Error during download: {e}")  # Log error for debugging
        return jsonify({'error': str(e)}), 500

@app.route('/api/worker/<worker_id>/delete', methods=['POST'])
@token_required
def delete_worker(worker_id):
    """Delete a worker node"""
    try:
        token = request.headers['X-API-Token']
        conn = get_db()
        cursor = conn.cursor()
        
        # Get user from token
        cursor.execute('SELECT is_admin FROM users WHERE api_token = ?', (token,))
        user = cursor.fetchone()
        
        if not user:
            return jsonify({'error': 'User not found'}), 401
        
        # Check if user is admin
        if not user['is_admin']:
            return jsonify({'error': 'Only admin users can delete workers'}), 403
        
        # Get the worker details
        cursor.execute('''
            SELECT id, status, worker_id
            FROM workers
            WHERE worker_id = ?
        ''', (worker_id,))
        
        worker = cursor.fetchone()
        
        if not worker:
            return jsonify({'error': 'Worker not found'}), 404
        
        # Check if worker is currently busy
        if worker['status'] == 'busy':
            return jsonify({
                'error': 'Cannot delete busy worker. Wait for it to finish or cancel its assigned frames first.'
            }), 400
        
        # Free any frames assigned to this worker
        cursor.execute('''
            UPDATE frames 
            SET worker_id = NULL, 
                status = 'pending',
                assigned_at = NULL,
                retry_count = retry_count + 1
            WHERE worker_id = ? AND status = 'assigned'
        ''', (worker['id'],))
        
        # Delete the worker
        cursor.execute('DELETE FROM workers WHERE id = ?', (worker['id'],))
        
        # Log the deletion
        cursor.execute('''
            INSERT INTO render_logs (worker_id, log_level, message)
            VALUES (?, 'warning', ?)
        ''', (worker['id'], f"Worker {worker_id} deleted by admin"))
        
        conn.commit()
        conn.close()
        
        print(f"✓ Worker {worker_id} deleted")
        
        return jsonify({
            'success': True,
            'message': f'Worker {worker_id} deleted successfully'
        })
        
    except Exception as e:
        print(f"✗ Error deleting worker: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        conn.close()

# ===== Main Coordinator Function =====
def main():
    print('\n' + '='*60)
    print('BLENDER 5.0 RENDER FARM PRO - COORDINATOR')
    print('='*60 + '\n')
    
    # Initialize database
    init_db()
    
    # Create load balancer
    load_balancer = LoadBalancer()
    
    # Start worker server
    worker_server = WorkerServer(load_balancer)
    worker_server.start()
    
    # Start frame upload server
    upload_server = FrameUploadServer()
    upload_server.start()
    
    # Start cleanup thread for stale workers
    def cleanup_thread():
        while True:
            time.sleep(60)
            conn = get_db()
            cursor = conn.cursor()
            
            # Mark workers offline if no heartbeat for 5 minutes
            cursor.execute('''
                UPDATE workers 
                SET status = 'offline'
                WHERE last_heartbeat IS NOT NULL 
                AND datetime(last_heartbeat) < datetime('now', '-5 minutes')
                AND status != 'offline'
            ''')
            
            # Reset load for offline workers
            cursor.execute('''
                UPDATE workers 
                SET current_load = 0
                WHERE status = 'offline' AND current_load > 0
            ''')
            
            conn.commit()
            conn.close()
    
    cleanup = threading.Thread(target=cleanup_thread, daemon=True)
    cleanup.start()
    
    print(f'\n🌐 Web UI:          http://localhost:{COORDINATOR_WEB_PORT}')
    print(f'🔧 Worker Port:     {COORDINATOR_NETWORK_PORT}')
    print(f'📤 Frame Upload:    {COORDINATOR_UPLOAD_PORT}')
    print('='*60)
    print('✅ PRO Coordinator ready with all features!')
    print('='*60 + '\n')
    
    # Print admin credentials reminder
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT username, api_token FROM users WHERE is_admin = 1 LIMIT 1')
    admin = cursor.fetchone()
    conn.close()
    
    if admin:
        print(f'🔑 Admin API Token: {admin["api_token"]}')
        print('   Use this token for API authentication\n')
    
    try:
        app.run(
            host=COORDINATOR_HOST,
            port=COORDINATOR_WEB_PORT,
            debug=False,
            use_reloader=False
        )
    except KeyboardInterrupt:
        print("\nShutting down coordinator...")
    finally:
        worker_server.stop_event.set()
        upload_server.stop_event.set()
        print("✅ Coordinator stopped cleanly")

if __name__ == '__main__':
    main()