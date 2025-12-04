import socket
import threading
import subprocess
import os
import sys
import time
import json
import platform
import psutil
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime
import hashlib
import http.client
import mimetypes

# Try to import GPU detection libraries
try:
    import GPUtil
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    print("⚠️  GPUtil not installed. Install with: pip install gputil")

try:
    import pynvml
    HAS_NVML = True
except ImportError:
    HAS_NVML = False

# ===== Configuration =====
COORDINATOR_HOST = 'localhost'
COORDINATOR_NETWORK_PORT = 5555
COORDINATOR_WEB_PORT = 5000
COORDINATOR_UPLOAD_PORT = 5556
WORKER_ID = None  # Will be set after registration
API_TOKEN = None  # Will be set after registration
BLENDER_PATH = ''
# Disable_Addons_Script_Path = Path('./disable_addons.py')
WORK_DIR = Path('./worker_data')
WORK_DIR.mkdir(parents=True, exist_ok=True)

# ===== Advanced GPU Detection =====
class GPUDetector:
    @staticmethod
    def detect_gpu_info():
        """Comprehensive GPU detection for Blender 5.0: CUDA, OptiX, HIP, Metal, oneAPI"""
        import subprocess
        import re
        gpu_info = {
            'gpu_count': 0,
            'gpus': [],
            'cuda': False, 'optix': False, 'hip': False, 'metal': False, 'oneapi': False,
            'vram_total_gb': 0,
            'driver_version': None,
            'platform': platform.system()
        }
        detected_gpus = []

        # Helper to run system commands safely
        def run_cmd(cmd):
            try:
                return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5).stdout.strip()
            except:
                return ""

        # === DETECTION LOGIC BY PLATFORM ===
        system = platform.system()
        
        if system == "Windows":
            # Windows: Check for NVIDIA via nvidia-smi, AMD via device manager pattern
            nvidia_info = run_cmd("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader")
            if nvidia_info:
                for line in nvidia_info.split('\n'):
                    if line:
                        parts = line.split(',')
                        name = parts[0].strip()
                        driver = parts[1].strip() if len(parts) > 1 else "Unknown"
                        mem_str = parts[2].strip() if len(parts) > 2 else "0 MiB"
                        # Convert "8192 MiB" to GB
                        vram_gb = float(re.search(r'(\d+)', mem_str).group(1)) / 1024.0 if 'MiB' in mem_str else 0
                        
                        gpu = {'name': name, 'vram_gb': vram_gb, 'driver': driver}
                        detected_gpus.append(gpu)
                        gpu_info['cuda'] = True
                        # Check if GPU is RTX for OptiX
                        if 'RTX' in name.upper() or 'GEFORCE RTX' in name:
                            gpu_info['optix'] = True
            
            # Check for AMD on Windows (simplified check)
            amd_check = run_cmd("wmic path win32_VideoController get caption")
            if "AMD" in amd_check or "Radeon" in amd_check:
                gpu_info['hip'] = True # Assume modern AMD supports HIP

        elif system == "Linux":
            # Linux: Use lspci for initial detection
            pci_info = run_cmd("lspci | grep -i 'vga\\|3d\\|display'")
            gpu_info['hip'] = 'amd' in pci_info.lower() or 'radeon' in pci_info.lower()
            
            # Check for NVIDIA
            nvidia_info = run_cmd("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null")
            if nvidia_info:
                for line in nvidia_info.split('\n'):
                    if line:
                        parts = line.split(',')
                        name = parts[0].strip()
                        mem_str = parts[1].strip() if len(parts) > 1 else "0 MiB"
                        vram_gb = float(re.search(r'(\d+)', mem_str).group(1)) / 1024.0 if 'MiB' in mem_str else 0
                        detected_gpus.append({'name': name, 'vram_gb': vram_gb, 'driver': "NVIDIA"})
                        gpu_info['cuda'] = True
                        if 'RTX' in name:
                            gpu_info['optix'] = True

        elif system == "Darwin":  # macOS
            # Apple Silicon supports Metal
            mac_model = run_cmd("sysctl -n hw.model")
            if 'Mac' in mac_model:
                gpu_info['metal'] = True
                # Try to get GPU info
                gpu_name = run_cmd("system_profiler SPDisplaysDataType | grep -A2 'Chipset Model' | tail -n1 | cut -d':' -f2").strip() or "Apple Silicon GPU"
                # Estimate VRAM (unified memory). This is an approximation.
                total_ram_gb = psutil.virtual_memory().total / (1024**3)
                est_vram = total_ram_gb / 2  # Rough estimate: half of system RAM
                detected_gpus.append({'name': gpu_name, 'vram_gb': est_vram, 'driver': "macOS"})

        # === POST-PROCESSING ===
        gpu_info['gpu_count'] = len(detected_gpus)
        gpu_info['gpus'] = detected_gpus
        gpu_info['vram_total_gb'] = sum(g['vram_gb'] for g in detected_gpus)
        
        # Determine the BEST render device for Blender CLI argument
        if gpu_info['optix']:
            gpu_info['cycles_device'] = 'OPTIX'
        elif gpu_info['cuda']:
            gpu_info['cycles_device'] = 'CUDA'
        elif gpu_info['hip']:
            gpu_info['cycles_device'] = 'HIP'
        elif gpu_info['metal']:
            gpu_info['cycles_device'] = 'METAL'
        elif gpu_info['oneapi']:
            gpu_info['cycles_device'] = 'ONEAPI'
        else:
            gpu_info['cycles_device'] = 'CPU'
            gpu_info['gpu_count'] = 0 # Fallback to CPU
            
        return gpu_info

def get_system_info():
    """Get comprehensive system information for load balancing"""
    gpu_detector = GPUDetector()
    gpu_info = gpu_detector.detect_gpu_info()
    
    # Calculate a render speed score for load balancing
    render_score = 1.0
    if gpu_info['gpu_count'] > 0:
        render_score *= 10.0  # Base GPU bonus
        render_score *= min(3.0, 1.0 + (gpu_info['vram_total_gb'] / 8.0))  # VRAM factor
        if gpu_info['optix']:
            render_score *= 2.0  # OptiX is fastest
        elif gpu_info['cuda']:
            render_score *= 1.5
        elif gpu_info['metal']:
            render_score *= 1.8
        elif gpu_info['hip']:
            render_score *= 1.7
    else:
        render_score *= 0.3  # CPU-only penalty
        
    return {
        'hostname': platform.node(),
        'platform': platform.system(),
        'platform_version': platform.version(),
        'cpu_cores': psutil.cpu_count(logical=True),
        'cpu_freq': psutil.cpu_freq().current if psutil.cpu_freq() else None,
        'total_ram_gb': psutil.virtual_memory().total / (1024**3),
        'gpu_info': gpu_info,
        'cycles_device': gpu_info['cycles_device'],
        'render_speed_score': round(render_score, 2),
        'ip_address': socket.gethostbyname(socket.gethostname()) # For same-machine detection
    }
   
def get_blender_version():
    """Get Blender version by running blender --version"""
    global BLENDER_PATH
    if not BLENDER_PATH:
        return 'Unknown'
    
    try:
        result = subprocess.run([BLENDER_PATH, '--version'], 
                              capture_output=True, text=True, timeout=5)
        lines = result.stdout.split('\n')
        for line in lines:
            if 'Blender' in line:
                return line.strip()
        return 'Unknown'
    except:
        return 'Unknown'

# ===== Worker Class =====
class ProRenderWorker:
    def __init__(self, coordinator_host, coordinator_port, api_token=None):
        self.coordinator_host = coordinator_host
        self.coordinator_port = coordinator_port
        self.api_token = api_token
        self.worker_id = None
        self.connected = False
        self.socket = None
        self.stop_event = threading.Event()
        self.system_info = get_system_info()
        self.current_load = 0.0
        self.render_speed_score = self.calculate_render_speed()
        
    def calculate_render_speed(self):
        """Calculate a render speed score based on hardware"""
        score = 1.0
        
        # GPU score
        gpu_info = self.system_info['gpu_info']
        if gpu_info['gpu_count'] > 0:
            # Base score for having GPU
            score *= 10.0
            
            # VRAM multiplier
            vram_gb = gpu_info.get('vram_gb', 0)
            score *= min(3.0, 1.0 + (vram_gb / 8.0))
            
            # CUDA/OptiX bonus
            if gpu_info.get('cuda', False):
                score *= 1.5
            if gpu_info.get('optix', False):
                score *= 2.0
            if gpu_info.get('metal', False):
                score *= 1.8
        else:
            # CPU-only penalty
            score *= 0.3
        
        # CPU cores multiplier
        cpu_cores = self.system_info['cpu_cores']
        score *= min(2.0, 1.0 + (cpu_cores / 16.0))
        
        # RAM multiplier
        ram_gb = self.system_info['total_ram_gb']
        score *= min(1.5, 1.0 + (ram_gb / 32.0))
        
        return round(score, 2)
    
    def register_with_coordinator(self):
        """Register this worker with the coordinator and get credentials"""
        try:
            # Get system info
            sys_info = get_system_info()
            
            # Prepare registration data
            registration_data = {
                'hostname': sys_info['hostname'],
                'system_info': sys_info,
                'port': 0,  # We're a client, not a server
                'render_speed_score': self.render_speed_score,
                'current_load': self.current_load
            }
            
            # Send registration request
            url = f"http://{self.coordinator_host}:{COORDINATOR_WEB_PORT}/api/register_worker"
            
            # If we have an admin token, use it
            admin_token = input("Enter admin API token (or press Enter to skip): ").strip()
            
            headers = {'Content-Type': 'application/json'}
            if admin_token:
                headers['X-API-Token'] = admin_token
            
            req = urllib.request.Request(
                url,
                data=json.dumps(registration_data).encode('utf-8'),
                headers=headers,
                method='POST'
            )
            
            with urllib.request.urlopen(req, timeout=30) as response:
                if response.status == 200:
                    result = json.loads(response.read().decode('utf-8'))
                    
                    if result.get('success'):
                        self.worker_id = result['worker_id']
                        self.api_token = result['api_token']
                        
                        # Save credentials to file for future use
                        creds_file = WORK_DIR / 'worker_credentials.json'
                        with open(creds_file, 'w') as f:
                            json.dump({
                                'worker_id': self.worker_id,
                                'api_token': self.api_token,
                                'coordinator_host': self.coordinator_host,
                                'registered_at': datetime.now().isoformat()
                            }, f, indent=2)
                        
                        print(f"✅ Registered as worker: {self.worker_id}")
                        print(f"🔑 API Token saved to {creds_file}")
                        return True
                    else:
                        print(f"❌ Registration failed: {result.get('error', 'Unknown error')}")
                        return False
                else:
                    print(f"❌ Registration failed with status: {response.status}")
                    return False
                    
        except urllib.error.URLError as e:
            print(f"❌ Network error during registration: {e}")
            return False
        except Exception as e:
            print(f"❌ Registration error: {e}")
            return False
    
    def load_credentials(self):
        """Load saved credentials if they exist"""
        creds_file = WORK_DIR / 'worker_credentials.json'
        if creds_file.exists():
            try:
                with open(creds_file, 'r') as f:
                    creds = json.load(f)
                
                # Check if coordinator host matches
                if creds.get('coordinator_host') == self.coordinator_host:
                    self.worker_id = creds['worker_id']
                    self.api_token = creds['api_token']
                    print(f"✅ Loaded credentials for worker: {self.worker_id}")
                    return True
                else:
                    print("⚠️  Saved credentials are for a different coordinator")
                    return False
                    
            except Exception as e:
                print(f"⚠️  Error loading credentials: {e}")
                return False
        
        return False
    
    def connect(self):
        """Connect to coordinator and authenticate"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(30.0)
            self.socket.connect((self.coordinator_host, self.coordinator_port))
            
            # Authenticate with token
            auth_msg = f"AUTH:{self.api_token}\n"
            self.socket.send(auth_msg.encode('utf-8'))
            
            response = self.socket.recv(1024).decode('utf-8').strip()
            
            if response == 'AUTH_OK':
                print(f"✅ Connected and authenticated to coordinator")
                self.connected = True
                
                # Send initial stats
                # self.send_stats_update()
                
                return True
            else:
                print(f"❌ Authentication failed: {response}")
                return False
                
        except socket.timeout:
            print("❌ Connection timeout")
            return False
        except ConnectionRefusedError:
            print(f"❌ Connection refused. Is coordinator running at {self.coordinator_host}:{self.coordinator_port}?")
            return False
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return False
    
    def send_stats_update(self):
        """Send updated statistics to coordinator"""
        if not self.connected or not self.socket:
            return
        
        try:
            stats = {
                'gpu_info': self.system_info['gpu_info'],
                'cpu_cores': self.system_info['cpu_cores'],
                'total_ram_gb': self.system_info['total_ram_gb'],
                'render_speed_score': self.render_speed_score,
                'current_load': self.current_load,
                'blender_version': get_blender_version()
            }
            
            self.socket.send(b'UPDATE_STATS\n')
            self.socket.send(json.dumps(stats).encode('utf-8'))
            
            response = self.socket.recv(1024).decode('utf-8').strip()
            if response == 'STATS_UPDATED':
                print("📊 Stats updated with coordinator")
            else:
                print(f"⚠️  Stats update response: {response}")
                
        except Exception as e:
            print(f"⚠️  Error updating stats: {e}")
    
    def download_blend_file(self, job_uuid, blend_filename):
        """Download blend file from coordinator"""
        try:
            url = f"http://{self.coordinator_host}:{COORDINATOR_WEB_PORT}/api/get_blend_file/{job_uuid}"
            
            headers = {'X-Worker-Token': self.api_token}
            req = urllib.request.Request(url, headers=headers)
            
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(req, timeout=60) as response:
                        if response.status == 200:
                            # FIXED: Don't create nested worker_data directories
                            # Create directory directly in current directory
                            job_dir = WORK_DIR / job_uuid
                            job_dir.mkdir(parents=True, exist_ok=True)
                            blend_path = job_dir / blend_filename
                            
                            print(f"📥 Downloading to: {blend_path.absolute()}")
                            
                            total_size = int(response.headers.get('content-length', 0))
                            print(f"📥 Downloading {blend_filename} ({total_size/1024/1024:.1f} MB)...")
                            
                            with open(blend_path, 'wb') as f:
                                downloaded = 0
                                chunk_size = 8192 * 8  # 64KB chunks
                                
                                while True:
                                    chunk = response.read(chunk_size)
                                    if not chunk:
                                        break
                                    f.write(chunk)
                                    downloaded += len(chunk)
                                    
                                    if total_size > 0 and attempt == 0:
                                        percent = (downloaded / total_size) * 100
                                        if downloaded % (1024*1024*10) < chunk_size:  # Update every 10MB
                                            print(f"  {percent:.1f}% ({downloaded/1024/1024:.1f}/{total_size/1024/1024:.1f} MB)", end='\r')
                                
                                if total_size > 0:
                                    print()  # New line after progress
                            
                            print(f"✅ Downloaded to: {blend_path.absolute()}")
                            
                            # Verify file integrity
                            file_size = blend_path.stat().st_size
                            if total_size > 0 and abs(file_size - total_size) > 1024:  # 1KB tolerance
                                print(f"⚠️  File size mismatch: {file_size} vs {total_size}")
                            
                            return blend_path
                        else:
                            print(f"❌ Download failed with status: {response.status}")
                            if attempt < 2:
                                time.sleep(2)
                                continue
                            return None
                            
                except urllib.error.URLError as e:
                    print(f"❌ Download error (attempt {attempt+1}/3): {e}")
                    if attempt < 2:
                        time.sleep(2)
                        continue
                    return None
                    
        except Exception as e:
            print(f"❌ Unexpected download error: {e}")
            return None
    
    def upload_frame(self, frame_id, file_path):
        """Upload rendered frame back to coordinator"""
        try:
            if not file_path.exists():
                print(f"❌ Frame file not found: {file_path}")
                return False
            
            # Prepare multipart form data
            boundary = '----WebKitFormBoundary' + ''.join([str(i) for i in range(10)])
            
            with open(file_path, 'rb') as f:
                file_content = f.read()
            
            # Build request body
            body = []
            body.append(f'--{boundary}\r\n'.encode())
            body.append(f'Content-Disposition: form-data; name="frame_id"\r\n\r\n'.encode())
            body.append(f'{frame_id}\r\n'.encode())
            
            body.append(f'--{boundary}\r\n'.encode())
            body.append(f'Content-Disposition: form-data; name="frame_file"; filename="{file_path.name}"\r\n'.encode())
            body.append(f'Content-Type: application/octet-stream\r\n\r\n'.encode())
            body.append(file_content)
            body.append(f'\r\n--{boundary}--\r\n'.encode())
            
            # Send request
            conn = http.client.HTTPConnection(self.coordinator_host, COORDINATOR_UPLOAD_PORT, timeout=60)
            conn.putrequest('POST', '/')
            conn.putheader('Content-Type', f'multipart/form-data; boundary={boundary}')
            conn.putheader('X-Worker-Token', self.api_token)
            conn.putheader('Content-Length', str(sum(len(part) for part in body)))
            conn.endheaders()
            
            for part in body:
                conn.send(part)
            
            # Get response
            response = conn.getresponse()
            response_data = response.read().decode('utf-8')
            
            if response.status == 200:
                result = json.loads(response_data)
                if result.get('success'):
                    file_size = file_path.stat().st_size
                    print(f"✅ Frame uploaded successfully ({file_size/1024:.1f} KB)")
                    return True
                else:
                    print(f"❌ Upload failed: {result.get('error', 'Unknown error')}")
                    return False
            else:
                print(f"❌ Upload failed with status {response.status}: {response_data}")
                return False
                
        except Exception as e:
            print(f"❌ Upload error: {e}")
            return False
    
    def render_frame(self, job_uuid, frame_number, output_format, blend_path, frame_id, cycles_device):
        # Convert blend_path to Path object if it isn't already
        if isinstance(blend_path, str):
            blend_path = Path(blend_path)
        
        # Make sure blend_path is absolute
        blend_path = blend_path.absolute()
        
        # Use absolute path for WORK_DIR
        output_dir = WORK_DIR.absolute() / job_uuid / 'frames'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Map format to extension
        format_to_ext = {
            'PNG': 'png',
            'JPEG': 'jpg',
            'OPEN_EXR': 'exr',
            'TIFF': 'tif'
        }
        ext = format_to_ext.get(output_format, 'png')
        
        # The file Blender will actually create - USE ABSOLUTE PATH
        actual_file = output_dir / f"frame_{frame_number:04d}.{ext}"
        
        # The pattern we give to Blender - USE ABSOLUTE PATH
        output_pattern = output_dir / "frame_"
        
        cmd = [
            BLENDER_PATH, 
            "-b",
            str(blend_path),  # This should now be an absolute path
            "-o", str(output_pattern),  # Absolute path
            "-F", output_format,
            "-f", str(frame_number),
        ]

        # Inject GPU backend
        if cycles_device in ("OPTIX", "CUDA", "HIP", "METAL"):
            cmd += ["--", "--cycles-device", cycles_device]
            
        print(f"Using GPU: {cycles_device}")
        print(f"Blend file path: {blend_path}")
        print(f"Output directory: {output_dir}")
        print(f"Expected file: {actual_file}")

        # Debug: check if blend file exists
        if not blend_path.exists():
            print(f"❌ ERROR: Blend file does not exist at: {blend_path}")
            print(f"    Looking for file: {blend_path.name}")
            print(f"    In directory: {blend_path.parent}")
            if blend_path.parent.exists():
                print(f"    Directory exists. Contents:")
                for f in blend_path.parent.glob("*"):
                    print(f"      - {f.name}")
            else:
                print(f"    Directory does not exist!")
            return False, 0, cycles_device, f"Blend file not found: {blend_path}"

        start = time.time()

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=False,
            bufsize=1,
            cwd=str(WORK_DIR)  # Make sure we're in the right directory
        )

        # Read output
        for raw_line in process.stdout:
            try:
                line = raw_line.decode('utf-8', errors='replace')
            except UnicodeDecodeError:
                line = raw_line.decode('latin-1', errors='replace')
            print(line, end="")

        process.wait()
        duration = time.time() - start

        if process.returncode != 0:
            return False, duration, cycles_device, f"Blender error code: {process.returncode}"

        # Check for the actual file Blender created
        if not actual_file.exists():
            # Debug: list directory contents
            print(f"❌ File not found: {actual_file}")
            print(f"📁 Directory contents of {output_dir}:")
            if output_dir.exists():
                for f in output_dir.glob("*"):
                    print(f"  - {f.name} (size: {f.stat().st_size} bytes)")
            else:
                print(f"  Directory does not exist: {output_dir}")
            return False, duration, cycles_device, f"File not found: {actual_file}"

        print(f"✅ File found: {actual_file} (size: {actual_file.stat().st_size} bytes)")
        return True, duration, cycles_device, actual_file
        
    def process_tasks(self):
        """Main task processing loop - FIXED TO WAIT FOR UPLOAD_NOW"""
        while not self.stop_event.is_set() and self.connected:
            try:
                self.socket.settimeout(300.0)
                
                # Wait for coordinator message
                data = self.socket.recv(1024).decode('utf-8').strip()
                print(f"📨 [{time.strftime('%H:%M:%S')}] Received: {repr(data)}")
                
                if data == 'READY?':
                    print(f"📤 [{time.strftime('%H:%M:%S')}] Requesting task...")
                    self.socket.send(b'REQUEST_TASK\n')
                    
                    task_response = self.socket.recv(1024).decode('utf-8').strip()
                    print(f"📋 [{time.strftime('%H:%M:%S')}] Task response: {task_response}")
                    
                    if task_response.startswith('TASK|'):
                        parts = task_response.split('|')
                        
                        if len(parts) >= 8 and parts[5] == 'LOCAL':
                            # LOCAL task
                            job_uuid = parts[1]
                            frame_number = int(parts[2])
                            output_format = parts[3]
                            blend_filename = parts[4]
                            blend_file_path = parts[6]
                            frame_id = int(parts[7])
                            
                            blend_path = Path(blend_file_path)
                            print(f"🎯 LOCAL task: frame {frame_number} of {job_uuid}")
                            print(f"📂 File: {blend_path}")
                            
                            if blend_path.exists():
                                self.current_load = min(1.0, self.current_load + 0.5)
                                
                                try:
                                    # Grab the gpu info
                                    gpu_info = GPUDetector.detect_gpu_info()
                                    cycles_device = gpu_info["cycles_device"]
                                
                                    # Render the frame
                                    success, render_time, gpu_used, output_file = self.render_frame(
                                        job_uuid, frame_number, output_format, blend_path, frame_id, cycles_device
                                    )
                                    
                                    self.current_load = max(0.0, self.current_load - 0.5)
                                    
                                    if success and output_file:
                                        # Send COMPLETE
                                        response = f'COMPLETE|{frame_id}|{render_time:.1f}|{gpu_used}\n'
                                        self.socket.send(response.encode('utf-8'))
                                        print(f"✅ [{time.strftime('%H:%M:%S')}] Task completed in {render_time:.1f}s")
                                        
                                        # Wait for coordinator to send UPLOAD_NOW
                                        print(f"⏳ [{time.strftime('%H:%M:%S')}] Waiting for UPLOAD_NOW command...")
                                        upload_data = self.socket.recv(1024).decode('utf-8').strip()
                                        print(f"📨 [{time.strftime('%H:%M:%S')}] Received upload command: {repr(upload_data)}")
                                        
                                        # Handle possible concatenated commands
                                        if 'UPLOAD_NOW' in upload_data:
                                            # Upload the rendered frame
                                            print(f"📤 [{time.strftime('%H:%M:%S')}] Uploading frame...")
                                            if self.upload_frame(frame_id, output_file):
                                                print(f"✅ [{time.strftime('%H:%M:%S')}] Frame uploaded")
                                            else:
                                                print(f"❌ [{time.strftime('%H:%M:%S')}] Frame upload failed")
                                            
                                            # If there's another command after UPLOAD_NOW, process it
                                            if '\n' in upload_data:
                                                remaining = upload_data.split('\n', 1)[1].strip()
                                                if remaining:
                                                    print(f"📨 [{time.strftime('%H:%M:%S')}] Processing remaining command: {remaining}")
                                                    # Set data to the remaining command for next loop iteration
                                                    # We'll handle this by simulating receiving it
                                                    data = remaining
                                                    continue  # Skip the rest and process this command
                                        else:
                                            print(f"❌ [{time.strftime('%H:%M:%S')}] Expected UPLOAD_NOW, got: {upload_data}")
                                    else:
                                        # Handle the error message properly
                                        error_msg = "Render failed"
                                        if isinstance(output_file, str):
                                            error_msg = output_file
                                        elif output_file is None:
                                            error_msg = "No output file returned"
                                        
                                        response = f'FAILED|{frame_id}|{error_msg}\n'
                                        self.socket.send(response.encode('utf-8'))
                                        print(f"❌ [{time.strftime('%H:%M:%S')}] Task failed: {error_msg}")
                                except Exception as e:
                                    error_msg = str(e)
                                    response = f'FAILED|{frame_id}|{error_msg}\n'
                                    self.socket.send(response.encode('utf-8'))
                                    print(f"❌ [{time.strftime('%H:%M:%S')}] Task error: {error_msg}")
                            else:
                                response = f'FAILED|{frame_id}|Local blend file not found\n'
                                self.socket.send(response.encode('utf-8'))
                                print(f"❌ [{time.strftime('%H:%M:%S')}] Blend file not found: {blend_path}")
                        
                        elif len(parts) >= 7 and parts[5] == 'REMOTE':
                            # REMOTE task
                            job_uuid = parts[1]
                            frame_number = int(parts[2])
                            output_format = parts[3]
                            blend_filename = parts[4]
                            frame_id = int(parts[6])
                            
                            print(f"🎯 REMOTE task: frame {frame_number} of {job_uuid}")
                            
                            self.current_load = min(1.0, self.current_load + 0.5)
                            
                            # Download blend file
                            blend_path = self.download_blend_file(job_uuid, blend_filename)
                            
                            if blend_path:
                                try:
                                    # Need to get cycles_device for remote tasks too!
                                    gpu_info = GPUDetector.detect_gpu_info()
                                    cycles_device = gpu_info["cycles_device"]
                                    
                                    # Render the frame
                                    success, render_time, gpu_used, output_file = self.render_frame(
                                        job_uuid, frame_number, output_format, blend_path, frame_id, cycles_device
                                    )
                                    
                                    self.current_load = max(0.0, self.current_load - 0.5)
                                    
                                    if success and output_file:
                                        # Send COMPLETE
                                        response = f'COMPLETE|{frame_id}|{render_time:.1f}|{gpu_used}\n'
                                        self.socket.send(response.encode('utf-8'))
                                        print(f"✅ [{time.strftime('%H:%M:%S')}] Task completed in {render_time:.1f}s")
                                        
                                        # Wait for coordinator to send UPLOAD_NOW
                                        print(f"⏳ [{time.strftime('%H:%M:%S')}] Waiting for UPLOAD_NOW command...")
                                        upload_data = self.socket.recv(1024).decode('utf-8').strip()
                                        print(f"📨 [{time.strftime('%H:%M:%S')}] Received upload command: {repr(upload_data)}")
                                        
                                        # Handle possible concatenated commands
                                        if 'UPLOAD_NOW' in upload_data:
                                            # Upload the rendered frame
                                            print(f"📤 [{time.strftime('%H:%M:%S')}] Uploading frame...")
                                            if self.upload_frame(frame_id, output_file):
                                                print(f"✅ [{time.strftime('%H:%M:%S')}] Frame uploaded")
                                            else:
                                                print(f"❌ [{time.strftime('%H:%M:%S')}] Frame upload failed")
                                            
                                            # If there's another command after UPLOAD_NOW, process it
                                            if '\n' in upload_data:
                                                remaining = upload_data.split('\n', 1)[1].strip()
                                                if remaining:
                                                    print(f"📨 [{time.strftime('%H:%M:%S')}] Processing remaining command: {remaining}")
                                                    # Set data to the remaining command for next loop iteration
                                                    data = remaining
                                                    continue  # Skip the rest and process this command
                                        else:
                                            print(f"❌ [{time.strftime('%H:%M:%S')}] Expected UPLOAD_NOW, got: {upload_data}")
                                    else:
                                        # Handle the error message properly
                                        error_msg = "Render failed"
                                        if isinstance(output_file, str):
                                            error_msg = output_file
                                        elif output_file is None:
                                            error_msg = "No output file returned"
                                        
                                        response = f'FAILED|{frame_id}|{error_msg}\n'
                                        self.socket.send(response.encode('utf-8'))
                                        print(f"❌ [{time.strftime('%H:%M:%S')}] Task failed: {error_msg}")
                                except Exception as e:
                                    error_msg = str(e)
                                    response = f'FAILED|{frame_id}|{error_msg}\n'
                                    self.socket.send(response.encode('utf-8'))
                                    print(f"❌ [{time.strftime('%H:%M:%S')}] Task error: {error_msg}")
                            else:
                                response = f'FAILED|{frame_id}|Could not download blend file\n'
                                self.socket.send(response.encode('utf-8'))
                                print(f"❌ [{time.strftime('%H:%M:%S')}] Could not download blend file")
                        
                        else:
                            print(f"❌ [{time.strftime('%H:%M:%S')}] Invalid task format: {task_response}")
                            self.socket.send(b'FAILED|0|Invalid task format\n')
                    
                    elif task_response == 'NO_TASK':
                        print(f"⏳ [{time.strftime('%H:%M:%S')}] No tasks available")
                        time.sleep(5)
                        self.socket.send(b'HEARTBEAT\n')
                        heartbeat_response = self.socket.recv(1024).decode('utf-8').strip()
                        print(f"❤️  [{time.strftime('%H:%M:%S')}] Heartbeat response: {heartbeat_response}")
                    
                    else:
                        print(f"⚠️  [{time.strftime('%H:%M:%S')}] Unexpected task response: {task_response}")
                        
                elif data == 'HEARTBEAT_CHECK':
                    print(f"❤️  [{time.strftime('%H:%M:%S')}] Heartbeat check received")
                    self.socket.send(b'ALIVE\n')
                    
                elif data == 'ALIVE':
                    print(f"❤️  [{time.strftime('%H:%M:%S')}] Alive confirmation received")
                    
                else:
                    print(f"⚠️  [{time.strftime('%H:%M:%S')}] Unexpected message: {data}")
                    
            except socket.timeout:
                print(f"⏰ [{time.strftime('%H:%M:%S')}] Socket timeout - sending heartbeat")
                try:
                    self.socket.send(b'HEARTBEAT\n')
                    response = self.socket.recv(1024).decode('utf-8').strip()
                    if response != 'ALIVE':
                        print(f"❌ [{time.strftime('%H:%M:%S')}] Heartbeat failed")
                        self.connected = False
                    else:
                        print(f"❤️  [{time.strftime('%H:%M:%S')}] Heartbeat succeeded")
                except Exception as e:
                    print(f"❌ [{time.strftime('%H:%M:%S')}] Heartbeat error: {e}")
                    self.connected = False
                    
            except ConnectionResetError:
                print(f"❌ [{time.strftime('%H:%M:%S')}] Connection reset by coordinator")
                self.connected = False
                    
            except Exception as e:
                print(f"❌ [{time.strftime('%H:%M:%S')}] Error in task loop: {e}")
                import traceback
                traceback.print_exc()
                self.connected = False
            
    def run(self):
        """Main worker loop with smart reconnection"""
        reconnect_attempts = 0
        max_reconnect_attempts = 20
        
        print(f"\n📊 System Information:")
        print(f"   Hostname: {self.system_info['hostname']}")
        print(f"   CPU Cores: {self.system_info['cpu_cores']}")
        print(f"   RAM: {self.system_info['total_ram_gb']:.1f} GB")
        print(f"   GPUs: {self.system_info['gpu_info']['gpu_count']}")
        # print(f"   Render Device: {self.system_info['render_device']}")
        print(f"   Render Device: {self.system_info.get('cycles_device', 'CPU')}")
        print(f"   Render Speed Score: {self.render_speed_score:.2f}")
        
        while not self.stop_event.is_set():
            if not self.connected:
                if reconnect_attempts >= max_reconnect_attempts:
                    print(f"❌ Max reconnection attempts ({max_reconnect_attempts}) reached. Giving up.")
                    break
                
                print(f"\nAttempting to connect to coordinator (attempt {reconnect_attempts + 1}/{max_reconnect_attempts})...")
                
                if self.connect():
                    reconnect_attempts = 0
                    print("✅ Connected successfully!")
                else:
                    reconnect_attempts += 1
                    wait_time = min(60, reconnect_attempts * 5)  # Exponential backoff
                    print(f"⏰ Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                    continue
            
            # Process tasks
            self.process_tasks()
            
            # If disconnected, wait before reconnecting
            if not self.connected:
                time.sleep(5)
    
    def stop(self):
        """Stop the worker cleanly"""
        self.stop_event.set()
        if self.socket:
            try:
                self.socket.send(b'DISCONNECT\n')
                self.socket.close()
            except:
                pass
        print("\n👋 PRO Worker stopped")

# ===== Main Worker Function =====
def main():
    global BLENDER_PATH, WORKER_ID, API_TOKEN
    
    print('\n' + '='*60)
    print('BLENDER 5.0 RENDER FARM PRO - WORKER')
    print('='*60 + '\n')
    
    # Get coordinator address
    default_coordinator = 'localhost'
    coordinator_host = input(f'Coordinator IP address [{default_coordinator}]: ').strip()
    if not coordinator_host:
        coordinator_host = default_coordinator
    
    # Auto-detect same machine
    if coordinator_host != 'localhost' and coordinator_host != '127.0.0.1':
        try:
            import socket
            local_ip = socket.gethostbyname(socket.gethostname())
            if coordinator_host == local_ip:
                print(f"ℹ️  Coordinator IP is this machine. Using 'localhost' for better performance.")
                coordinator_host = 'localhost'
        except:
            pass
    
    # Get Blender path
    blender_path = input('Blender 5.0 executable path: ').strip()
    
    if not os.path.exists(blender_path):
        print(f"❌ Error: Blender not found at {blender_path}")
        print("Please provide the correct path to Blender 5.0 executable.")
        return
    
    BLENDER_PATH = blender_path
    blender_version = get_blender_version()
    print(f"✅ {blender_version}")
    
    # Create worker instance
    worker = ProRenderWorker(coordinator_host, COORDINATOR_NETWORK_PORT)
    
    # Try to load saved credentials
    if not worker.load_credentials():
        # Need to register
        print("\n🔑 No saved credentials found. Need to register with coordinator.")
        print("You'll need an admin API token from the coordinator.")
        
        if worker.register_with_coordinator():
            print("✅ Registration successful!")
        else:
            print("❌ Registration failed. Cannot continue.")
            return
    
    # Update worker's API token
    API_TOKEN = worker.api_token
    WORKER_ID = worker.worker_id
    
    print(f"\n🚀 Starting PRO Worker: {WORKER_ID}")
    print(f"🔧 Blender: {blender_path}")
    print(f"📍 Coordinator: {coordinator_host}")
    print(f"📁 Work directory: {WORK_DIR.absolute()}")
    print('\n' + '-'*60)
    print('Press Ctrl+C to stop\n')
    
    try:
        worker.run()
    except KeyboardInterrupt:
        print("\n🛑 Shutdown requested...")
        worker.stop()
    except Exception as e:
        print(f"\n💥 Fatal error: {e}")
        import traceback
        traceback.print_exc()
        worker.stop()

if __name__ == '__main__':
    main()