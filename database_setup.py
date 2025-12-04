#!/usr/bin/env python3
"""
Database setup for Blender Render Farm PRO
Run this once to initialize the database
"""

import sqlite3
import hashlib
import secrets
import json
from datetime import datetime

def init_database():
    """Initialize the SQLite database with all tables"""
    conn = sqlite3.connect('render_farm.db')
    cursor = conn.cursor()
    
    # Enable foreign keys and WAL mode for better performance
    cursor.execute("PRAGMA foreign_keys = ON")
    cursor.execute("PRAGMA journal_mode = WAL")
    
    # ===== USERS TABLE =====
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        api_token TEXT UNIQUE,
        is_admin BOOLEAN DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_login TIMESTAMP
    )
    ''')
    
    # ===== WORKERS TABLE =====
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS workers (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        worker_id TEXT UNIQUE NOT NULL,
        hostname TEXT NOT NULL,
        ip_address TEXT,
        port INTEGER,
        gpu_info TEXT,  -- JSON: {"cuda": true, "optix": true, "vram_gb": 8, ...}
        cpu_cores INTEGER,
        total_ram_gb REAL,
        render_speed_score REAL DEFAULT 1.0,
        current_load REAL DEFAULT 0.0,  -- 0.0 to 1.0
        status TEXT DEFAULT 'offline',  -- offline, idle, busy
        last_heartbeat TIMESTAMP,
        api_token TEXT UNIQUE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # ===== JOBS TABLE ===== (UPDATED with all required columns)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS jobs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        job_uuid TEXT UNIQUE NOT NULL,
        user_id INTEGER,
        job_name TEXT NOT NULL,
        blend_filename TEXT NOT NULL,
        blend_file_hash TEXT NOT NULL,
        blend_file_size INTEGER NOT NULL,
        start_frame INTEGER NOT NULL,
        end_frame INTEGER NOT NULL,
        output_format TEXT NOT NULL,
        render_settings TEXT,  -- JSON: {"samples": 128, "resolution": "1920x1080", ...}
        priority INTEGER DEFAULT 1,
        status TEXT DEFAULT 'pending',  -- pending, processing, completed, failed, cancelled
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        started_at TIMESTAMP,
        completed_at TIMESTAMP,
        cancelled_at TIMESTAMP,  -- NEW: For tracking when job was cancelled
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,  -- NEW: For tracking last update
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    ''')
    
    # ===== FRAMES TABLE ===== (UPDATED with updated_at column)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS frames (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        job_id INTEGER NOT NULL,
        frame_number INTEGER NOT NULL,
        worker_id INTEGER,
        status TEXT DEFAULT 'pending',  -- pending, assigned, rendering, completed, failed, cancelled
        assigned_at TIMESTAMP,
        started_at TIMESTAMP,
        completed_at TIMESTAMP,
        render_time_seconds REAL,
        file_path TEXT,
        file_size INTEGER,
        checksum TEXT,
        error_message TEXT,
        retry_count INTEGER DEFAULT 0,
        gpu_used TEXT,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,  -- NEW: For tracking last update
        UNIQUE(job_id, frame_number),
        FOREIGN KEY (job_id) REFERENCES jobs (id),
        FOREIGN KEY (worker_id) REFERENCES workers (id)
    )
    ''')
    
    # ===== JOB_QUEUE TABLE =====
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS job_queue (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        job_id INTEGER NOT NULL,
        frame_number INTEGER NOT NULL,
        worker_id INTEGER,
        priority INTEGER DEFAULT 0,
        assigned_at TIMESTAMP,
        status TEXT DEFAULT 'queued',  -- queued, assigned, processing
        gpu_requirements TEXT,  -- JSON: {"min_vram_gb": 4, "prefer_cuda": true}
        FOREIGN KEY (job_id) REFERENCES jobs (id),
        FOREIGN KEY (worker_id) REFERENCES workers (id)
    )
    ''')
    
    # ===== RENDER_LOGS TABLE =====
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS render_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        job_id INTEGER,
        frame_id INTEGER,
        worker_id INTEGER,
        log_level TEXT NOT NULL,  -- info, warning, error, debug
        message TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (job_id) REFERENCES jobs (id),
        FOREIGN KEY (frame_id) REFERENCES frames (id),
        FOREIGN KEY (worker_id) REFERENCES workers (id)
    )
    ''')
    
    # ===== INDEXES =====
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_frames_status ON frames(status)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_frames_job_id ON frames(job_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_workers_status ON workers(status)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_job_queue_status ON job_queue(status)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_job_queue_priority ON job_queue(priority DESC)')
    
    # ===== CREATE DEFAULT ADMIN USER =====
    # Default password: "admin123" - CHANGE THIS IN PRODUCTION!
    default_password = "BADMANowo123"
    password_hash = hashlib.sha256(default_password.encode()).hexdigest()
    api_token = secrets.token_urlsafe(32)
    
    try:
        cursor.execute('''
        INSERT OR IGNORE INTO users (username, password_hash, api_token, is_admin)
        VALUES (?, ?, ?, ?)
        ''', ('admin', password_hash, api_token, 1))
        
        print(f"✓ Created default admin user")
        print(f"  Username: admin")
        print(f"  Password: {default_password}")
        print(f"  API Token: {api_token}")
        print(f"  ⚠️  CHANGE THESE CREDENTIALS IN PRODUCTION!")
    except Exception as e:
        print(f"⚠️  Note: {e}")
    
    conn.commit()
    conn.close()
    
    print(f"\n✅ Database initialized successfully!")
    print(f"   Database file: render_farm.db")
    print(f"   Tables created: users, workers, jobs, frames, job_queue, render_logs")

if __name__ == '__main__':
    print("="*60)
    print("BLENDER RENDER FARM PRO - DATABASE SETUP")
    print("="*60)
    init_database()
