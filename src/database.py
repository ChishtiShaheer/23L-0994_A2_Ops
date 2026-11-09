"""
Task 5: Database Trail (Audit Logging)
Author: 23L-0994
Project: 23L-0994_A2_Ops
Maintains an immutable queryable database log of all image classification decisions:
Timestamp, Filename, Verdict, Confidence, Model Version, and Client Metadata.
"""
import os
import sqlite3
from datetime import datetime

DB_PATH = os.getenv("DATABASE_PATH", "visionguard.db")

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                filename TEXT NOT NULL,
                verdict TEXT NOT NULL,
                confidence REAL NOT NULL,
                raw_score REAL NOT NULL,
                model_name TEXT NOT NULL,
                latency_ms REAL NOT NULL
            );
        """)
    conn.close()
    print(f"[+] Initialized audit database: {DB_PATH}")

def log_prediction(filename, verdict, confidence, raw_score, model_name, latency_ms):
    conn = get_db_connection()
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    with conn:
        conn.execute("""
            INSERT INTO audit_logs (timestamp, filename, verdict, confidence, raw_score, model_name, latency_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (timestamp, filename, verdict, confidence, raw_score, model_name, latency_ms))
    conn.close()

def get_all_logs(limit=100):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, timestamp, filename, verdict, confidence, raw_score, model_name, latency_ms
        FROM audit_logs
        ORDER BY id DESC
        LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

if __name__ == "__main__":
    init_db()
