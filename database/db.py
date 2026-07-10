import os
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dermai.db")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")   # prevents "database locked" on concurrent Streamlit reruns
    return conn


# ──────────────────────────────────────────────
# INITIALISATION
# ──────────────────────────────────────────────
def init_db() -> None:
    """Create tables if they don't already exist."""
    conn = _connect()
    try:
        cur = conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS patients (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT    NOT NULL,
                age         INTEGER,
                gender      TEXT,
                contact     TEXT,
                created_at  TEXT
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS records (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id     INTEGER NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
                scan_date      TEXT,
                disease        TEXT,
                confidence     REAL,
                affected_area  REAL,
                severity       TEXT,
                severity_score REAL,
                notes          TEXT,
                image_path     TEXT
            )
        """)

        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_records_patient_id ON records (patient_id)
        """)

        conn.commit()
    finally:
        conn.close()


# ──────────────────────────────────────────────
# PATIENT CRUD
# ──────────────────────────────────────────────
def add_patient(name: str, age: int, gender: str, contact: str) -> int:
    """Insert a new patient and return the new row id."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = _connect()
    try:
        cur = conn.execute(
            "INSERT INTO patients (name, age, gender, contact, created_at) VALUES (?, ?, ?, ?, ?)",
            (name, age, gender, contact, now),
        )
        patient_id = cur.lastrowid
        conn.commit()
        return patient_id
    finally:
        conn.close()


def get_all_patients() -> List[Dict]:
    """Return all patients as a list of dicts, newest first."""
    conn = _connect()
    try:
        rows = conn.execute(
            """
            SELECT p.*,
                   COALESCE(COUNT(r.id), 0)  AS scan_count,
                   MAX(r.scan_date)           AS last_scan
            FROM patients p
            LEFT JOIN records r ON r.patient_id = p.id
            GROUP BY p.id
            ORDER BY p.created_at DESC
            """
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_patient(patient_id: int) -> Optional[Dict]:
    """Return a single patient dict or None."""
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT * FROM patients WHERE id = ?", (patient_id,)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def update_patient(patient_id: int, name: str, age: int, gender: str, contact: str) -> bool:
    """Update an existing patient's details. Returns True if a row was updated."""
    conn = _connect()
    try:
        cur = conn.execute(
            """
            UPDATE patients
               SET name=?, age=?, gender=?, contact=?
             WHERE id=?
            """,
            (name, age, gender, contact, patient_id),
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def delete_patient(patient_id: int) -> bool:
    """
    Delete a patient and all their records (CASCADE).
    Also removes scan images from disk to prevent orphaned files.
    Returns True if the patient existed and was deleted.
    """
    # Collect image paths before deleting so we can clean up files
    records = get_records_for_patient(patient_id)
    image_paths = [r["image_path"] for r in records if r.get("image_path")]

    conn = _connect()
    try:
        cur = conn.execute("DELETE FROM patients WHERE id = ?", (patient_id,))
        conn.commit()
        deleted = cur.rowcount > 0
    finally:
        conn.close()

    if deleted:
        for path in image_paths:
            try:
                if os.path.isfile(path):
                    os.remove(path)
            except OSError:
                pass  # Non-critical — don't crash if file removal fails

    return deleted


# ──────────────────────────────────────────────
# RECORD CRUD
# ──────────────────────────────────────────────
def add_record(
    patient_id: int,
    disease: str,
    confidence: float,
    affected_area: float,
    severity: str,
    severity_score: float,
    notes: str,
    image_path: str,
) -> int:
    """Insert a new scan record and return its id."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = _connect()
    try:
        cur = conn.execute(
            """
            INSERT INTO records
                (patient_id, scan_date, disease, confidence, affected_area,
                 severity, severity_score, notes, image_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                patient_id,
                now,
                disease,
                round(float(confidence), 4),
                round(float(affected_area), 4),
                severity,
                round(float(severity_score), 4),
                notes,
                image_path,
            ),
        )
        record_id = cur.lastrowid
        conn.commit()
        return record_id
    finally:
        conn.close()


def get_record_by_id(record_id: int) -> Optional[Dict]:
    """Return a single record dict or None."""
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT r.*, p.name AS patient_name FROM records r JOIN patients p ON p.id=r.patient_id WHERE r.id=?",
            (record_id,),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_records_for_patient(patient_id: int) -> List[Dict]:
    """Return all scan records for a patient, oldest first."""
    conn = _connect()
    try:
        rows = conn.execute(
            """
            SELECT * FROM records
            WHERE patient_id = ?
            ORDER BY scan_date ASC
            """,
            (patient_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_all_records() -> List[Dict]:
    """Return all records joined with patient name, newest first."""
    conn = _connect()
    try:
        rows = conn.execute(
            """
            SELECT r.*, p.name AS patient_name
            FROM records r
            JOIN patients p ON p.id = r.patient_id
            ORDER BY r.scan_date DESC
            """
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def delete_record(record_id: int) -> bool:
    """
    Delete a single scan record and its image file.
    Returns True if a row was deleted.
    """
    rec = get_record_by_id(record_id)
    conn = _connect()
    try:
        cur = conn.execute("DELETE FROM records WHERE id = ?", (record_id,))
        conn.commit()
        deleted = cur.rowcount > 0
    finally:
        conn.close()

    if deleted and rec and rec.get("image_path"):
        try:
            if os.path.isfile(rec["image_path"]):
                os.remove(rec["image_path"])
        except OSError:
            pass

    return deleted


# ──────────────────────────────────────────────
# COUNTS
# ──────────────────────────────────────────────
def get_patient_count() -> int:
    conn = _connect()
    try:
        return conn.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
    finally:
        conn.close()


def get_record_count() -> int:
    conn = _connect()
    try:
        return conn.execute("SELECT COUNT(*) FROM records").fetchone()[0]
    finally:
        conn.close()