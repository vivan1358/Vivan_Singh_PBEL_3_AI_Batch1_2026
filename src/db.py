"""
SQLite persistence for the attendance system.

Two tables:
  students   -- one row per registered person
  attendance -- one row per (student, date) attendance mark;
                a UNIQUE constraint stops the same student being
                marked present twice on the same day.
"""
import sqlite3
from datetime import datetime

import config


def get_connection():
    conn = sqlite3.connect(config.DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            roll_no TEXT,
            branch TEXT,
            registered_on TEXT NOT NULL
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT NOT NULL,
            name TEXT NOT NULL,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            status TEXT DEFAULT 'Present',
            UNIQUE(student_id, date)
        )
    """)
    conn.commit()
    conn.close()


def student_exists(student_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM students WHERE student_id = ?", (student_id,))
    exists = cur.fetchone() is not None
    conn.close()
    return exists


def add_student(student_id, name, roll_no, branch):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO students (student_id, name, roll_no, branch, registered_on) "
        "VALUES (?, ?, ?, ?, ?)",
        (student_id, name, roll_no, branch, datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
    )
    conn.commit()
    conn.close()


def get_all_students():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT student_id, name, roll_no, branch, registered_on FROM students ORDER BY name")
    rows = cur.fetchall()
    conn.close()
    return rows


def mark_attendance(student_id, name):
    """Marks attendance for today. Returns False if already marked today."""
    conn = get_connection()
    cur = conn.cursor()
    now = datetime.now()
    try:
        cur.execute(
            "INSERT INTO attendance (student_id, name, date, time, status) "
            "VALUES (?, ?, ?, ?, 'Present')",
            (student_id, name, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")),
        )
        conn.commit()
        marked = True
    except sqlite3.IntegrityError:
        marked = False
    conn.close()
    return marked


def get_attendance_by_date(date_str):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT student_id, name, date, time, status FROM attendance WHERE date = ? ORDER BY time",
        (date_str,),
    )
    rows = cur.fetchall()
    conn.close()
    return rows


def get_all_attendance():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT student_id, name, date, time, status FROM attendance ORDER BY date DESC, time DESC")
    rows = cur.fetchall()
    conn.close()
    return rows


def get_attendance_summary():
    """Returns (rows of [student_id, name, days_present], total_distinct_days_recorded)."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(DISTINCT date) FROM attendance")
    total_days = cur.fetchone()[0] or 0
    cur.execute("""
        SELECT student_id, name, COUNT(DISTINCT date) as days_present
        FROM attendance GROUP BY student_id, name
    """)
    rows = cur.fetchall()
    conn.close()
    return rows, total_days
