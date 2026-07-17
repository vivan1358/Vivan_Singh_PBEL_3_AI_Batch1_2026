"""
Turns raw attendance rows into things a human wants: a CSV file, or a
bar chart of attendance percentage per student.
"""
import os
from datetime import datetime

import pandas as pd

import config
from src import db


def export_attendance_csv(date_str=None):
    rows = db.get_attendance_by_date(date_str) if date_str else db.get_all_attendance()
    if not rows:
        return None, "No attendance records to export."

    df = pd.DataFrame(rows, columns=["Student ID", "Name", "Date", "Time", "Status"])
    filename = f"attendance_{date_str or 'all'}_{datetime.now().strftime('%H%M%S')}.csv"
    filepath = os.path.join(config.EXPORT_DIR, filename)
    df.to_csv(filepath, index=False)

    return filepath, f"Exported {len(df)} record(s)."


def attendance_percentage_chart():
    import matplotlib.pyplot as plt

    summary_rows, total_days = db.get_attendance_summary()
    if not summary_rows or total_days == 0:
        return False, "Not enough attendance data yet."

    df = pd.DataFrame(summary_rows, columns=["Student ID", "Name", "Days Present"])
    df["Attendance %"] = (df["Days Present"] / total_days * 100).round(1)

    plt.figure(figsize=(8, 5))
    bars = plt.bar(df["Name"], df["Attendance %"], color="#4C72B0")
    plt.ylabel("Attendance %")
    plt.title(f"Attendance Summary (over {total_days} recorded day(s))")
    plt.ylim(0, 100)
    plt.xticks(rotation=30, ha="right")
    for bar, pct in zip(bars, df["Attendance %"]):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{pct}%", ha="center", fontsize=8)
    plt.tight_layout()
    plt.show()

    return True, "Chart displayed."
