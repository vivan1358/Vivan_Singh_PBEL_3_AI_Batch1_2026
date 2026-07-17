"""
Tkinter front end. Kept deliberately simple: this window handles
registration forms, buttons, and record tables, while the webcam views
(registration capture, live attendance) open in their own OpenCV
windows, controlled with 'q'.
"""
import re
import tkinter as tk
from tkinter import ttk, messagebox

from src import db, face_capture, face_trainer, face_recognizer, reports

VALID_ID = re.compile(r"^[A-Za-z0-9_-]+$")


class AttendanceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Based Smart Attendance Monitoring System")
        self.root.geometry("520x560")
        self.root.resizable(False, False)

        tk.Label(root, text="Smart Attendance System", font=("Segoe UI", 18, "bold")).pack(pady=(24, 4))
        tk.Label(root, text="Face recognition attendance (OpenCV + LBPH)", font=("Segoe UI", 10)).pack(pady=(0, 20))

        btn_style = {"width": 32, "height": 2}
        tk.Button(root, text="Register New Student", command=self.register_student, **btn_style).pack(pady=6)
        tk.Button(root, text="View Registered Students", command=self.view_students, **btn_style).pack(pady=6)
        tk.Button(root, text="Train Recognition Model", command=self.train_model, **btn_style).pack(pady=6)
        tk.Button(root, text="Take Attendance", command=self.take_attendance, **btn_style).pack(pady=6)
        tk.Button(root, text="View Attendance Records", command=self.view_attendance, **btn_style).pack(pady=6)
        tk.Button(root, text="Attendance Report (Chart)", command=self.show_report, **btn_style).pack(pady=6)
        tk.Button(root, text="Exit", command=root.quit, **btn_style).pack(pady=6)

    # ---- Register ----
    def register_student(self):
        form = tk.Toplevel(self.root)
        form.title("Register Student")
        form.geometry("320x300")
        form.grab_set()

        fields = {}
        for label in ["Student ID", "Name", "Roll No", "Branch"]:
            tk.Label(form, text=label).pack(pady=(10, 0))
            entry = tk.Entry(form, width=30)
            entry.pack()
            fields[label] = entry

        def submit():
            student_id = fields["Student ID"].get().strip()
            name = fields["Name"].get().strip()
            roll_no = fields["Roll No"].get().strip()
            branch = fields["Branch"].get().strip()

            if not student_id or not name:
                messagebox.showwarning("Missing info", "Student ID and Name are required.")
                return
            if not VALID_ID.match(student_id):
                messagebox.showwarning("Invalid ID", "Student ID may only contain letters, numbers, - and _.")
                return

            form.destroy()
            messagebox.showinfo(
                "Get ready",
                "The camera will open. Look at it and slowly turn your head "
                "left/right for a few seconds. Press 'q' in that window to cancel early."
            )
            ok, msg = face_capture.capture_faces(student_id, name, roll_no, branch)
            if ok:
                messagebox.showinfo("Registered", msg + "\n\nRemember to retrain the model.")
            else:
                messagebox.showerror("Registration failed", msg)

        tk.Button(form, text="Start Capture", command=submit).pack(pady=20)

    def view_students(self):
        window = tk.Toplevel(self.root)
        window.title("Registered Students")
        window.geometry("640x360")

        columns = ("student_id", "name", "roll_no", "branch", "registered_on")
        tree = ttk.Treeview(window, columns=columns, show="headings")
        for col, label in zip(columns, ["Student ID", "Name", "Roll No", "Branch", "Registered On"]):
            tree.heading(col, text=label)
            tree.column(col, width=115)
        tree.pack(fill="both", expand=True, padx=10, pady=10)

        for row in db.get_all_students():
            tree.insert("", "end", values=row)

    # ---- Train ----
    def train_model(self):
        self.root.config(cursor="watch")
        self.root.update()
        ok, msg = face_trainer.train_model()
        self.root.config(cursor="")
        (messagebox.showinfo if ok else messagebox.showerror)(
            "Training complete" if ok else "Training failed", msg
        )

    # ---- Attendance ----
    def take_attendance(self):
        messagebox.showinfo(
            "Starting camera",
            "The camera window will open. Press 'q' in that window to stop the session."
        )
        ok, msg = face_recognizer.run_attendance()
        (messagebox.showinfo if ok else messagebox.showerror)(
            "Session ended" if ok else "Could not start", msg
        )

    def view_attendance(self):
        window = tk.Toplevel(self.root)
        window.title("Attendance Records")
        window.geometry("640x400")

        columns = ("student_id", "name", "date", "time", "status")
        tree = ttk.Treeview(window, columns=columns, show="headings")
        for col, label in zip(columns, ["Student ID", "Name", "Date", "Time", "Status"]):
            tree.heading(col, text=label)
            tree.column(col, width=110)
        tree.pack(fill="both", expand=True, padx=10, pady=10)

        for row in db.get_all_attendance():
            tree.insert("", "end", values=row)

        def export():
            path, msg = reports.export_attendance_csv()
            if path:
                messagebox.showinfo("Exported", f"{msg}\nSaved to: {path}")
            else:
                messagebox.showwarning("Nothing to export", msg)

        tk.Button(window, text="Export to CSV", command=export).pack(pady=(0, 10))

    def show_report(self):
        ok, msg = reports.attendance_percentage_chart()
        if not ok:
            messagebox.showwarning("Report", msg)


def launch():
    root = tk.Tk()
    AttendanceApp(root)
    root.mainloop()
