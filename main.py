import tkinter as tk
from tkinter import ttk
from tkinter import PhotoImage
from PIL import Image, ImageTk
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import cv2
import numpy as np
import sqlite3
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class EmotionDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Emotion Detector")
        self.root.configure(bg="white")
        self.face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.classifier = load_model('testing.h5')
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        
        self.fig, self.ax = plt.subplots(figsize=(8, 4))  # Ubah ukuran line chart disini
        # Inisialisasi line satu kali
        self.line, = self.ax.plot([], [], 'o-', label='Emotion', marker='o')  # Tambahkan marker untuk menunjukkan garis
        self.ax.legend(loc='upper right')
        self.ax.set_xlabel('Kategori Emosi')
        self.ax.set_ylabel('Percentage')
        self.canvas_chart = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas_chart.draw()

        self.ax.add_line(self.line)
        
        self.stop_flag = False

        # Update chart with initial data
        self.update_chart([], [])

        # SQLite database initialization
        self.conn = sqlite3.connect('emotion_database.db')
        self.create_table()

        self.video_source = 0  # Change this if you want to use a different video source
        self.cap = cv2.VideoCapture(self.video_source)

        self.create_widgets()

    def create_table(self):
        cursor = self.conn.cursor()
        cursor.execute('DROP TABLE IF EXISTS emotions')
        cursor.execute('''CREATE TABLE IF NOT EXISTS emotions
                          (id INTEGER PRIMARY KEY AUTOINCREMENT,
                           timestamp DATETIME,
                           emotion_category TEXT,
                           confidence REAL)''')
        self.conn.commit()

    def insert_data_to_database(self, category, confidence):
        timestamp = datetime.now()
        cursor = self.conn.cursor()
        cursor.execute("INSERT INTO emotions (timestamp, emotion_category, confidence) VALUES (?, ?, ?)",
                       (timestamp, category, confidence))
        self.conn.commit()

    def create_widgets(self):
        self.canvas = tk.Canvas(self.root, width=640, height=480, background='white')
        self.canvas.pack(side=tk.LEFT)

        # Table for displaying emotions
        self.tree = ttk.Treeview(self.root, columns=('Waktu', 'Kategori Emosi', 'Persentase'), show='headings')
        self.tree.heading('Waktu', text='Waktu')
        self.tree.heading('Kategori Emosi', text='Kategori Emosi')
        self.tree.heading('Persentase', text='Persentase')
        self.tree.pack(side=tk.RIGHT, padx=50)

        self.update()

        self.style = ttk.Style()
        self.style.configure("TButton", background="white", borderwidth=0, font=("Arial", 12))
        self.style.map("TButton", foreground=[('pressed', 'blue'), ('active', 'blue')])

        self.canvas_chart.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.stop_button = ttk.Button(self.root, text="Stop", command=self.stop_application, style="TButton")
        self.stop_button.pack(side=tk.LEFT, pady=20)

        self.quit_button = ttk.Button(self.root, text="Keluar", command=self.root.destroy, style="TButton")
        self.quit_button.pack(side=tk.BOTTOM, pady=20)
        
    def update_chart(self, categories, percentages):
        self.ax.clear()  # Hapus data sebelumnya pada chart
        self.line, = self.ax.plot(categories, percentages, 'o-', label='Jumlah', marker='o')  
        self.ax.legend(loc='upper right')
        self.ax.set_xlabel('Kategori Emosi')
        self.ax.set_ylabel('Persentase')
        self.ax.set_xticks(range(len(categories)))
        self.ax.set_xticklabels(categories, rotation=45, ha='right')
        self.ax.set_yticks(np.arange(0, 101, 10))
        self.ax.set_ylim(0, 100)

    def stop_application(self):
        self.stop_flag = True
        self.stop_video_capture()
        self.plot_chart()

    def plot_chart(self):
        categories, percentages = self.get_chart_data()
        self.update_chart(categories, percentages)
        self.canvas_chart.draw()  # Pindahkan pemanggilan draw ke sini

    def get_chart_data(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT emotion_category, AVG(confidence) FROM emotions GROUP BY emotion_category")
        rows = cursor.fetchall()
        categories = [row[0] for row in rows]
        percentages = [row[1] for row in rows]
        return categories, percentages

    def stop_video_capture(self):
        self.cap.release()  # Hentikan kamera

    def update(self):
        _, frame = self.cap.read()
        labels = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_classifier.detectMultiScale(gray)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                prediction = self.classifier.predict(roi)[0]
                label = self.emotion_labels[prediction.argmax()]
                confidence = max(prediction) * 100
                label_position = (x, y)
                cv2.putText(frame, f'{label}: {confidence:.2f}%', label_position,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Insert data to database
                self.insert_data_to_database(label, confidence)
                
                # Update table
                self.update_table()

                # Update chart
                self.update_chart(*self.get_chart_data())

            else:
                cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        self.photo = self.convert_frame_to_image(frame)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        self.root.after(10, self.update)

        # Periksa stop_flag, jika True, hentikan aplikasi
        if self.stop_flag:
            self.stop_video_capture()
    
    def convert_frame_to_image(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        img = ImageTk.PhotoImage(image=img)
        return img

    def update_table(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM emotions ORDER BY timestamp DESC LIMIT 1")
        row = cursor.fetchone()
        if row:
            self.tree.insert('', '0', values=(row[1], row[2], f'{row[3]:.2f}%'))


if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionDetectorApp(root)
    root.mainloop()
