import flet as ft
import cv2
import numpy as np
from rembg import remove, new_session
import threading
import os
import tkinter as tk
from tkinter import filedialog
import sys

# AI Session (Global so it loads once)
session = new_session("u2netp")

def main(page: ft.Page):
    page.title = "DSLR AI Video Studio"
    page.theme_mode = ft.ThemeMode.DARK
    page.vertical_alignment = ft.MainAxisAlignment.CENTER
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER

    selected_video_path = {"path": None}
    selected_file_text = ft.Text("कोई वीडियो सिलेक्ट नहीं की गई", size=13, color="grey")
    status_text = ft.Text("वीडियो चुनें और 'Start Processing' दबाएं", size=14, weight=ft.FontWeight.BOLD)
    progress_bar = ft.ProgressBar(width=300, visible=False, value=0)
    start_btn = ft.FilledButton("⚡ 1-Click DSLR AI Effect", icon=ft.Icons.AUTO_AWESOME, disabled=True, width=300, height=55)

    def process_video():
        try:
            input_path = selected_video_path["path"]
            output_folder = os.path.join(os.getcwd(), "Processed_Videos")
            if not os.path.exists(output_folder): os.makedirs(output_folder)
            
            output_path = os.path.join(output_folder, "output_dslr_ai.mp4")
            
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                raise Exception("वीडियो फाइल नहीं खुल रही है! (शायद फाइल करप्ट है)")

            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            max_process_frames = 75
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            for i in range(max_process_frames):
                ret, frame = cap.read()
                if not ret: break
                
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = remove(rgb_frame, session=session)
                alpha_mask = result[:, :, 3] / 255.0
                blurred_bg = cv2.GaussianBlur(frame, (45, 45), 0)
                
                for c in range(3):
                    frame[:, :, c] = (alpha_mask * frame[:, :, c] + (1.0 - alpha_mask) * blurred_bg[:, :, c]).astype(np.uint8)
                
                out.write(frame)
                
                # UI Update
                progress_bar.value = (i + 1) / max_process_frames
                status_text.value = f"⚡ फ्रेम: {i+1}/{max_process_frames}"
                page.update()
            
            cap.release()
            out.release()
            status_text.value = f"✅ तैयार! फोल्डर चेक करें: Processed_Videos"
            status_text.color = "green"
            
        except Exception as e:
            status_text.value = f"❌ एरर: {str(e)}"
            status_text.color = "red"
        finally:
            progress_bar.visible = False
            start_btn.disabled = False
            page.update()

    def pick_video(e):
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename()
        root.destroy()
        if file_path:
            selected_video_path["path"] = file_path
            selected_file_text.value = f"फ़ाइल: {os.path.basename(file_path)}"
            selected_file_text.color = "green"
            start_btn.disabled = False
            page.update()

    def start_process(e):
        progress_bar.visible = True
        status_text.value = "🔥 AI काम कर रहा है... ज़रा रुकें..."
        start_btn.disabled = True
        threading.Thread(target=process_video, daemon=True).start()

    start_btn.on_click = start_process

    page.add(
        ft.Text("DSLR AI Studio", size=26, weight=ft.FontWeight.BOLD),
        ft.FilledButton("📁 वीडियो चुनें", icon=ft.Icons.VIDEO_FILE, on_click=pick_video),
        selected_file_text,
        start_btn,
        progress_bar,
        status_text
    )

ft.app(target=main)