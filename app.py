import os
import time
import winsound
import threading
import numpy as np
import sounddevice as sd
import pyperclip
import pyautogui
import customtkinter as ctk
from scipy.io.wavfile import write
from pynput import keyboard
from dotenv import load_dotenv
from groq import Groq

# --- CONFIGURATION ---
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
FILENAME = "buffer_segment.wav"
FS = 44100
SYSTEM_PROMPT = (
    "You are a developer's prompt architect. Convert messy voice transcripts into "
    "concise, professional technical instructions. If it's a greeting, just clean the grammar."
)

# --- UI THEME ---
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class PromptBufferUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Window Setup
    
        self.geometry("450x250")
        self.attributes("-topmost", True)  # Keep on top of VS Code
        self.resizable(False, False)

        # State
        self.recording = False
        self.audio_data = []

        # UI Elements
        self.label = ctk.CTkLabel(self, text="HOLD [RIGHT CTRL] TO SPEAK", font=("Inter", 14, "bold"))
        self.label.pack(pady=(20, 10))

        self.status_indicator = ctk.CTkButton(self, text="", width=20, height=20, corner_radius=10, 
                                              fg_color="gray", state="disabled")
        self.status_indicator.pack(pady=5)

        self.raw_text_view = ctk.CTkTextbox(self, width=400, height=60, font=("Consolas", 11))
        self.raw_text_view.pack(pady=10)
        self.raw_text_view.insert("0.0", "Waiting for input...")

        self.footer = ctk.CTkLabel(self, text="Start Within Automation v1.0", font=("Inter", 10), text_color="gray")
        self.footer.pack(side="bottom", pady=5)

        # Start Keyboard Listener
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()

        # Start Audio Stream
        self.stream = sd.InputStream(samplerate=FS, channels=1, callback=self.record_callback)
        self.stream.start()

    def record_callback(self, indata, frames, time, status):
        if self.recording:
            self.audio_data.append(indata.copy())

    def on_press(self, key):
        if key == keyboard.Key.ctrl_r and not self.recording:
            self.recording = True
            self.audio_data = []
            self.after(0, lambda: self.update_ui_state("recording"))
            winsound.Beep(600, 100)

    def on_release(self, key):
        if key == keyboard.Key.ctrl_r and self.recording:
            self.recording = False
            self.after(0, lambda: self.update_ui_state("processing"))
            # Run processing in a background thread to keep UI responsive
            threading.Thread(target=self.process_audio).start()

    def update_ui_state(self, state):
        if state == "recording":
            self.status_indicator.configure(fg_color="#ff4b4b") # Red
            self.label.configure(text="🔴 RECORDING...", text_color="#ff4b4b")
        elif state == "processing":
            self.status_indicator.configure(fg_color="#f1c40f") # Yellow
            self.label.configure(text="🟡 OPTIMIZING PROMPT...", text_color="#f1c40f")
        else:
            self.status_indicator.configure(fg_color="gray")
            self.label.configure(text="HOLD [RIGHT CTRL] TO SPEAK", text_color="white")

    def process_audio(self):
        try:
            if not self.audio_data: return
            
            audio_stack = np.concatenate(self.audio_data) * 5.0
            write(FILENAME, FS, audio_stack)
            
            # A. Transcribe
            with open(FILENAME, "rb") as file:
                transcription = client.audio.transcriptions.create(
                    file=(FILENAME, file.read()),
                    model="whisper-large-v3",
                    response_format="text"
                )
            
            # B. Refine
            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": transcription}]
            )
            
            final_text = completion.choices[0].message.content.strip()
            
            # C. Inject & Update UI
            pyperclip.copy(final_text)
            time.sleep(0.3)
            pyautogui.hotkey('ctrl', 'v')
            
            self.raw_text_view.delete("0.0", "end")
            self.raw_text_view.insert("0.0", f"Cleaned: {final_text}")
            
            winsound.Beep(1000, 150)
            self.after(0, lambda: self.update_ui_state("idle"))

        except Exception as e:
            self.raw_text_view.insert("end", f"\nError: {e}")
            self.after(0, lambda: self.update_ui_state("idle"))

if __name__ == "__main__":
    app = PromptBufferUI()
    app.mainloop()
