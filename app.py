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

# --- 1. CONFIGURATION ---
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
FILENAME = "buffer_segment.wav"
FS = 48000 
DEVICE_ID = 1  # Standard Lenovo Mic Array Index

# --- 2. UI THEME ---
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class PromptBufferUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Prompt Buffer")
        self.geometry("500x350") # Slightly larger for the dual-text view
        self.attributes("-topmost", True)
        self.resizable(False, False)

        self.recording = False
        self.audio_data = []

        self.label = ctk.CTkLabel(self, text="HOLD [RIGHT CTRL] TO SPEAK", font=("Inter", 14, "bold"))
        self.label.pack(pady=(20, 5))

        self.status_indicator = ctk.CTkButton(self, text="", width=16, height=16, corner_radius=8, 
                                              fg_color="gray", state="disabled")
        self.status_indicator.pack(pady=5)

        # Display Box for Before/After
        self.raw_text_view = ctk.CTkTextbox(self, width=450, height=160, font=("Consolas", 12))
        self.raw_text_view.pack(pady=10)
        self.raw_text_view.insert("0.0", "System Ready for Demo...")

        self.footer = ctk.CTkLabel(self, text="Utility v2.5 | Demo Mode", font=("Inter", 10), text_color="gray")
        self.footer.pack(side="bottom", pady=5)

        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()

        try:
            self.stream = sd.InputStream(samplerate=FS, channels=1, device=DEVICE_ID, callback=self.record_callback)
            self.stream.start()
        except Exception as e:
            print(f"Mic Error: {e}")

    def record_callback(self, indata, frames, time, status):
        if self.recording:
            self.audio_data.append(indata.copy())
            vol = np.linalg.norm(indata) * 30
            print(f"Level: {'█' * int(vol)}".ljust(50), end="\r")

    def on_press(self, key):
        if key == keyboard.Key.ctrl_r and not self.recording:
            self.recording = True
            self.audio_data = []
            self.after(0, lambda: self.update_ui_state("recording"))
            winsound.Beep(500, 100)

    def on_release(self, key):
        if key == keyboard.Key.ctrl_r and self.recording:
            self.recording = False
            self.after(0, lambda: self.update_ui_state("processing"))
            threading.Thread(target=self.process_audio).target=self.process_audio().start() if hasattr(threading.Thread(target=self.process_audio), 'start') else threading.Thread(target=self.process_audio).start()

    # NOTE: Simplified thread call for clarity
    def on_release(self, key):
        if key == keyboard.Key.ctrl_r and self.recording:
            self.recording = False
            self.after(0, lambda: self.update_ui_state("processing"))
            threading.Thread(target=self.process_audio).start()

    def update_ui_state(self, state):
        if state == "recording":
            self.status_indicator.configure(fg_color="#ff4b4b")
            self.label.configure(text="🔴 LISTENING...", text_color="#ff4b4b")
        elif state == "processing":
            self.status_indicator.configure(fg_color="#f1c40f")
            self.label.configure(text="🟡 OPTIMIZING...", text_color="#f1c40f")
        else:
            self.status_indicator.configure(fg_color="gray")
            self.label.configure(text="HOLD [RIGHT CTRL] TO SPEAK", text_color="white")

    def process_audio(self):
        try:
            if not self.audio_data:
                self.after(0, lambda: self.update_ui_state("idle"))
                return
            
            # --- AUDIO BOOST ---
            audio_stack = np.concatenate(self.audio_data) * 20.0
            write(FILENAME, FS, audio_stack)
            
            # 1. TRANSCRIPTION
            with open(FILENAME, "rb") as file:
                transcription = client.audio.transcriptions.create(
                    file=(FILENAME, file.read()),
                    model="whisper-large-v3",
                    response_format="text"
                )
            
            raw_text = transcription.strip()

            # 2. LLAMA REFINEMENT
            prompt_logic = (
                "Task: Convert messy speech into a single technical command.\n"
                "Example Input: 'Create a... no, a React button... actually a Redux store.'\n"
                "Example Output: Create a Redux store.\n\n"
                f"Input: '{raw_text}'\n"
                "Output: "
            )

            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt_logic}],
                temperature=0.0,
                max_tokens=20,
                stop=["\n", "1.", "Cleaned:"]
            )
            
            final_text = completion.choices[0].message.content.strip()

            # 3. UI UPDATE (MESSY VS CLEAN)
            # This is the part that will impress Boris
            self.raw_text_view.delete("0.0", "end")
            display_content = f"SPOKEN:\n\"{raw_text}\"\n\nCLEANED:\n\"{final_text}\""
            self.raw_text_view.insert("0.0", display_content)

            # 4. OUTPUT TO SYSTEM
            pyperclip.copy(final_text)
            time.sleep(0.4)
            pyautogui.hotkey('ctrl', 'v')
            
            winsound.Beep(1000, 150)
            self.after(0, lambda: self.update_ui_state("idle"))

        except Exception as e:
            print(f"\nError: {e}")
            self.after(0, lambda: self.update_ui_state("idle"))

if __name__ == "__main__":
    app = PromptBufferUI()
    app.mainloop()
