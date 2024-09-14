# Import necessary libraries
import librosa
import numpy as np
import whisper
from pyannote.audio import Pipeline
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import threading
import os
import soundfile as sf
import traceback
from tkinter import ttk
import queue
import logging
import ffmpeg  # Ensure this is the correct module

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Monkey-patch the load_audio function in whisper to use the specific FFmpeg path
def custom_load_audio(file, sr=16000):
    if not os.path.isfile(file):
        raise FileNotFoundError(f"Audio file not found: {file}")
    try:
        # Specify the full path to the FFmpeg executable
        ffmpeg_executable = r'C:\ffmpeg\ffmpeg.exe'  # Adjust this path if necessary

        out, _ = (
            ffmpeg
            .input(file, threads=0)
            .output('pipe:', format='s16le', acodec='pcm_s16le', ac=1, ar=sr)
            .overwrite_output()
            .run(cmd=ffmpeg_executable, capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e
    return np.frombuffer(out, np.int16).astype(np.float32) / 32768.0

# Apply the monkey-patch
whisper.audio.load_audio = custom_load_audio

# Updated function to align speaker segments with transcript segments
def align_segments(speaker_segments, transcript_segments):
    combined = []
    for t_seg in transcript_segments:
        t_start = t_seg['start']
        t_end = t_seg['end']
        t_text = t_seg['text']

        # Find all speaker segments that overlap with this transcript segment
        overlapping_speakers = []
        for s_seg in speaker_segments:
            s_start = s_seg['start']
            s_end = s_seg['end']
            s_speaker = s_seg['speaker']

            # Check for overlap
            if s_start < t_end and s_end > t_start:
                overlap_start = max(s_start, t_start)
                overlap_end = min(s_end, t_end)
                overlap_duration = overlap_end - overlap_start

                overlapping_speakers.append({
                    'speaker': s_speaker,
                    'overlap_duration': overlap_duration
                })

        # Determine the speaker with the maximum overlap
        if overlapping_speakers:
            primary_speaker = max(overlapping_speakers, key=lambda x: x['overlap_duration'])['speaker']
        else:
            primary_speaker = 'Unknown'

        combined.append({
            'start': t_start,
            'end': t_end,
            'speaker': primary_speaker,
            'text': t_text
        })

    return combined

# Main application class
class TranscriptionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Audio Transcription with Speaker Identification")

        # Variables
        self.audio_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.model_size = tk.StringVar(value='base')
        self.language = tk.StringVar(value='en')
        self.diarization_model = tk.StringVar(value='pyannote/speaker-diarization')

        # Queue for thread-safe communication
        self.queue = queue.Queue()

        # Create GUI elements
        self.create_widgets()

        # Start processing the queue
        self.root.after(100, self.process_queue)

    def create_widgets(self):
        # Audio File Selection
        tk.Label(self.root, text="Audio File:").grid(row=0, column=0, sticky='e', padx=5, pady=5)
        tk.Entry(self.root, textvariable=self.audio_path, width=50).grid(row=0, column=1, padx=5, pady=5)
        tk.Button(self.root, text="Browse", command=self.browse_audio_file).grid(row=0, column=2, padx=5, pady=5)

        # Output File Selection
        tk.Label(self.root, text="Output File:").grid(row=1, column=0, sticky='e', padx=5, pady=5)
        tk.Entry(self.root, textvariable=self.output_path, width=50).grid(row=1, column=1, padx=5, pady=5)
        tk.Button(self.root, text="Save As", command=self.save_output_file).grid(row=1, column=2, padx=5, pady=5)

        # Whisper Model Size
        tk.Label(self.root, text="Whisper Model Size:").grid(row=2, column=0, sticky='e', padx=5, pady=5)
        model_options = ['tiny', 'base', 'small', 'medium', 'large']
        tk.OptionMenu(self.root, self.model_size, *model_options).grid(row=2, column=1, sticky='w', padx=5, pady=5)

        # Language Selection
        tk.Label(self.root, text="Language:").grid(row=3, column=0, sticky='e', padx=5, pady=5)
        tk.Entry(self.root, textvariable=self.language).grid(row=3, column=1, sticky='w', padx=5, pady=5)

        # Diarization Model
        tk.Label(self.root, text="Diarization Model:").grid(row=4, column=0, sticky='e', padx=5, pady=5)
        tk.Entry(self.root, textvariable=self.diarization_model, width=50).grid(row=4, column=1, padx=5, pady=5)

        # Transcribe Button
        self.transcribe_button = tk.Button(self.root, text="Transcribe", command=self.start_transcription)
        self.transcribe_button.grid(row=5, column=0, columnspan=3, pady=10)

        # Progress Bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.root, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=6, column=0, columnspan=3, padx=10, pady=10, sticky='we')

        # Transcript Display
        self.transcript_box = scrolledtext.ScrolledText(self.root, width=80, height=20)
        self.transcript_box.grid(row=7, column=0, columnspan=3, padx=10, pady=10)

    def browse_audio_file(self):
        filepath = filedialog.askopenfilename(
            filetypes=[("Audio Files", "*.wav;*.mp3;*.flac;*.m4a"), ("All Files", "*.*")]
        )
        if filepath:
            self.audio_path.set(filepath)

    def save_output_file(self):
        filepath = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
        )
        if filepath:
            self.output_path.set(filepath)

    def start_transcription(self):
        if not self.audio_path.get():
            messagebox.showerror("Error", "Please select an audio file.")
            return

        # Disable the transcribe button to prevent multiple clicks
        self.transcribe_button.config(state='disabled')
        self.transcript_box.delete('1.0', tk.END)
        self.queue.put(('message', "Processing...\n"))
        self.progress_var.set(0)  # Reset progress bar

        # Run transcription in a separate thread to keep the GUI responsive
        threading.Thread(target=self.transcribe_audio).start()

    def transcribe_audio(self):
        try:
            total_steps = 6  # Increased total steps for more granular updates
            current_step = 0

            audio_path = self.audio_path.get()
            output_path = self.output_path.get()
            model_size = self.model_size.get()
            language = self.language.get()
            diarization_model_name = self.diarization_model.get()

            # Load and preprocess the audio
            self.queue.put(('message', "Loading and preprocessing audio..."))
            logging.info("Loading audio file.")
            audio, sr = librosa.load(audio_path, sr=16000)
            logging.info("Audio loaded and resampled to 16kHz.")
            current_step += 1
            progress = (current_step / total_steps) * 100
            self.queue.put(('progress', progress))

            # Save the audio in the required format
            processed_audio_path = 'processed_audio.wav'
            sf.write(processed_audio_path, audio, sr, subtype='PCM_16')
            logging.info(f"Processed audio saved to {processed_audio_path}.")

            # Check if the file exists
            if not os.path.exists(processed_audio_path):
                error_msg = "Failed to write processed audio file."
                self.queue.put(('message', error_msg))
                logging.error(error_msg)
                return

            # Speaker Diarization using PyAnnote.audio
            self.queue.put(('message', "Loading speaker diarization model..."))
            logging.info("Loading speaker diarization model.")
            diarization_pipeline = Pipeline.from_pretrained(
                diarization_model_name,
                use_auth_token="your_hugging_face_token_here"  # Replace with your Hugging Face token
            )
            logging.info("Speaker diarization model loaded.")
            current_step += 1
            progress = (current_step / total_steps) * 100
            self.queue.put(('progress', progress))

            self.queue.put(('message', "Performing speaker diarization..."))
            logging.info("Starting speaker diarization.")
            diarization = diarization_pipeline(processed_audio_path)
            logging.info("Speaker diarization completed.")
            current_step += 1
            progress = (current_step / total_steps) * 100
            self.queue.put(('progress', progress))

            # Process diarization results
            speaker_segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                speaker_segments.append({
                    'start': turn.start,
                    'end': turn.end,
                    'speaker': speaker
                })
            logging.info("Processed speaker segments.")

            # Speech Recognition using OpenAI's Whisper
            self.queue.put(('message', "Loading Whisper model..."))
            logging.info(f"Loading Whisper model '{model_size}'.")
            whisper_model = whisper.load_model(model_size)
            logging.info("Whisper model loaded.")
            current_step += 1
            progress = (current_step / total_steps) * 100
            self.queue.put(('progress', progress))

            # Transcribe the audio
            self.queue.put(('message', "Transcribing audio..."))
            logging.info("Starting transcription.")
            transcription_result = whisper_model.transcribe(
                processed_audio_path,
                language=language,
                beam_size=5,
                best_of=5,
                temperature=0.0,
                patience=1.0
            )
            logging.info("Transcription completed.")
            current_step += 1
            progress = (current_step / total_steps) * 100
            self.queue.put(('progress', progress))

            # Process transcription results
            transcript_segments = transcription_result['segments']

            # Align speaker and transcription segments
            self.queue.put(('message', "Aligning segments..."))
            logging.info("Aligning speaker and transcription segments.")
            combined_segments = align_segments(speaker_segments, transcript_segments)
            current_step += 1
            progress = (current_step / total_steps) * 100
            self.queue.put(('progress', progress))

            # Output the transcript with speaker labels
            self.queue.put(('message', "\nFinal Transcript:\n"))
            logging.info("Generating final transcript.")
            transcript_text = ""
            last_text = None  # To prevent duplicate text entries
            for segment in combined_segments:
                start_time = segment['start']
                end_time = segment['end']
                speaker = segment['speaker']
                text = segment['text']
                if text != last_text:
                    line = f"Speaker {speaker} [{start_time:.2f}-{end_time:.2f}]: {text}"
                    self.queue.put(('message', line))
                    transcript_text += line + '\n'
                last_text = text

            # Save the transcript to a file if output path is provided
            if output_path:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(transcript_text)
                success_msg = f"\nTranscript saved to '{output_path}'."
                self.queue.put(('message', success_msg))
                logging.info(success_msg)

        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            traceback_str = traceback.format_exc()
            self.queue.put(('message', error_message))
            self.queue.put(('message', traceback_str))
            self.queue.put(('error', str(e)))
            logging.error(error_message)
            logging.error(traceback_str)
        finally:
            # Re-enable the transcribe button
            self.queue.put(('done', None))
            # Clean up temporary files
            processed_audio_path = 'processed_audio.wav'
            if os.path.exists(processed_audio_path):
                os.remove(processed_audio_path)
                logging.info(f"Deleted temporary file {processed_audio_path}.")
            # Reset the progress bar
            self.queue.put(('progress', 0))

    def process_queue(self):
        try:
            while True:
                item = self.queue.get_nowait()
                if item[0] == 'message':
                    message = item[1]
                    self.update_transcript_box(message)
                elif item[0] == 'progress':
                    progress = item[1]
                    self.progress_var.set(progress)
                    self.progress_bar.update_idletasks()
                elif item[0] == 'error':
                    error_msg = item[1]
                    messagebox.showerror("Error", error_msg)
                elif item[0] == 'done':
                    # Re-enable the transcribe button
                    self.transcribe_button.config(state='normal')
                self.queue.task_done()
        except queue.Empty:
            pass
        self.root.after(100, self.process_queue)

    def update_transcript_box(self, message):
        self.transcript_box.insert(tk.END, message + '\n')
        self.transcript_box.see(tk.END)

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = TranscriptionApp(root)
    root.mainloop()
