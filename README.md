
# Audio Transcription with Speaker Identification

This project provides a Python-based GUI application for transcribing audio files with speaker identification, utilizing libraries such as Whisper, PyAnnote, and Librosa. The application allows users to select an audio file, process it for transcription, and identify speakers automatically.

## Features

- **Audio Transcription**: Uses OpenAI's Whisper model to transcribe audio files in various formats.
- **Speaker Diarization**: Identifies different speakers in the audio using the PyAnnote library.
- **Progress Tracking**: A visual progress bar displays the current stage of processing.
- **GUI Interface**: An easy-to-use graphical user interface built with Tkinter.

## Requirements

Before running the application, make sure you have the following dependencies installed:

### Python Libraries
- librosa
- numpy
- whisper
- pyannote.audio
- tkinter
- soundfile
- queue
- logging
- ffmpeg-python
- threading

### External Tools
- **FFmpeg**: Make sure FFmpeg is installed and configured in your system's environment. You may need to adjust the path to the FFmpeg executable in the script.

## Installation

### Windows

1. **Clone the repository**:
   Download or clone this repository using:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   ```

2. **Install Python and dependencies**:
   - Make sure you have Python 3.8+ installed. You can download Python [here](https://www.python.org/downloads/).
   - Open a command prompt and install the required dependencies:
     ```bash
     pip install librosa numpy whisper pyannote.audio soundfile ffmpeg-python
     ```

3. **Install FFmpeg**:
   - Download [FFmpeg for Windows](https://ffmpeg.org/download.html).
   - Extract the FFmpeg files to `C:\ffmpeg` (or any directory of your choice).
   - Add the FFmpeg `bin` folder to your PATH:
     - Right-click on "This PC" > "Properties" > "Advanced system settings" > "Environment Variables".
     - Under "System Variables", find the `Path` variable, click "Edit", and add the path to `C:\ffmpeg\bin`.

4. **Run the application**:
   ```bash
   python transcription_app.py
   ```

### Linux

1. **Clone the repository**:
   Open a terminal and run:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   ```

2. **Install Python and dependencies**:
   - Install Python 3.8+ and pip:
     ```bash
     sudo apt-get install python3 python3-pip
     ```
   - Install the required Python libraries:
     ```bash
     pip3 install librosa numpy whisper pyannote.audio soundfile ffmpeg-python
     ```

3. **Install FFmpeg**:
   - On Linux, FFmpeg can be installed through package managers. Run the following:
     ```bash
     sudo apt-get install ffmpeg
     ```

4. **Run the application**:
   ```bash
   python3 transcription_app.py
   ```

## Usage

1. Launch the application by running the appropriate command for your operating system:
   - On **Windows**:
     ```bash
     python transcription_app.py
     ```
   - On **Linux**:
     ```bash
     python3 transcription_app.py
     ```

2. Use the graphical interface to:
   - Select an audio file (supports `.wav`, `.mp3`, `.flac`, `.m4a`, and more).
   - Specify the output file for saving the transcription.
   - Choose the model size and language for Whisper-based transcription.
   - Use the diarization model to identify speakers in the audio.

## Packaging the Application for Windows and Linux

To make the application executable on both Windows and Linux without requiring Python, you can package it using **PyInstaller**.

### Packaging for Windows

1. Install **PyInstaller**:
   ```bash
   pip install pyinstaller
   ```

2. Build the executable:
   ```bash
   pyinstaller --onefile --windowed transcription_app.py
   ```

   This will create a single `.exe` file in the `dist` folder that you can run without needing Python installed.

### Packaging for Linux

1. Install **PyInstaller**:
   ```bash
   pip3 install pyinstaller
   ```

2. Build the executable:
   ```bash
   pyinstaller --onefile --windowed transcription_app.py
   ```

   This will create a standalone binary in the `dist` folder that can be run directly on Linux.

## Customization

### FFmpeg Path
If FFmpeg is not in your system's PATH, you can customize the script to point to your FFmpeg installation by modifying this line:

```python
ffmpeg_executable = r'C:\ffmpeg\ffmpeg.exe'  # Adjust this path if necessary
```

On Linux, this can be left as is if FFmpeg is installed via the package manager.

### Hugging Face Token
You need to replace the placeholder `"your_hugging_face_token_here"` with your actual Hugging Face API token for using the PyAnnote speaker diarization model.

```python
diarization_pipeline = Pipeline.from_pretrained(
    diarization_model_name,
    use_auth_token="your_hugging_face_token_here"
)
```

## Contributing

If you'd like to contribute, feel free to submit a pull request or open an issue for any bugs or features you'd like to see.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
