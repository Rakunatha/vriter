# Troubleshooting Guide

---

## Render deployment errors

### "Dockerfile not found"

Cause: Your files are inside a subfolder (e.g. Vrite/) instead of at
the repo root.

Fix: Check github.com/YOUR_USERNAME/vrite - Dockerfile must be visible
on the repo homepage. If it is inside a folder, move everything up:

    cd the-folder-containing-Dockerfile
    git init
    git add .
    git commit -m "fix: files at root"
    git push -u origin main --force

### ".streamlit/config.toml not found"

Cause: The old Dockerfile had a COPY .streamlit/config.toml line.
This file was never in the repo (dotfolders are hidden on GitHub).

Fix: Delete the .streamlit folder and push again.
The current Dockerfile writes the config internally using RUN printf.
No file upload is needed.

    git rm -r --cached .streamlit
    git commit -m "Remove .streamlit"
    git push

### Build runs out of memory

Cause: Render free tier has 512 MB RAM. PyTorch is large.

Fix: Upgrade to Starter plan ($7/month) or wait - sometimes a retry works.

---

## Local setup errors

### ffmpeg not found

    sudo apt install ffmpeg        # Ubuntu/Debian
    brew install ffmpeg            # macOS
    ffmpeg -version                # verify

Windows: download from ffmpeg.org, extract, add bin\ to system PATH.

### pip install fails with UnicodeDecodeError

Cause: requirements.txt has non-ASCII characters (box-drawing symbols
in comments). This is fixed in the current version.

Fix: open requirements.txt and delete any comment lines that contain
symbols like ===, ----, or similar decorative characters.

### No space left on device

You need about 8 GB free. Use --no-cache-dir:

    pip install --no-cache-dir -r requirements.txt

### Virtual env activation fails on Windows

Run this once to allow scripts:

    Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned

Then activate using dot-space (both are required):

    . .\.venv\Scripts\Activate.ps1

---

## Runtime errors

### No face detected in source video

The reference video needs a clear, forward-facing close-up.
Minimum face size: 60x60 pixels.
Avoid: extreme side angles, sunglasses, heavy motion blur.

### TTS sounds robotic or generic

On Render free tier gTTS is used (standard Google voice).
For voice cloning, install Coqui TTS:

    pip install TTS

Coqui downloads ~1.8 GB on first run.

### Generated video has no lip-sync

Wav2Lip requires models/wav2lip_gan.pth (~430 MB).
On Render free tier with no persistent disk, this file is not saved.
The app falls back to audio-swap (original face movements, new voice).

To enable full lip-sync on Render:
    Render dashboard -> your service -> Disks -> Add Disk
    Mount path: /app/models  |  Size: 10 GB

### CUDA out of memory

Reduce batch sizes in vrite/config.py:

    wav2lip_face_det_batch = 8    (default 16)
    wav2lip_batch = 64            (default 128)

Or force CPU mode:

    python run.py --device cpu ...

### Output video has no audio

Check that TTS produced a valid file:

    python -c "import subprocess, json; r = subprocess.run(['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', 'path/to/tts_output.wav'], capture_output=True, text=True); print(r.stdout)"

If the file is 0 bytes, all TTS engines failed. Run with --verbose to see why.
