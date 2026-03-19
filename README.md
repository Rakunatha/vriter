# Vrite

Generate style-matched talking-head videos from a script.
100% free, open-source. Run locally or host on Render.

---

## Project structure

    Vrite/
    |
    |-- Dockerfile              Render-ready container (no .streamlit needed)
    |-- render.yaml             Render auto-deploy config
    |-- requirements.txt        Cloud-optimised, CPU-only dependencies
    |-- run.py                  CLI + Python API entry point
    |-- setup.sh                Linux/macOS one-command local setup
    |-- setup.ps1               Windows PowerShell local setup
    |-- .gitignore
    |
    |-- vrite/                  Core Python package
    |   |-- config.py           PipelineConfig - all settings
    |   |-- utils.py            Shared helpers
    |   |-- pipeline/
    |       |-- style_analyser.py       Visual + audio style extraction
    |       |-- script_preprocessor.py NLP cleaning, pause injection
    |       |-- tts_engine.py           Coqui / gTTS / pyttsx3 fallback
    |       |-- audio_post_processor.py Loudness norm, EQ match
    |       |-- lipsync_engine.py       Wav2Lip / SadTalker / audio-swap
    |       |-- video_compositor.py     Colour grading + ffmpeg encode
    |       |-- video_enhancer.py       Optional upscale, sharpen, LUT
    |       |-- model_downloader.py     One-time model setup
    |
    |-- ui/
    |   |-- app.py              Streamlit web interface
    |
    |-- scripts/
    |   |-- check_environment.py    Pre-flight checker
    |   |-- batch_process.py        Multi-video batch runner
    |   |-- sample_jobs.json        Example batch job file
    |
    |-- tests/
    |   |-- test_suite.py       28 unit tests (no GPU needed)
    |
    |-- docs/
    |   |-- TROUBLESHOOTING.md
    |
    |-- models/                 Model weights (downloaded on first run)
    |-- uploads/                Put reference files here
    |-- outputs/                Generated videos saved here

---

## Deploy to Render (recommended)

    1. Push this repo to GitHub (files must be at root, not in a subfolder)
    2. dashboard.render.com -> New -> Web Service -> Connect repo
    3. Render detects the Dockerfile automatically
    4. Add env vars: VRITE_DEVICE=cpu and PYTHONUNBUFFERED=1
    5. Deploy - live in ~15 minutes

See DEPLOY.md for full step-by-step instructions.

---

## Run locally

Linux / macOS:

    bash setup.sh
    streamlit run ui/app.py

Windows (PowerShell - run once first):

    Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
    .\setup.ps1
    streamlit run ui\app.py

CLI:

    python run.py --sample-video uploads/ref.mp4 --script "Your script"
    python run.py --sample-video uploads/ref.mp4 --script script.txt --output out.mp4
    python run.py --help

---

## Configuration

All settings are in vrite/config.py.
Override any setting via environment variables:

    VRITE_DEVICE=cpu
    VRITE_OUTPUT_CRF=22
    VRITE_MAX_DURATION_SECONDS=60
    VRITE_TTS_SPEED=1.1

---

## Tech stack

    Voice synthesis  : Coqui XTTS-v2 (or gTTS on free tier)
    Lip-sync         : Wav2Lip GAN
    Avatar animation : SadTalker
    Video processing : ffmpeg + OpenCV
    Web UI           : Streamlit
    Hosting          : Render (Docker)
