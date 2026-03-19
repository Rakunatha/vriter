"""
Vrite - Streamlit Web UI
Cloud-aware: uses gTTS on Render free tier,
full Coqui XTTS-v2 when available.
"""
from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

st.set_page_config(
    page_title="Vrite",
    page_icon="V",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@700;800&family=Figtree:wght@400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'Figtree', sans-serif; }
.block-container { padding-top: 2rem; }
.vrite-header {
    background: linear-gradient(135deg, #0f0f0f 0%, #1a1a2e 100%);
    border: 1px solid #2d2d4e;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
}
.vrite-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    color: #f1f0fb;
    margin: 0 0 0.3rem 0;
}
.vrite-subtitle { font-size: 1rem; color: #8b8da0; margin: 0; }
.vrite-badge {
    display: inline-block;
    background: rgba(99,102,241,0.2);
    border: 1px solid rgba(99,102,241,0.4);
    color: #a5b4fc;
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    padding: 2px 10px;
    border-radius: 20px;
    margin-left: 12px;
}
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.75rem 1.5rem !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
}
.stButton > button:hover { opacity: 0.88 !important; }
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #6366f1, #8b5cf6) !important;
}
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


def _has(pkg: str) -> bool:
    try:
        __import__(pkg)
        return True
    except ImportError:
        return False


HAS_COQUI   = _has("TTS")
HAS_WAV2LIP = (Path("Wav2Lip/inference.py").exists()
               and Path("models/wav2lip_gan.pth").exists())
VOICE_ENGINE  = ("Coqui XTTS-v2" if HAS_COQUI else "gTTS")
LIPSYNC_ENGINE = ("Wav2Lip GAN" if HAS_WAV2LIP else "Audio-swap")

st.markdown("""
<div class="vrite-header">
  <p class="vrite-title">
    Vrite
    <span class="vrite-badge">cloud</span>
  </p>
  <p class="vrite-subtitle">
    Generate style-matched talking-head videos from a script.
  </p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Settings")
    tts_speed    = st.slider("Speech speed", 0.7, 1.4, 1.0, 0.05)
    output_crf   = st.slider("Video quality (CRF)", 15, 28, 20)
    max_dur      = st.slider("Max duration (s)", 15, 120, 60)
    colour_grade = st.toggle("Colour grading", value=True)
    fade_io      = st.toggle("Fade in / out", value=True)
    st.divider()
    st.markdown(f"**Voice:** {VOICE_ENGINE}")
    st.markdown(f"**Lip-sync:** {LIPSYNC_ENGINE}")
    st.divider()
    st.caption(
        "Files are temporary and available for download "
        "during your session only.")

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown("##### Step 1 - Upload reference video")
    sample_video = st.file_uploader(
        "Sample video (MP4 / MOV)",
        type=["mp4", "mov", "avi"],
        help="Provides the face and visual style.")
    sample_audio = st.file_uploader(
        "Voice reference - optional (WAV / MP3)",
        type=["wav", "mp3", "m4a"])
    if sample_video:
        st.video(sample_video)

    st.divider()
    st.markdown("##### Step 2 - Write your script")
    script_text = st.text_area(
        "Script",
        placeholder="Type or paste your script here...",
        height=220,
        label_visibility="collapsed")
    st.caption(f"{len(script_text):,} characters")

with col_right:
    st.markdown("##### Step 3 - Generate")
    ready = sample_video is not None and bool(script_text.strip())

    if not sample_video:
        st.info("Upload a sample video to get started.")
    elif not script_text.strip():
        st.info("Enter your script on the left.")

    generate_btn = st.button(
        "Generate video",
        disabled=not ready,
        use_container_width=True)

    if generate_btn and ready:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            video_path = str(tmp / sample_video.name)
            with open(video_path, "wb") as f:
                f.write(sample_video.read())

            audio_path = None
            if sample_audio:
                audio_path = str(tmp / sample_audio.name)
                with open(audio_path, "wb") as f:
                    f.write(sample_audio.read())

            output_path = str(tmp / "result.mp4")

            from vrite.config import PipelineConfig
            cfg = PipelineConfig(
                device="cpu",
                tts_speed=tts_speed,
                output_crf=output_crf,
                max_duration_seconds=max_dur,
                colour_grade=colour_grade,
                add_intro_fade=fade_io,
                add_outro_fade=fade_io,
                tmp_dir=str(tmp / "tmp"))

            progress_bar = st.progress(0)
            status_box = st.empty()

            def on_progress(msg: str, pct: int) -> None:
                progress_bar.progress(pct)
                status_box.info(f"**{msg}**")

            from run import VideoPipeline
            pipe = VideoPipeline(cfg)
            pipe.set_progress_callback(on_progress)

            t_start = time.perf_counter()
            try:
                result = pipe.run(
                    sample_video=video_path,
                    script=script_text,
                    sample_audio=audio_path,
                    output_path=output_path)
                elapsed = time.perf_counter() - t_start
                progress_bar.progress(100)
                status_box.success("Done! Your video is ready.")

                from vrite.utils import format_duration
                c1, c2, c3 = st.columns(3)
                c1.metric("Length",
                          f"{result.duration_seconds:.0f}s")
                c2.metric("Processed in",
                          format_duration(elapsed))
                c3.metric("Device", "CPU")

                st.markdown("#### Preview")
                video_bytes = Path(result.output_path).read_bytes()
                st.video(video_bytes)
                st.download_button(
                    label="Download MP4",
                    data=video_bytes,
                    file_name=f"vrite_{int(time.time())}.mp4",
                    mime="video/mp4",
                    use_container_width=True)

            except Exception as exc:
                status_box.error(f"Error: {exc}")
                with st.expander("Full error details"):
                    import traceback
                    st.code(traceback.format_exc())

st.divider()
cols = st.columns(4)
for col, (label, tool) in zip(cols, [
    ("Voice", VOICE_ENGINE),
    ("Lip-sync", LIPSYNC_ENGINE),
    ("Video", "ffmpeg + OpenCV"),
    ("UI", "Streamlit"),
]):
    col.markdown(
        f"<div style='text-align:center;font-size:0.8rem;"
        f"color:#555'>{label}<br><b>{tool}</b></div>",
        unsafe_allow_html=True)
