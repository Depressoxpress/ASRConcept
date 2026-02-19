"""
main.py — Streamlit Dashboard for Real-Time ASR Overlay

Launch:  streamlit run main.py --server.headless true
"""

import streamlit as st
import subprocess
import sys
import os
import queue
import time
import tempfile
import threading

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine import AudioCapture, TranscriptionEngine

# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Real-Time ASR Overlay",
    page_icon="🎙",
    layout="wide",
)

# ── Shared temp files for overlay IPC ────────────────────────────────
TEMP_DIR = os.path.join(tempfile.gettempdir(), "asr_overlay")
os.makedirs(TEMP_DIR, exist_ok=True)
TRANSCRIPT_FILE = os.path.join(TEMP_DIR, "transcript.txt")
CONTROL_FILE = os.path.join(TEMP_DIR, "control.txt")


# ── Session-state initialization ─────────────────────────────────────
def _init_state():
    defaults = {
        "transcribing": False,
        "transcript": "",
        "capture": None,
        "engine": None,
        "text_queue": None,
        "overlay_proc": None,
        "model_size": "base.en",
        "buffer_seconds": 10,
        "overlay_opacity": 0.82,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ── Helper: launch overlay subprocess ────────────────────────────────
def _launch_overlay():
    if (
        st.session_state.overlay_proc is not None
        and st.session_state.overlay_proc.poll() is None
    ):
        return  # already running

    overlay_script = os.path.join(os.path.dirname(__file__), "overlay.py")
    opacity = str(st.session_state.overlay_opacity)
    st.session_state.overlay_proc = subprocess.Popen(
        [
            sys.executable,
            overlay_script,
            TRANSCRIPT_FILE,
            CONTROL_FILE,
            opacity,
        ],
        creationflags=subprocess.CREATE_NO_WINDOW
        if sys.platform == "win32"
        else 0,
    )


def _kill_overlay():
    proc = st.session_state.overlay_proc
    if proc is not None and proc.poll() is None:
        # Send EXIT command
        try:
            with open(CONTROL_FILE, "w", encoding="utf-8") as f:
                f.write("EXIT")
            proc.wait(timeout=3)
        except Exception:
            proc.kill()
    st.session_state.overlay_proc = None


# ── Helper: drain text queue and update transcript ───────────────────
def _drain_queue():
    q = st.session_state.text_queue
    if q is None:
        return
    new_chunks = []
    while True:
        try:
            new_chunks.append(q.get_nowait())
        except queue.Empty:
            break
    if new_chunks:
        new_text = " ".join(new_chunks)
        st.session_state.transcript += " " + new_text
        # Append to shared file for overlay
        try:
            with open(TRANSCRIPT_FILE, "a", encoding="utf-8") as f:
                f.write(" " + new_text)
        except Exception:
            pass


# ── Sidebar — Settings ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.session_state.model_size = st.selectbox(
        "Whisper Model",
        ["tiny.en", "base.en", "small.en"],
        index=1,
        help="Larger = more accurate but slower.",
    )
    st.session_state.buffer_seconds = st.slider(
        "Audio Buffer (seconds)",
        min_value=5,
        max_value=30,
        value=st.session_state.buffer_seconds,
        step=5,
        help="Sliding window duration for transcription context.",
    )
    st.session_state.overlay_opacity = st.slider(
        "Overlay Opacity",
        min_value=0.3,
        max_value=1.0,
        value=st.session_state.overlay_opacity,
        step=0.05,
    )

    st.markdown("---")
    st.caption("Real-Time ASR Overlay v1.0")
    st.caption("Powered by Faster-Whisper + WASAPI")


# ── Custom CSS ───────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    .transcript-box {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        color: #e0e0e0;
        border-radius: 12px;
        padding: 20px 24px;
        font-family: 'Consolas', 'Courier New', monospace;
        font-size: 15px;
        line-height: 1.8;
        min-height: 280px;
        max-height: 500px;
        overflow-y: auto;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 4px 24px rgba(0,0,0,0.3);
        white-space: pre-wrap;
        word-wrap: break-word;
    }

    .status-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .status-on {
        background: rgba(0,200,100,0.15);
        color: #00c864;
        border: 1px solid rgba(0,200,100,0.3);
    }
    .status-off {
        background: rgba(200,80,80,0.12);
        color: #cc5050;
        border: 1px solid rgba(200,80,80,0.25);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Header ───────────────────────────────────────────────────────────
st.markdown("# 🎙 Real-Time ASR Overlay")

if st.session_state.transcribing:
    st.markdown(
        '<span class="status-badge status-on">● LISTENING</span>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<span class="status-badge status-off">● IDLE</span>',
        unsafe_allow_html=True,
    )

st.markdown("")

# ── Action buttons ───────────────────────────────────────────────────
col1, col2, col3 = st.columns([1, 1, 3])

with col1:
    if st.session_state.transcribing:
        if st.button("⏹ Stop Transcription", type="primary", use_container_width=True):
            # Stop engine + capture
            if st.session_state.engine:
                st.session_state.engine.stop()
            if st.session_state.capture:
                st.session_state.capture.stop()
            _kill_overlay()
            st.session_state.transcribing = False
            st.rerun()
    else:
        if st.button(
            "▶ Start Transcription", type="primary", use_container_width=True
        ):
            # Initialize engine
            text_q = queue.Queue()
            capture = AudioCapture(
                buffer_seconds=st.session_state.buffer_seconds
            )
            engine = TranscriptionEngine(
                audio_capture=capture,
                text_queue=text_q,
                model_size=st.session_state.model_size,
                compute_type="int8",
            )

            # Reset transcript file
            with open(TRANSCRIPT_FILE, "w", encoding="utf-8") as f:
                f.write("")
            with open(CONTROL_FILE, "w", encoding="utf-8") as f:
                f.write("")

            capture.start()
            engine.start()

            st.session_state.capture = capture
            st.session_state.engine = engine
            st.session_state.text_queue = text_q
            st.session_state.transcribing = True

            _launch_overlay()
            st.rerun()

with col2:
    if st.button("🗑 Clear Transcript", use_container_width=True):
        st.session_state.transcript = ""
        if st.session_state.engine:
            st.session_state.engine.clear()
        if st.session_state.capture:
            st.session_state.capture.clear_buffer()
        # Signal overlay
        try:
            with open(TRANSCRIPT_FILE, "w", encoding="utf-8") as f:
                f.write("")
            with open(CONTROL_FILE, "w", encoding="utf-8") as f:
                f.write("CLEAR")
        except Exception:
            pass
        st.rerun()


# ── Live transcript display ──────────────────────────────────────────
st.markdown("### 📝 Live Transcript")

_drain_queue()

display_text = st.session_state.transcript.strip() or "Waiting for audio…"

st.markdown(
    f'<div class="transcript-box">{display_text}</div>',
    unsafe_allow_html=True,
)

# ── Auto-refresh while transcribing ──────────────────────────────────
if st.session_state.transcribing:
    time.sleep(1.5)
    st.rerun()
