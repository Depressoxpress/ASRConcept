"""
main.py - Streamlit Dashboard for Real-Time ASR Overlay
Launch:  streamlit run main.py --server.headless true
"""

import html
import os
import queue
import subprocess
import sys
import tempfile
import time

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.engine import AudioCapture, TranscriptionEngine

# -- Page config -----------------------------------------------------------
st.set_page_config(page_title="Real-Time ASR Overlay", page_icon="🎙", layout="wide")

# -- Shared temp files for overlay IPC -------------------------------------
TEMP_DIR = os.path.join(tempfile.gettempdir(), "asr_overlay")
os.makedirs(TEMP_DIR, exist_ok=True)
TRANSCRIPT_FILE = os.path.join(TEMP_DIR, "transcript.txt")
CONTROL_FILE = os.path.join(TEMP_DIR, "control.txt")

# -- Session state ---------------------------------------------------------
def _init_state():
    defaults = {
        "transcribing": False,
        "transcript": "",
        "capture": None,
        "engine": None,
        "text_queue": None,
        "overlay_proc": None,
        "model_size": "small.en",
        "buffer_seconds": 10,
        "overlay_opacity": 0.82,
        "overlay_width": 700,
        "overlay_height": 200,
        "overlay_font_size": 14,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# -- Overlay helpers -------------------------------------------------------
def _launch_overlay():
    if (
        st.session_state.overlay_proc is not None
        and st.session_state.overlay_proc.poll() is None
    ):
        return
    overlay_script = os.path.join(os.path.dirname(__file__), "src", "overlay.py")
    st.session_state.overlay_proc = subprocess.Popen(
        [
            sys.executable, overlay_script,
            TRANSCRIPT_FILE, CONTROL_FILE,
            str(st.session_state.overlay_opacity),
            str(st.session_state.overlay_width),
            str(st.session_state.overlay_height),
            str(st.session_state.overlay_font_size),
        ],
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )

def _kill_overlay():
    proc = st.session_state.overlay_proc
    if proc is not None and proc.poll() is None:
        try:
            with open(CONTROL_FILE, "w", encoding="utf-8") as f:
                f.write("EXIT")
            proc.wait(timeout=3)
        except Exception:
            proc.kill()
    st.session_state.overlay_proc = None

def _drain_queue():
    q = st.session_state.text_queue
    if q is None:
        return
    chunks = []
    while True:
        try:
            chunks.append(q.get_nowait())
        except queue.Empty:
            break
    if chunks:
        new_text = " ".join(chunks)
        if st.session_state.transcript:
            st.session_state.transcript += " " + new_text
        else:
            st.session_state.transcript = new_text
        try:
            with open(TRANSCRIPT_FILE, "a", encoding="utf-8") as f:
                f.write(" " + new_text)
        except Exception:
            pass

# -- Sidebar ---------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    st.session_state.model_size = st.selectbox(
        "Model",
        ["tiny.en", "base.en", "small.en", "medium.en"],
        index=2,
        help="small.en recommended. medium.en = best accuracy but slower.",
    )

    st.session_state.buffer_seconds = st.slider(
        "Buffer (seconds)", 5, 30,
        value=st.session_state.buffer_seconds,
        step=5,
        help="Audio buffer size. Higher = more safety margin.",
    )

    st.markdown("---")
    st.markdown("## 🖥️ Overlay")

    st.slider("Opacity", 0.3, 1.0, key="overlay_opacity",
              step=0.05, on_change=_kill_overlay)
    st.slider("Width (px)", 300, 1200, key="overlay_width",
              step=50, on_change=_kill_overlay)
    st.slider("Height (px)", 100, 800, key="overlay_height",
              step=20, on_change=_kill_overlay)
    st.slider("Font Size", 10, 40, key="overlay_font_size",
              step=1, on_change=_kill_overlay)

    st.markdown("---")
    st.caption("Real-Time ASR Overlay v3.1")

# -- CSS -------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700&display=swap');
.stApp { font-family: 'Outfit', sans-serif; background-color: #0e1117; }
.transcript-box {
    background: linear-gradient(145deg, #1e2530 0%, #151922 100%);
    color: #e0e6ed; border-radius: 12px; padding: 24px;
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 16px; line-height: 1.6;
    min-height: 300px; max-height: 600px; overflow-y: auto;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 4px 12px rgba(0,0,0,0.4);
    white-space: pre-wrap; word-wrap: break-word;
}
.transcript-box::-webkit-scrollbar { width: 8px; }
.transcript-box::-webkit-scrollbar-track { background: #151922; }
.transcript-box::-webkit-scrollbar-thumb { background: #3b4252; border-radius: 4px; }
.status-badge {
    display: inline-flex; align-items: center;
    padding: 6px 16px; border-radius: 20px;
    font-size: 14px; font-weight: 600; letter-spacing: 0.5px;
}
.status-on {
    background: rgba(46,204,113,0.15); color: #2ecc71;
    border: 1px solid rgba(46,204,113,0.3);
    box-shadow: 0 0 10px rgba(46,204,113,0.2);
}
.status-off {
    background: rgba(231,76,60,0.12); color: #e74c3c;
    border: 1px solid rgba(231,76,60,0.25);
}
</style>
""", unsafe_allow_html=True)

# -- Header ----------------------------------------------------------------
st.markdown("# 🎙 Real-Time ASR Overlay")

if st.session_state.transcribing:
    st.markdown(
        '<span class="status-badge status-on">● LISTENING</span>',
        unsafe_allow_html=True,
    )
    _launch_overlay()
else:
    st.markdown(
        '<span class="status-badge status-off">● IDLE</span>',
        unsafe_allow_html=True,
    )

st.markdown("")

# -- Buttons ---------------------------------------------------------------
col1, col2, col3 = st.columns([1, 1, 3])

with col1:
    if st.session_state.transcribing:
        if st.button("⏹ Stop", type="primary", use_container_width=True):
            if st.session_state.engine:
                st.session_state.engine.stop()
            if st.session_state.capture:
                st.session_state.capture.stop()
            _kill_overlay()
            st.session_state.transcribing = False
            st.rerun()
    else:
        if st.button("▶ Start", type="primary", use_container_width=True):
            try:
                text_q = queue.Queue()
                capture = AudioCapture(
                    buffer_seconds=st.session_state.buffer_seconds
                )
                engine = TranscriptionEngine(
                    audio_capture=capture,
                    text_queue=text_q,
                    model_size=st.session_state.model_size,
                    compute_type="int8",
                    chunk_seconds=3.0,
                )
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
            except Exception as e:
                st.error(f"Failed to start: {e}")

with col2:
    if st.button("🗑 Clear", use_container_width=True):
        st.session_state.transcript = ""
        if st.session_state.engine:
            st.session_state.engine.clear()
        if st.session_state.capture:
            st.session_state.capture.clear_buffer()
        try:
            with open(TRANSCRIPT_FILE, "w", encoding="utf-8") as f:
                f.write("")
            with open(CONTROL_FILE, "w", encoding="utf-8") as f:
                f.write("CLEAR")
        except Exception:
            pass
        st.rerun()

# -- Live transcript -------------------------------------------------------
st.markdown("### 📝 Live Transcript")
_drain_queue()
display_text = st.session_state.transcript.strip() or "Waiting for audio…"
st.markdown(
    f'<div class="transcript-box">{html.escape(display_text)}</div>',
    unsafe_allow_html=True,
)

if st.session_state.transcribing:
    time.sleep(0.8)
    st.rerun()
