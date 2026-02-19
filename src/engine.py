"""
engine.py - WASAPI Loopback Audio Capture + Faster-Whisper Inference

Architecture: CHUNKED with OVERLAP CONTEXT (English-only)
  - AudioCapture: ring buffer with incremental reads via get_new_audio()
  - TranscriptionEngine: 2s chunks with 0.5s overlap from previous chunk
  - Uses word_timestamps to filter overlap words (no duplication, no lost words)
"""

import os
import sys
import warnings

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*huggingface.*")
warnings.filterwarnings("ignore", message=".*token.*")

import queue
import threading
import time
import logging

import numpy as np
import pyaudiowpatch as pyaudio
from faster_whisper import WhisperModel

__all__ = ["AudioCapture", "TranscriptionEngine"]

_LOG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "engine_debug.log"
)
logging.basicConfig(
    filename=os.path.normpath(_LOG_PATH),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    filemode="w",
)


# ---------------------------------------------------------------------------
# Audio Capture
# ---------------------------------------------------------------------------
class AudioCapture:
    """Captures system audio (WASAPI loopback) -> mono 16 kHz ring buffer."""

    TARGET_RATE = 16_000

    def __init__(self, buffer_seconds: int = 10):
        self._buffer_seconds = buffer_seconds
        self._buffer = np.zeros(self.TARGET_RATE * buffer_seconds, dtype=np.float32)
        self._write_pos = 0
        self._read_pos = 0
        self._lock = threading.Lock()
        self._running = threading.Event()
        self._pa = None
        self._stream = None
        self._device_channels: int = 1
        self._device_rate: int = self.TARGET_RATE

    def _init_pyaudio(self):
        self._pa = pyaudio.PyAudio()
        try:
            wasapi_info = self._pa.get_host_api_info_by_type(pyaudio.paWASAPI)
            default_speakers = self._pa.get_device_info_by_index(
                wasapi_info["defaultOutputDevice"]
            )
            for i in range(self._pa.get_device_count()):
                dev = self._pa.get_device_info_by_index(i)
                if dev.get("isLoopbackDevice", False):
                    if (
                        "loopback" in dev["name"].lower()
                        or dev["index"] == default_speakers["index"]
                        or default_speakers["name"] in dev["name"]
                    ):
                        return dev
            for i in range(self._pa.get_device_count()):
                dev = self._pa.get_device_info_by_index(i)
                if dev.get("isLoopbackDevice", False):
                    return dev
            raise RuntimeError("No WASAPI loopback device found.")
        except Exception:
            if self._pa:
                self._pa.terminate()
                self._pa = None
            raise

    def _audio_callback(self, in_data, frame_count, time_info, status):
        audio = np.frombuffer(in_data, dtype=np.float32)
        if self._device_channels > 1:
            audio = audio.reshape(-1, self._device_channels).mean(axis=1)
        if self._device_rate != self.TARGET_RATE:
            ratio = self._device_rate / self.TARGET_RATE
            indices = np.linspace(
                0, len(audio) - 1, int(len(audio) / ratio)
            ).astype(int)
            audio = audio[indices]
        with self._lock:
            n = len(audio)
            buf_len = len(self._buffer)
            end = self._write_pos + n
            if end <= buf_len:
                self._buffer[self._write_pos:end] = audio
            else:
                first = buf_len - self._write_pos
                self._buffer[self._write_pos:] = audio[:first]
                self._buffer[:n - first] = audio[first:]
            self._write_pos = end % buf_len
        return (None, pyaudio.paContinue)

    def get_new_audio(self) -> np.ndarray:
        """Return ONLY audio captured since the last call."""
        with self._lock:
            wp = self._write_pos
            rp = self._read_pos
            if wp == rp:
                return np.array([], dtype=np.float32)
            if wp > rp:
                audio = self._buffer[rp:wp].copy()
            else:
                audio = np.concatenate([
                    self._buffer[rp:],
                    self._buffer[:wp],
                ])
            self._read_pos = wp
            return audio

    def get_buffer(self) -> np.ndarray:
        with self._lock:
            return np.roll(self._buffer, -self._write_pos).copy()

    def start(self):
        if self._running.is_set():
            return
        device_info = self._init_pyaudio()
        self._device_channels = int(device_info["maxInputChannels"])
        self._device_rate = int(device_info["defaultSampleRate"])
        self._running.set()
        try:
            self._stream = self._pa.open(
                format=pyaudio.paFloat32,
                channels=self._device_channels,
                rate=self._device_rate,
                input=True,
                input_device_index=int(device_info["index"]),
                frames_per_buffer=pyaudio.paFramesPerBufferUnspecified,
                stream_callback=self._audio_callback,
            )
            self._stream.start_stream()
        except Exception as e:
            self.stop()
            raise RuntimeError(
                f"Audio stream failed: {device_info['name']}. {e}"
            ) from e

    def stop(self):
        self._running.clear()
        if self._stream is not None:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        if self._pa is not None:
            try:
                self._pa.terminate()
            except Exception:
                pass
            self._pa = None

    def clear_buffer(self):
        with self._lock:
            self._buffer[:] = 0.0
            self._write_pos = 0
            self._read_pos = 0

    @property
    def is_running(self) -> bool:
        return self._running.is_set()

    def terminate(self):
        self.stop()


# ---------------------------------------------------------------------------
# Transcription Engine (Chunked with Overlap, English-only)
# ---------------------------------------------------------------------------
class TranscriptionEngine:
    """
    Accumulates 2s of new audio, prepends 0.5s overlap from previous chunk
    as context, transcribes 2.5s total, and uses word_timestamps to emit
    only words from the NEW portion. This prevents both duplication and
    lost words at chunk boundaries.
    """

    OVERLAP_SECONDS = 0.5

    _HALLUCINATIONS = frozenset({
        "", "you", "You", "Thank you", "Thanks",
        "Silence", "silence", "...", "Thank you.",
        "Thanks for watching!", "Thanks for watching.",
        "Bye.", "Bye!", "The end.", "Bye",
        "Please subscribe.", "Subscribe.",
        "Thank you for watching.", "Thank you for watching!",
        "I'm sorry.", "I'm sorry",
    })

    def __init__(
        self,
        audio_capture: AudioCapture,
        text_queue: queue.Queue,
        model_size: str = "small.en",
        compute_type: str = "int8",
        chunk_seconds: float = 2.0,
    ):
        self._capture = audio_capture
        self._queue = text_queue
        self._model_size = model_size
        self._compute_type = compute_type
        self._chunk_samples = int(chunk_seconds * AudioCapture.TARGET_RATE)
        self._overlap_samples = int(
            self.OVERLAP_SECONDS * AudioCapture.TARGET_RATE
        )

        self._model: WhisperModel | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        self._audio_acc: list[np.ndarray] = []
        self._acc_samples: int = 0
        self._last_text: str = ""
        self._prev_tail: np.ndarray = np.zeros(0, dtype=np.float32)

    def _load_model(self):
        if self._model is None:
            self._model = WhisperModel(
                self._model_size,
                device="cpu",
                compute_type=self._compute_type,
            )

    def _inference_loop(self):
        """Accumulate 2s new audio + 0.5s overlap -> transcribe -> emit."""
        self._load_model()
        while not self._stop_event.is_set():
            new_audio = self._capture.get_new_audio()
            if len(new_audio) > 0:
                self._audio_acc.append(new_audio)
                self._acc_samples += len(new_audio)

            if self._acc_samples >= self._chunk_samples:
                new_chunk = np.concatenate(self._audio_acc)
                self._audio_acc.clear()
                self._acc_samples = 0

                # Build context: overlap from previous chunk + new audio
                if len(self._prev_tail) > 0:
                    audio_input = np.concatenate([
                        self._prev_tail, new_chunk
                    ])
                    overlap_dur = (
                        len(self._prev_tail) / AudioCapture.TARGET_RATE
                    )
                else:
                    audio_input = new_chunk
                    overlap_dur = 0.0

                # Save tail of new chunk for next iteration's overlap
                tail_len = min(self._overlap_samples, len(new_chunk))
                self._prev_tail = new_chunk[-tail_len:].copy()

                # Skip silence (check new audio only, not overlap)
                rms = np.sqrt(np.mean(new_chunk ** 2))
                if rms < 3e-4:
                    time.sleep(0.05)
                    continue

                try:
                    segments, _ = self._model.transcribe(
                        audio_input,
                        beam_size=1,
                        language="en",
                        temperature=0.0,
                        condition_on_previous_text=False,
                        repetition_penalty=1.3,
                        no_speech_threshold=0.6,
                        vad_filter=True,
                        vad_parameters=dict(min_silence_duration_ms=500),
                        word_timestamps=True,
                    )

                    # Only keep words from the NEW portion
                    # (after overlap region, with small tolerance)
                    threshold = max(0.0, overlap_dur - 0.15)
                    words = []
                    for seg in segments:
                        if seg.words:
                            for w in seg.words:
                                if w.start >= threshold:
                                    words.append(w.word)
                    text = "".join(words).strip()

                except Exception as e:
                    logging.error("Transcription error: %s", e)
                    text = ""

                # Filter hallucinations and exact repeats
                if (
                    text
                    and text not in self._HALLUCINATIONS
                    and text != self._last_text
                ):
                    logging.info("Chunk: '%s'", text)
                    self._queue.put(text)
                    self._last_text = text
            else:
                time.sleep(0.05)

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._last_text = ""
        self._prev_tail = np.zeros(0, dtype=np.float32)
        self._audio_acc.clear()
        self._acc_samples = 0
        self._thread = threading.Thread(
            target=self._inference_loop, daemon=True
        )
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def clear(self):
        """Reset all state and drain queue."""
        self._last_text = ""
        self._prev_tail = np.zeros(0, dtype=np.float32)
        self._audio_acc.clear()
        self._acc_samples = 0
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
