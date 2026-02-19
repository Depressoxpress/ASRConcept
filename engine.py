"""
engine.py — WASAPI Loopback Audio Capture + Faster-Whisper Inference Engine

Provides AudioCapture (system audio via WASAPI loopback) and TranscriptionEngine
(real-time speech-to-text with sliding-window buffer).
"""

import threading
import queue
import time
import numpy as np

import pyaudiowpatch as pyaudio
from faster_whisper import WhisperModel


# ---------------------------------------------------------------------------
# Audio Capture — WASAPI Loopback
# ---------------------------------------------------------------------------

class AudioCapture:
    """
    Captures system audio (WASAPI loopback) using PyAudioWPatch.
    Feeds mono 16 kHz float32 PCM into a thread-safe ring buffer.
    """

    TARGET_RATE = 16_000  # Whisper expects 16 kHz

    def __init__(self, buffer_seconds: int = 10):
        self._buffer_seconds = buffer_seconds
        self._buffer = np.zeros(self.TARGET_RATE * buffer_seconds, dtype=np.float32)
        self._write_pos = 0
        self._lock = threading.Lock()
        self._running = threading.Event()
        
        self._pa = None
        self._stream = None
        
    def _init_pyaudio(self):
        """Initialize PyAudio and find loopback device."""
        self._pa = pyaudio.PyAudio()
        
        try:
            # 1. Get default WASAPI info
            wasapi_info = self._pa.get_host_api_info_by_type(pyaudio.paWASAPI)
            default_speakers = self._pa.get_device_info_by_index(
                wasapi_info["defaultOutputDevice"]
            )
            
            # 2. Try to find loopback specifically for default speakers
            for i in range(self._pa.get_device_count()):
                dev = self._pa.get_device_info_by_index(i)
                if dev.get("isLoopbackDevice", False):
                    # Robust check: if it looks like the default output
                    if "loopback" in dev["name"].lower() or \
                       dev["index"] == default_speakers["index"] or \
                       default_speakers["name"] in dev["name"]:
                        return dev
            
            # 3. Fallback: First available loopback
            for i in range(self._pa.get_device_count()):
                if self._pa.get_device_info_by_index(i).get("isLoopbackDevice", False):
                    return self._pa.get_device_info_by_index(i)
                    
            raise RuntimeError("No WASAPI loopback device found.")
            
        except Exception as e:
            if self._pa:
                self._pa.terminate()
                self._pa = None
            raise e

    # ------------------------------------------------------------------
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Stream callback — downsample + push into ring buffer."""
        audio = np.frombuffer(in_data, dtype=np.float32)

        # Stereo→mono
        if self._device_channels > 1:
            audio = audio.reshape(-1, self._device_channels).mean(axis=1)

        # Resample to 16 kHz (simple decimation when ratio is integer)
        if self._device_rate != self.TARGET_RATE:
            ratio = self._device_rate / self.TARGET_RATE
            indices = np.round(np.arange(0, len(audio), ratio)).astype(int)
            indices = indices[indices < len(audio)]
            audio = audio[indices]

        # Write into ring buffer
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

    # ------------------------------------------------------------------
    def start(self):
        """Open the WASAPI loopback stream and begin capturing."""
        if self._running.is_set():
            return
            
        # Re-initialize PyAudio each time to avoid "Unanticipated host error"
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
            self.stop() # Cleanup
            raise RuntimeError(f"Failed to start audio stream. Device: {device_info['name']}. Error: {e}")

    def stop(self):
        """Stop capturing audio and release resources."""
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

    def get_buffer(self) -> np.ndarray:
        """Return a copy of the current ring buffer (oldest→newest)."""
        with self._lock:
            return np.roll(self._buffer, -self._write_pos).copy()

    def clear_buffer(self):
        """Zero out the ring buffer."""
        with self._lock:
            self._buffer[:] = 0.0
            self._write_pos = 0

    @property
    def is_running(self) -> bool:
        return self._running.is_set()

    def terminate(self):
        """Release PortAudio resources."""
        self.stop()


# ---------------------------------------------------------------------------
# Transcription Engine — Faster-Whisper
# ---------------------------------------------------------------------------

class TranscriptionEngine:
    """
    Runs Faster-Whisper inference on the AudioCapture ring buffer
    in a background thread.  Pushes new transcript text into a queue.
    """

    def __init__(
        self,
        audio_capture: AudioCapture,
        text_queue: queue.Queue,
        model_size: str = "base.en",
        compute_type: str = "int8",
        inference_interval: float = 2.0,
    ):
        self._capture = audio_capture
        self._queue = text_queue
        self._model_size = model_size
        self._compute_type = compute_type
        self._interval = inference_interval

        self._model = None
        self._thread = None
        self._stop_event = threading.Event()

        self._prev_text = ""  # for suffix-diff deduplication

    # ------------------------------------------------------------------
    def _load_model(self):
        if self._model is None:
            self._model = WhisperModel(
                self._model_size,
                device="cpu",
                compute_type=self._compute_type,
            )

    # ------------------------------------------------------------------
    def _inference_loop(self):
        """Background loop: grab buffer → transcribe → diff → enqueue."""
        self._load_model()

        while not self._stop_event.is_set():
            audio = self._capture.get_buffer()

            # Skip silence (RMS below threshold)
            rms = np.sqrt(np.mean(audio ** 2))
            if rms < 1e-4:
                time.sleep(self._interval)
                continue

            try:
                segments, _ = self._model.transcribe(
                    audio,
                    beam_size=1,
                    language="en",
                    vad_filter=True,
                    vad_parameters=dict(
                        min_silence_duration_ms=500,
                    ),
                )
                full_text = " ".join(seg.text.strip() for seg in segments).strip()
            except Exception:
                full_text = ""

            # Suffix-diff: only emit text that is new
            if full_text and full_text != self._prev_text:
                new_text = self._get_new_suffix(self._prev_text, full_text)
                if new_text:
                    self._queue.put(new_text)
                self._prev_text = full_text

            time.sleep(self._interval)

    # ------------------------------------------------------------------
    @staticmethod
    def _get_new_suffix(old: str, new: str) -> str:
        """Return the portion of *new* that extends beyond *old*."""
        if not old:
            return new
        # Find the longest overlap between the end of old and start of new
        best = 0
        for i in range(1, min(len(old), len(new)) + 1):
            if old[-i:] == new[:i]:
                best = i
        return new[best:].strip() if best else new

    # ------------------------------------------------------------------
    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._prev_text = ""
        self._thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def clear(self):
        """Reset transcript state."""
        self._prev_text = ""
        # Drain the queue
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
