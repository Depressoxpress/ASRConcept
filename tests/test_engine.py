"""
tests/test_engine.py — Unit tests for src.engine

Covers:
  - AudioCapture ring buffer (write / read / wrap / clear)
  - AudioCapture.get_new_audio() incremental reads
  - TranscriptionEngine lifecycle (start / stop / clear)
  - Hallucination filter validation
"""

import queue
import unittest
from unittest.mock import patch, MagicMock

import numpy as np

from src.engine import AudioCapture, TranscriptionEngine


# ======================================================================
# 1. Ring Buffer — AudioCapture (mocked hardware)
# ======================================================================

class TestRingBuffer(unittest.TestCase):
    """Test AudioCapture's ring buffer without actual audio hardware."""

    def _make_capture(self, buffer_seconds=2):
        cap = AudioCapture(buffer_seconds=buffer_seconds)
        cap._device_channels = 1
        cap._device_rate = AudioCapture.TARGET_RATE
        return cap

    def test_write_and_read_small(self):
        cap = self._make_capture(buffer_seconds=2)
        data = np.ones(1000, dtype=np.float32) * 0.5
        cap._audio_callback(data.tobytes(), len(data), {}, 0)
        buf = cap.get_buffer()
        total_samples = AudioCapture.TARGET_RATE * 2
        self.assertEqual(len(buf), total_samples)
        np.testing.assert_array_almost_equal(buf[-1000:], 0.5)

    def test_wrap_around(self):
        cap = self._make_capture(buffer_seconds=1)
        data = np.arange(20000, dtype=np.float32)
        cap._audio_callback(data.tobytes(), len(data), {}, 0)
        buf = cap.get_buffer()
        self.assertEqual(len(buf), 16000)
        np.testing.assert_array_almost_equal(buf, data[4000:20000])

    def test_clear_buffer(self):
        cap = self._make_capture(buffer_seconds=1)
        data = np.ones(8000, dtype=np.float32) * 0.5
        cap._audio_callback(data.tobytes(), len(data), {}, 0)
        cap.clear_buffer()
        buf = cap.get_buffer()
        np.testing.assert_array_almost_equal(buf, np.zeros_like(buf))
        self.assertEqual(cap._write_pos, 0)
        self.assertEqual(cap._read_pos, 0)

    def test_stereo_to_mono(self):
        cap = self._make_capture(buffer_seconds=2)
        cap._device_channels = 2
        stereo = np.zeros(2000, dtype=np.float32)
        stereo[0::2] = 1.0
        stereo[1::2] = 0.0
        cap._audio_callback(stereo.tobytes(), 1000, {}, 0)
        buf = cap.get_buffer()
        nonzero = buf[buf != 0.0]
        if len(nonzero) > 0:
            np.testing.assert_array_almost_equal(nonzero, 0.5)

    def test_is_running_flag(self):
        cap = self._make_capture()
        self.assertFalse(cap.is_running)
        cap._running.set()
        self.assertTrue(cap.is_running)
        cap._running.clear()
        self.assertFalse(cap.is_running)


# ======================================================================
# 2. Incremental read — get_new_audio()
# ======================================================================

class TestGetNewAudio(unittest.TestCase):
    """Test AudioCapture.get_new_audio() incremental read."""

    def _make_capture(self, buffer_seconds=2):
        cap = AudioCapture(buffer_seconds=buffer_seconds)
        cap._device_channels = 1
        cap._device_rate = AudioCapture.TARGET_RATE
        return cap

    def test_empty_before_write(self):
        """get_new_audio returns empty before any writes."""
        cap = self._make_capture()
        audio = cap.get_new_audio()
        self.assertEqual(len(audio), 0)

    def test_returns_written_data(self):
        """get_new_audio returns data written via callback."""
        cap = self._make_capture()
        data = np.ones(1000, dtype=np.float32) * 0.5
        cap._audio_callback(data.tobytes(), len(data), {}, 0)
        audio = cap.get_new_audio()
        self.assertEqual(len(audio), 1000)
        np.testing.assert_array_almost_equal(audio, 0.5)

    def test_second_call_empty(self):
        """Second call without new writes returns empty."""
        cap = self._make_capture()
        data = np.ones(1000, dtype=np.float32) * 0.5
        cap._audio_callback(data.tobytes(), len(data), {}, 0)
        cap.get_new_audio()  # consume
        audio = cap.get_new_audio()
        self.assertEqual(len(audio), 0)

    def test_incremental_reads(self):
        """Multiple writes with reads in between."""
        cap = self._make_capture()
        d1 = np.ones(500, dtype=np.float32) * 0.1
        d2 = np.ones(300, dtype=np.float32) * 0.2
        cap._audio_callback(d1.tobytes(), len(d1), {}, 0)
        a1 = cap.get_new_audio()
        self.assertEqual(len(a1), 500)
        cap._audio_callback(d2.tobytes(), len(d2), {}, 0)
        a2 = cap.get_new_audio()
        self.assertEqual(len(a2), 300)
        np.testing.assert_array_almost_equal(a2, 0.2)

    def test_wraparound_read(self):
        """get_new_audio handles wrap-around correctly."""
        cap = self._make_capture(buffer_seconds=1)  # 16000 samples
        # Write 15000 samples (near end of buffer)
        d1 = np.ones(15000, dtype=np.float32) * 0.3
        cap._audio_callback(d1.tobytes(), len(d1), {}, 0)
        cap.get_new_audio()  # consume
        # Write 2000 more (wraps around)
        d2 = np.ones(2000, dtype=np.float32) * 0.7
        cap._audio_callback(d2.tobytes(), len(d2), {}, 0)
        audio = cap.get_new_audio()
        self.assertEqual(len(audio), 2000)
        np.testing.assert_array_almost_equal(audio, 0.7)

    def test_clear_resets_read_pos(self):
        """clear_buffer resets read position."""
        cap = self._make_capture()
        data = np.ones(1000, dtype=np.float32)
        cap._audio_callback(data.tobytes(), len(data), {}, 0)
        cap.clear_buffer()
        audio = cap.get_new_audio()
        self.assertEqual(len(audio), 0)


# ======================================================================
# 3. TranscriptionEngine — lifecycle & state
# ======================================================================

class TestTranscriptionEngineState(unittest.TestCase):

    def _make_engine(self):
        cap = AudioCapture(buffer_seconds=2)
        cap._device_channels = 1
        cap._device_rate = AudioCapture.TARGET_RATE
        q = queue.Queue()
        engine = TranscriptionEngine(
            audio_capture=cap,
            text_queue=q,
            model_size="tiny.en",
            compute_type="int8",
            chunk_seconds=0.5,
        )
        return engine, q, cap

    def test_clear_drains_queue(self):
        engine, q, _ = self._make_engine()
        q.put("chunk1")
        q.put("chunk2")
        q.put("chunk3")
        engine.clear()
        self.assertTrue(q.empty())
        self.assertEqual(engine._last_text, "")

    def test_clear_resets_accumulator(self):
        engine, _, _ = self._make_engine()
        engine._audio_acc.append(np.ones(100, dtype=np.float32))
        engine._acc_samples = 100
        engine.clear()
        self.assertEqual(len(engine._audio_acc), 0)
        self.assertEqual(engine._acc_samples, 0)

    def test_clear_resets_overlap(self):
        """clear() should reset the overlap tail."""
        engine, _, _ = self._make_engine()
        engine._prev_tail = np.ones(1000, dtype=np.float32)
        engine.clear()
        self.assertEqual(len(engine._prev_tail), 0)

    def test_clear_on_empty_queue(self):
        engine, q, _ = self._make_engine()
        engine.clear()
        self.assertTrue(q.empty())

    def test_stop_without_start(self):
        engine, _, _ = self._make_engine()
        engine.stop()

    @patch("src.engine.WhisperModel")
    def test_start_stop_lifecycle(self, mock_whisper_cls):
        mock_model = MagicMock()
        mock_whisper_cls.return_value = mock_model
        mock_model.transcribe.return_value = (iter([]), None)
        engine, q, cap = self._make_engine()
        data = np.random.randn(16000).astype(np.float32) * 0.1
        cap._audio_callback(data.tobytes(), len(data), {}, 0)
        engine.start()
        self.assertIsNotNone(engine._thread)
        self.assertTrue(engine._thread.is_alive())
        engine.stop()
        self.assertFalse(
            engine._thread is not None and engine._thread.is_alive()
        )

    def test_double_start(self):
        engine, _, _ = self._make_engine()
        engine._inference_loop = lambda: None
        engine.start()
        thread1 = engine._thread
        thread1.join(timeout=2)
        engine.start()
        thread2 = engine._thread
        self.assertIsNotNone(thread2)


# ======================================================================
# 4. Hallucination Filter
# ======================================================================

class TestHallucinationFilter(unittest.TestCase):
    """Verify known hallucination phrases are in the filter set."""

    def test_common_hallucinations_filtered(self):
        phrases = [
            "You", "Thank you", "Silence", "Thanks",
            "Thanks for watching!", "Bye.", "The end.",
        ]
        for phrase in phrases:
            self.assertIn(
                phrase, TranscriptionEngine._HALLUCINATIONS,
                f"'{phrase}' should be in hallucination filter",
            )

    def test_empty_string_filtered(self):
        self.assertIn("", TranscriptionEngine._HALLUCINATIONS)


# ======================================================================
# 5. Edge-case regression tests
# ======================================================================

class TestRegressions(unittest.TestCase):

    def test_resampling_indices_no_out_of_bounds(self):
        cap = AudioCapture(buffer_seconds=1)
        cap._device_channels = 1
        cap._device_rate = 48000
        data = np.random.randn(1024).astype(np.float32) * 0.1
        cap._audio_callback(data.tobytes(), len(data), {}, 0)
        buf = cap.get_buffer()
        self.assertEqual(len(buf), 16000)

    def test_ring_buffer_exact_fill(self):
        cap = AudioCapture(buffer_seconds=1)
        cap._device_channels = 1
        cap._device_rate = AudioCapture.TARGET_RATE
        data = np.ones(16000, dtype=np.float32)
        cap._audio_callback(data.tobytes(), len(data), {}, 0)
        buf = cap.get_buffer()
        np.testing.assert_array_almost_equal(buf, np.ones(16000))


if __name__ == "__main__":
    unittest.main()
