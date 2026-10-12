"""faster-whisper adapter: VAD speech coverage and hallucination-guarded word transcripts."""
from __future__ import annotations

import contextlib
import gc
import logging
import os
import re

import numpy as np

from ..paths import CACHE

SR = 16000
log = logging.getLogger("quillcut.asr")
STOCK_PHRASES = re.compile(r"^(thanks? (you )?for watching|please subscribe|subscribe|like and subscribe)[.!]*$", re.I)


def speech_segments(audio: np.ndarray) -> list[tuple[float, float]]:
    """Silero VAD bundled with faster-whisper (no model download)."""
    if audio.size < SR:
        return []
    from faster_whisper.vad import VadOptions, get_speech_timestamps
    stamps = get_speech_timestamps(audio, VadOptions(min_silence_duration_ms=500), sampling_rate=SR)
    return [(s["start"] / SR, s["end"] / SR) for s in stamps]


def speech_coverage(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    voiced = sum(e - s for s, e in speech_segments(audio))
    return round(voiced / (audio.size / SR), 4)


def _repeats(text: str, limit: int = 3) -> bool:
    words = text.lower().split()
    for n in (1, 2, 3):
        for i in range(len(words) - n * (limit + 1) + 1):
            gram = words[i:i + n]
            if all(words[i + k * n:i + (k + 1) * n] == gram for k in range(limit + 1)):
                return True
    return False


def keep_segment(seg) -> bool:
    """Whisper invents text over music and game audio; drop the usual signatures. [INFERRED] thresholds."""
    text = seg.text.strip()
    if not text:
        return False
    if seg.no_speech_prob > 0.6 and seg.avg_logprob < -1.0:
        return False
    if seg.compression_ratio > 2.4 or _repeats(text):
        return False
    return not STOCK_PHRASES.match(text)


def _expose_cuda_dlls() -> None:
    """CTranslate2 needs cuBLAS/cuDNN 12 DLLs; torch cu12x ships them in torch/lib."""
    # Best effort: without torch's DLLs faster-whisper falls back to CPU in transcribe().
    with contextlib.suppress(Exception):
        import torch
        lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        if os.path.isdir(lib):
            os.add_dll_directory(lib)
            os.environ["PATH"] = lib + os.pathsep + os.environ.get("PATH", "")


class FasterWhisperASR:
    def __init__(self, model: str = "small", device: str = "cuda", compute_type: str = "int8_float16"):
        self.model_name, self.device, self.compute_type = model, device, compute_type

    def _run(self, audio, language, device, compute_type):
        from faster_whisper import WhisperModel
        model = WhisperModel(self.model_name, device=device, compute_type=compute_type,
                             download_root=str(CACHE / "hf" / "faster-whisper"))
        segments, info = model.transcribe(audio, language=language, word_timestamps=True, vad_filter=True,
                                          condition_on_previous_text=False)
        return model, list(segments), info

    def transcribe(self, audio: np.ndarray, language: str | None = None) -> dict:
        if self.device == "cuda":
            _expose_cuda_dlls()
        model = None
        try:
            try:
                model, segments, info = self._run(audio, language, self.device, self.compute_type)
            except RuntimeError as exc:
                if self.device != "cuda":
                    raise
                log.warning("whisper on CUDA failed (%s); falling back to CPU int8", exc)
                model, segments, info = self._run(audio, language, "cpu", "int8")
            words = []
            for seg in segments:
                if not keep_segment(seg):
                    continue
                for w in seg.words or []:
                    words.append({"word": w.word.strip(), "start": round(w.start, 3),
                                  "end": round(w.end, 3), "probability": round(w.probability, 3)})
            return {"language": info.language, "words": words}
        finally:
            model = None
            gc.collect()
            with contextlib.suppress(Exception):  # cache release is best effort
                import torch
                torch.cuda.empty_cache()
