import os
from pathlib import Path
from . import whisper_cpp
from .whisper_types import WhisperResult, WhisperSegment, WhisperToken
from .utils import download_model
from typing import List, Literal, Any
import ctypes
import librosa


class Whisper:
    WHISPER_SR = 16000

    def __init__(self, model: str | Path, strategy: int = 0, n_threads: int = 1) -> None:
        if isinstance(model, Path):
            model = str(model)
        model_path: str = model if os.path.isfile(model) else download_model(model)
        self.context = whisper_cpp.whisper_init_from_file(model_path.encode('utf-8'))
        self.params  = whisper_cpp.whisper_full_default_params(strategy)
        self.params.n_threads = n_threads
        self.params.print_special = False
        self.params.print_progress = False
        self.params.print_realtime = False
        self.params.print_timestamps = False

    def transcribe(
        self,
        file: str | Path,
        prompt: str | None = None,
        response_format: str = 'json',
        temperature: float = 0.8,
        language: str = 'en',
    ) -> Any:
        data, sr = librosa.load(file, sr=Whisper.WHISPER_SR)
        self.params.language = language.encode('utf-8')
        if prompt:
            self.params.initial_prompt = prompt.encode('utf-8')
        self.params.temperature = temperature
        result = self._full(data)
        return self._parse_format(result, response_format)

    def translate(self, file, prompt = None, response_format = 'json', temperature = 0.8) -> dict[str, Any]:
        data, sr = librosa.load(file, sr=Whisper.WHISPER_SR)
        self.params.translate = True
        self.params.initial_prompt = prompt.encode('utf-8')
        self.params.temperature = temperature
        result = self._full(data)
        return self._parse_format(result, response_format)

    def _full(self, data) -> WhisperResult:
        # run the inference
        r = whisper_cpp.whisper_full(ctypes.c_void_p(self.context), self.params, data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), len(data))
        if r != 0:
            raise "Error: {}".format(r)

        result: WhisperResult = {
            "task": "translate" if self.params.translate else "transcribe",
            "language": self.params.language,
            "duration": librosa.get_duration(y=data, sr=Whisper.WHISPER_SR),
        }

        segments: List[WhisperSegment] = []
        all_text = ''
        n_segments = whisper_cpp.whisper_full_n_segments(ctypes.c_void_p(self.context))
        for i in range(n_segments):
            t0  = whisper_cpp.whisper_full_get_segment_t0(ctypes.c_void_p(self.context), i) / 100.0
            t1  = whisper_cpp.whisper_full_get_segment_t1(ctypes.c_void_p(self.context), i) / 100.0
            txt = whisper_cpp.whisper_full_get_segment_text(ctypes.c_void_p(self.context), i).decode('utf-8')
            all_text += txt
            n_tokens = whisper_cpp.whisper_full_n_tokens(ctypes.c_void_p(self.context), i)
            tokens: List[WhisperToken] = []
            for j in range(n_tokens):
                token_data = whisper_cpp.whisper_full_get_token_data(ctypes.c_void_p(self.context), i, j)
                tokens.append({
                    "id": token_data.id,
                    "prob": token_data.p,
                    "logprob": token_data.plog,
                    "pt": token_data.pt,
                    "pt_sum": token_data.ptsum,
                })
            segments.append({
                "start": t0,
                "end": t1,
                "text": txt,
                "tokens": tokens,
            })

        result["segments"] = segments
        result["text"] = all_text.strip()
        return result

    def _parse_format(self, result: WhisperResult, response_format: Literal["json", "text", "srt", "verbose_json", "vtt"]) -> dict[str, Any]:
        return {
            "json": self._parse_format_json,
            "text": self._parse_format_text,
            "srt": self._parse_format_srt,
            "verbose_json": self._parse_format_verbose_json,
            "vtt": self._parse_format_vtt,
        }[response_format](result)

    def _parse_format_verbose_json(self, result: WhisperResult) -> dict[str, Any]:
        return {
            "task": result["task"],
            "language": result["language"],
            "duration": result["duration"],
            "text": result["text"],
            "segments": [{
                "id": i,
                "seek": s['start'],
                "start": s['start'],
                "end": s['end'],
                "text": s['text'],
                "tokens": [t["id"] for t in s["tokens"]],
                "temperature": self.params.temperature + self.params.temperature_inc * i,
                "avg_logprob": sum([t["logprob"] for t in s["tokens"]])/len(s["tokens"]),
                "compression_ratio": self.params.entropy_thold,
                "no_speech_prob": 0.0,
                "transient": False,
            } for i, s in enumerate(result["segments"])],
        }

    def _parse_format_json(self, result: WhisperResult) -> dict[str, Any]:
        return {
            "text": result["text"],
        }

    def _parse_format_text(self, result: WhisperResult) -> str:
        return result["text"]

    def _parse_format_srt(self, result: WhisperResult) -> str:
        output_tmpl = "{}\n{} --> {}\n{}\n"
        return "\n".join(
            [
                output_tmpl.format(
                    i + 1,
                    Whisper.format_time(s["start"]),
                    Whisper.format_time(s["end"]),
                    s["text"]
                ) for i, s in enumerate(result["segments"])
            ]
        )

    def _parse_format_vtt(self, result: WhisperResult) -> str:
        output_tmpl = "{}\n{} --> {} align:middle\n{}\n"
        return "\n".join(
            [
                output_tmpl.format(
                    i + 1,
                    Whisper.format_time(s["start"]),
                    Whisper.format_time(s["end"]),
                    s["text"]
                ) for i, s in enumerate(result["segments"])
            ]
        )

    def __dealloc__(self) -> None:
        # free the memory
        whisper_cpp.whisper_free(ctypes.c_void_p(self.context))

    @staticmethod
    def format_time(t: int) -> str:
        ms = t * 10
        h = ms / (1000 * 60 * 60)
        ms = ms - h * (1000 * 60 * 60)
        m = ms / (1000 * 60)
        ms = ms - m * (1000 * 60)
        s = ms / 1000
        ms = ms - s * 1000
        return "{:02}:{:02}:{:02}.{:03}".format(
            h,
            m,
            s,
            ms,
        )

