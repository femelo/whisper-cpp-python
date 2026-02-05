from typing import List
from typing_extensions import TypedDict, Literal

class WhisperToken(TypedDict):
    id: int
    prob: float
    logprob: float
    pt: float
    pt_sum: float

class WhisperSegment(TypedDict):
    start: int
    end: int
    text: str
    tokens: List[WhisperToken]

class WhisperResult(TypedDict):
    task: Literal["transcribe", "translate"]
    language: str
    duration: float
    segments: List[WhisperSegment]
    text: str

