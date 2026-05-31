"""Pydantic request/response models for the API."""
from __future__ import annotations
import warnings
from enum import Enum
from typing import Literal, Optional
from pydantic import BaseModel, ConfigDict, Field

# Pydantic emits a (cosmetic) UserWarning that `register` shadows a BaseModel
# attribute even though BaseModel exposes no such name. Silence it locally.
warnings.filterwarnings(
    "ignore",
    message=r'Field name "register" in "AskRequest" shadows.*',
    category=UserWarning,
)


# ---- Ingest / lecture meta -------------------------------------------------

class IngestRequest(BaseModel):
    url: str


class LectureMeta(BaseModel):
    video_id: str
    title: str
    duration: float = 0.0
    segments: int
    concepts: int = 0
    cached: bool = False
    transcript_source: Literal["captions", "stt", "unknown"] = "unknown"


# ---- Concept map -----------------------------------------------------------

class Concept(BaseModel):
    id: str
    name: str
    one_line_desc: str
    first_timestamp: float = 0.0
    chunk_ids: list[int] = Field(default_factory=list)
    prerequisites: list[str] = Field(default_factory=list)


class ConceptMapResponse(BaseModel):
    video_id: str
    concepts: list[Concept]


# ---- Mastery ---------------------------------------------------------------

class MasteryState(str, Enum):
    UNSEEN = "unseen"
    ENGAGED = "engaged"
    SHAKY = "shaky"
    DEMONSTRATED = "demonstrated"


class MasteryEntry(BaseModel):
    concept_id: str
    name: str
    state: MasteryState
    score: float
    evidence: list[str] = Field(default_factory=list)


class SessionMasteryResponse(BaseModel):
    session_id: str
    video_id: str
    mastery: list[MasteryEntry]


# ---- Ask -------------------------------------------------------------------

class Register(str, Enum):
    MORE_VERNACULAR = "more_vernacular"
    BALANCED = "balanced"
    MORE_ENGLISH = "more_english"


class AskRequest(BaseModel):
    # `register` shadows a BaseModel-internal name; opt out of namespace protection.
    model_config = ConfigDict(protected_namespaces=())

    video_id: str
    question: str
    language: str = "en"
    session_id: Optional[str] = None
    register: Register = Register.BALANCED


class Citation(BaseModel):
    start: float
    end: float
    snippet: str


class MisconceptionBlock(BaseModel):
    detected: bool = False
    misconception: Optional[str] = None
    correction: Optional[str] = None


class AskResponse(BaseModel):
    answer: str
    language: str
    grounded: bool
    citations: list[Citation] = Field(default_factory=list)
    session_id: str
    concepts_touched: list[str] = Field(default_factory=list)
    misconception: MisconceptionBlock = Field(default_factory=MisconceptionBlock)

    # Testing-observability fields (additive). Exposed so the UI / harness
    # can show how close we are to the refusal floor and which concept we got
    # nearest to even when we refused.
    top_sim: float = 0.0
    nearest_concept: Optional[str] = None


class Health(BaseModel):
    ok: bool = True


# ---- Voice -----------------------------------------------------------------

class AskVoiceResponse(AskResponse):
    """Everything /ask returns, plus the transcript we heard and an audio id
    for the spoken answer (retrievable at GET /audio/{id})."""
    transcript: str = ""
    audio: Optional[str] = None  # audio_id; null if synthesis was skipped


class SpeakRequest(BaseModel):
    """On-demand TTS for an already-displayed text answer."""
    text: str
    language: str = "en"
    speaker: Optional[str] = None


class SpeakResponse(BaseModel):
    audio: str  # audio_id, retrievable at GET /audio/{id}


# ---- Viva ------------------------------------------------------------------

class VivaMode(str, Enum):
    QUIZ = "quiz"
    TEACH_BACK = "teach_back"


class VivaStartRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    session_id: str
    video_id: str
    mode: VivaMode = VivaMode.QUIZ
    register: Register = Register.BALANCED
    language: str = "en"
    speaker: Optional[str] = None


class VivaStartResponse(BaseModel):
    viva_id: str
    concept_id: Optional[str] = None
    concept_name: Optional[str] = None
    question: str
    audio: Optional[str] = None
    done: bool = False
    asked: int = 0
    total: int = 0


class MasteryUpdate(BaseModel):
    concept_id: str
    from_state: str
    to_state: str


class VivaAnswerResponse(BaseModel):
    transcript: str = ""
    verdict: Literal["correct", "partial", "incorrect", "unsupported"] = "incorrect"
    rationale: str = ""
    citations: list[Citation] = Field(default_factory=list)
    mastery_update: Optional[MasteryUpdate] = None
    reexplanation: Optional[str] = None
    next_question: Optional[str] = None
    next_concept_id: Optional[str] = None
    audio: Optional[str] = None
    done: bool = False
    asked: int = 0
    total: int = 0


class VivaSummaryRow(BaseModel):
    concept_id: str
    name: str
    state: str
    score: float


class VivaSummaryResponse(BaseModel):
    viva_id: str
    session_id: str
    video_id: str
    mode: VivaMode
    asked: int
    solid: list[VivaSummaryRow] = Field(default_factory=list)
    shaky: list[VivaSummaryRow] = Field(default_factory=list)
    remaining: list[VivaSummaryRow] = Field(default_factory=list)
    headline: str = ""
