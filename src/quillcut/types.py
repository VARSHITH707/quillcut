"""EditPlan: the contract between planner and renderer. JSON Schema is exported from these models."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

Layout = Literal["center_crop", "wide_fill", "split", "fit_blur"]
MAX_HOLD_SECONDS = 2.0


class SourceRef(BaseModel):
    path: str
    sha1: str
    width: int
    height: int
    fps: str
    duration: float
    has_audio: bool
    content_box: tuple[int, int, int, int]   # w, h, x, y after letterbox removal


class Segment(BaseModel):
    id: str
    source: str
    src_in: float
    src_out: float
    speed: float = 1.0
    layout: Layout = "center_crop"
    second_source: str | None = None          # bottom half for layout == "split"
    second_in: float | None = None
    audio_gain: float = 1.0
    hold_after: float = 0.0                   # deliberate end-frame hold, last segment only
    push_in: float = 0.0                      # zoom growth over the segment, e.g. 0.06
    zoom: float = 1.0                         # static punch-in factor (1.0 = none), centred on the crop
    crop_x: float | None = None               # normalised x (0-1) of the person to keep in a vertical crop
    video_source: str | None = None           # b-roll: picture from here, audio stays from `source`
    video_in: float | None = None
    rationale: str = ""

    @property
    def out_duration(self) -> float:
        return (self.src_out - self.src_in) / self.speed + self.hold_after

    @model_validator(mode="after")
    def _check(self):
        if self.src_out - self.src_in < 0.2:
            raise ValueError(f"{self.id}: segment shorter than 0.2 s")
        if not 0.5 <= self.speed <= 2.0:
            raise ValueError(f"{self.id}: speed {self.speed} outside 0.5–2.0")
        if self.layout == "split" and (self.second_source is None or self.second_in is None):
            raise ValueError(f"{self.id}: split layout needs second_source and second_in")
        if not 1.0 <= self.zoom <= 1.35:
            raise ValueError(f"{self.id}: zoom {self.zoom} outside 1.0-1.35")
        if not 0 <= self.hold_after <= MAX_HOLD_SECONDS:
            raise ValueError(f"{self.id}: hold over {MAX_HOLD_SECONDS}s is padding")
        return self


class Word(BaseModel):
    text: str
    start: float
    end: float
    emphasis: bool = False                    # key term: drawn in the accent colour, slightly larger


class TextCard(BaseModel):
    # hook = opening title, label = plain text, speech = word-timed captions, lower_third = speaker/context bar,
    # callout = key-phrase pill, cta = closing line
    kind: Literal["hook", "label", "speech", "lower_third", "callout", "cta"]
    text: str
    start: float
    end: float
    lines: list[str]                          # resolved line breaks (planner decides)
    font: str
    font_size: int
    pos: tuple[int, int]                      # ASS \pos centre, 1080x1920 space
    bbox_norm: tuple[float, float, float, float]
    words: list[Word] = Field(default_factory=list)


class MusicBed(BaseModel):
    file: str
    licence: str
    start_offset: float          # track time at clip t=0; negative = track starts later
    level_db: float = -5.0       # bed loudness relative to the source when nobody speaks
    duck_db: float = -15.0       # bed loudness relative to the source under speech
    duck: list[tuple[float, float]] = Field(default_factory=list)   # clip-time speech spans


class SfxHit(BaseModel):
    file: str
    licence: str
    t: float
    gain_db: float = -6.0        # relative to the source loudness


class AudioPlan(BaseModel):
    target_lufs: float = -14.0
    true_peak_max: float = -1.0
    fade_out: float = 0.0
    denoise: bool = False        # gentle broadband noise reduction on the dialogue
    music: MusicBed | None = None
    sfx: list[SfxHit] = Field(default_factory=list)


class OutputSpec(BaseModel):
    width: int = 1080
    height: int = 1920
    fps: str


class EditPlan(BaseModel):
    schema_version: str = "2.0"
    plan_version: int = 1
    repair_of: int | None = None
    clip_id: str
    project_id: str
    format_key: str
    mode: Literal["cinematic", "commentary"]
    script: str
    sources: dict[str, SourceRef]
    rights: dict
    moment: dict
    segments: list[Segment]
    text_cards: list[TextCard] = Field(default_factory=list)
    caption_style: dict
    audio: AudioPlan = Field(default_factory=AudioPlan)
    output: OutputSpec
    look: dict = Field(default_factory=dict)      # eq grade, e.g. {"contrast": 1.06}
    progress_bar: str | None = None               # bar colour hex, None = no bar
    logo: dict | None = None                      # persistent brand logo: {file, width (frame fraction), top (frame fraction)}
    hard_floor_seconds: float
    placeholders_used: list[str] = Field(default_factory=list)
    rationale: list[str] = Field(default_factory=list)

    @property
    def duration(self) -> float:
        return round(sum(s.out_duration for s in self.segments), 3)

    @model_validator(mode="after")
    def _check(self):
        if not self.segments:
            raise ValueError("plan has no segments")
        for seg in self.segments:
            for key, start in ((seg.source, seg.src_in), (seg.second_source, seg.second_in),
                               (seg.video_source, seg.video_in)):
                if key is None:
                    continue
                if key not in self.sources:
                    raise ValueError(f"{seg.id}: unknown source '{key}'")
                end = start + (seg.src_out - seg.src_in)
                if start < 0 or end > self.sources[key].duration + 0.05:
                    raise ValueError(f"{seg.id}: {key} range {start:.2f}-{end:.2f} outside source")
        if any(s.hold_after for s in self.segments[:-1]):
            raise ValueError("holds are only allowed on the last segment")
        return self
