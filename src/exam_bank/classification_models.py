from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TopicCandidate:
    paper_family: str
    topic: str
    subtopic: str
    score: float = 0.0
    methods: list[str] = field(default_factory=list)
    objects: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    boosts: list[str] = field(default_factory=list)
    source_scores: dict[str, float] = field(default_factory=dict)
    method_scores: dict[str, float] = field(default_factory=dict)
    object_cue_prior_score: float = 0.0
    object_anchor_bonus: float = 0.0
    object_protection_penalty: float = 0.0

    @property
    def label(self) -> str:
        return f"{self.paper_family}:{self.topic}:{self.subtopic}"

    @property
    def topic_label(self) -> str:
        return f"{self.topic}:{self.subtopic}"

    @property
    def has_method_and_object(self) -> bool:
        return bool(self.methods and self.objects)

    @property
    def source_method_total(self) -> float:
        return sum(self.source_scores.values()) + sum(self.method_scores.values())


@dataclass(frozen=True)
class QuestionPartSegment:
    part_label: str
    text: str
    classification_text: str


@dataclass(frozen=True)
class FamilyDecision:
    source_paper_family: str
    source_paper_code: str
    inferred_paper_family: str
    paper_family: str
    paper_family_confidence: str
    allowed_families: list[str]
    review_flags: list[str]


@dataclass(frozen=True)
class DifficultyDecision:
    difficulty: str
    confidence: str
    evidence: str
    uncertain: bool
    numeric_confidence: float
    review_flags: list[str]


@dataclass(frozen=True)
class ObjectCueDecision:
    detected_object_cues: list[str]
    topic_scores: dict[str, float]
    evidence: dict[str, list[str]]
    primary_topic: str
    flags: list[str]
    conflict_with_method_scoring: bool = False
    source_topic_scores: dict[str, dict[str, float]] = field(default_factory=dict)
