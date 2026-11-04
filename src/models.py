from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Posting:
    doc_id: int
    term_freq: int


@dataclass(frozen=True, slots=True)
class DocMeta:
    doc_id: int
    doc_len: int
    title: str = ""