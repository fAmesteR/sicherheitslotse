import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PDF_DIR = RAW_DIR / "pdf"
PARSED_DIR = ROOT / "data" / "parsed"
MANIFEST = ROOT / "data" / "manifest.json"
CHUNKS = ROOT / "data" / "chunks.jsonl"
# LOTSE_INDEX_DIR picks another index, for example index_public for numbers that match the demo.
INDEX_DIR = Path(os.environ.get("LOTSE_INDEX_DIR") or ROOT / "index")
PUBLIC_INDEX_DIR = ROOT / "index_public"  # the part of the index that may leave this machine

GII_URL = "https://www.gesetze-im-internet.de"
BAUA_ASR_URL = "https://www.baua.de/DE/Angebote/Regelwerk/ASR"

# Citation label -> slug on gesetze-im-internet.de. The slugs are not derivable from jurabk.
LAWS = {
    "ArbSchG": "arbschg",
    "ArbStättV": "arbst_ttv_2004",
    "BetrSichV": "betrsichv_2015",
    "GefStoffV": "gefstoffv_2010",
    "ASiG": "asig",
    "ArbMedVV": "arbmedvv",
    "PSA-BV": "psa-bv",
    "LärmVibrationsArbSchV": "l_rmvibrationsarbschv",
    "BioStoffV": "biostoffv_2013",
    "LasthandhabV": "lasthandhabv",
    "ArbZG": "arbzg",
}

# Citation label -> (file stem, title, version as the first page prints it). BAuA's site blocks
# scripts, so these are saved by hand, and only as full texts: a "Textauszug" change sheet prints
# deleted wording as if it were current.
PDFS = {
    "ASR A1.8": ("ASR-A1-8", "Verkehrswege", "GMBl 2024, S. 412"),
    "ASR A2.3": ("ASR-A2-3", "Fluchtwege und Notausgänge", "GMBl 2024, S. 913"),
    "ASR A3.4": ("ASR-A3-4", "Beleuchtung und Sichtverbindung", "Ausgabe: Mai 2023"),
    "ASR A3.5": ("ASR-A3-5", "Raumtemperatur", "GMBl 2022, S. 198"),
    "ASR A4.1": ("ASR-A4-1", "Sanitärräume", "GMBl 2022, S. 212"),
}

EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_MAX_LENGTH = 512
MAX_CHUNK_TOKENS = 400

SEARCH_CANDIDATES = 30  # from BM25 and from dense search, before fusion
RRF_K = 60

RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
SMALL_RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
RERANK_CANDIDATES = 20
RERANK_MAX_LENGTH = 512

# Documents whose text may be sent to the answer API. The ASR stay on this machine until BAuA
# allows their use.
API_DOCS = frozenset(LAWS)

# Where the public index and the demo live on Hugging Face. The dataset is private, because it
# holds document text and should not hand it out again.
INDEX_REPO = "fAmesteR/sicherheitslotse-index"
INDEX_REVISION: str | None = "decc67c9c8c5c42eab45e097a8cd207a394231cb"  # set after each upload
SPACE_REPO = "fAmesteR/sicherheitslotse"

ANSWER_MODEL = "mistral-small-2603"  # a dated name, so the model can't change under the evaluation
ANSWER_MAX_TOKENS = 600  # several verbatim quotes don't fit in less
ANSWER_PASSAGES = 5
# Refuse without an API call if the best rerank score is below this. Not tuned yet: that needs the
# unanswerable dev questions.
REFUSAL_THRESHOLD: float | None = None

# The judge is a different model from the one that answers. It has no dated name, so the result
# file records which one ran.
JUDGE_MODEL = "mistral-large-4-0"
JUDGE_EFFORT = "medium"  # how much it reasons before the verdict: low, medium or high
JUDGE_MAX_TOKENS = 3000  # its reasoning counts as output

# USD per million tokens, input and output: list prices read on 2026-11-13.
PRICES = {ANSWER_MODEL: (0.15, 0.60), JUDGE_MODEL: (0.70, 2.10)}
