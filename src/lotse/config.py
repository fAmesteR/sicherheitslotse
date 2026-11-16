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
# The commit of each model on the Hub, as the index was built with it, so an update there can't
# change the scores under the stored index and REFUSAL_THRESHOLD. space/README.md preloads the
# same commits. A new commit means building the index and tuning the threshold again.
MODEL_REVISIONS = {
    EMBEDDING_MODEL: "5617a9f61b028005a4858fdac845db406aefb181",
    RERANKER_MODEL: "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
    SMALL_RERANKER_MODEL: "1427fd652930e4ba29e8149678df786c240d8825",
}
# Search reads each chunk with its context prefix, the reranker without it. On the 60
# answerable dev questions the reranker's Recall@5 was 0.80 with the prefix and 0.83 without.
RERANK_PREFIX = False

# Documents whose text may be sent to the answer API. The ASR stay on this machine until BAuA
# allows their use.
API_DOCS = frozenset(LAWS)

# Where the public index and the demo live on Hugging Face. The dataset is private, because it
# holds document text and should not hand it out again.
INDEX_REPO = "fAmesteR/sicherheitslotse-index"
INDEX_REVISION: str | None = "0d614c5c35d2278c3948e7224008dcb953060d75"  # set after each upload
SPACE_REPO = "fAmesteR/sicherheitslotse"

ANSWER_MODEL = "mistral-small-2603"  # a dated name, so the model can't change under the evaluation
ANSWER_MAX_TOKENS = 600  # several verbatim quotes don't fit in less
ANSWER_PASSAGES = 5
ANSWER_TIMEOUT_MS = 20_000  # p95 is about 3 s, so this only cuts off a call that hangs
# Refuse without an API call if the best rerank score is below this. Tuned on the 80 dev questions
# for bge-reranker-v2-m3 reading the bare text: the middle of the gap between the lowest answerable
# question (-5.00) and the unanswerable ones below it (-6.58 and lower). It refuses 8 of 20
# unanswerable ones and no answerable one. Scores depend on the reranker and on what it reads, so
# retune it whenever either changes.
REFUSAL_THRESHOLD: float | None = -5.8

# The judge is a different model from the one that answers. It has no dated name, so the result
# file records which one ran.
JUDGE_MODEL = "mistral-large-4-0"
JUDGE_EFFORT = "medium"  # how much it reasons before the verdict: low, medium or high
JUDGE_MAX_TOKENS = 3000  # its reasoning counts as output
JUDGE_TIMEOUT_MS = 120_000  # reasoning first makes it slow

# USD per million tokens, input and output: list prices read on 2026-11-13.
PRICES = {ANSWER_MODEL: (0.15, 0.60), JUDGE_MODEL: (0.70, 2.10)}
