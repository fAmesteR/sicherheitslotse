from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PDF_DIR = RAW_DIR / "pdf"
PARSED_DIR = ROOT / "data" / "parsed"
MANIFEST = ROOT / "data" / "manifest.json"
CHUNKS = ROOT / "data" / "chunks.jsonl"
INDEX_DIR = ROOT / "index"

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

ANSWER_MODEL = "mistral-small-2603"  # a dated name, so the model can't change under the evaluation
ANSWER_MAX_TOKENS = 600  # several verbatim quotes don't fit in less
ANSWER_PASSAGES = 5
# Refuse without an API call if the best rerank score is below this. Not tuned yet: that needs the
# unanswerable dev questions.
REFUSAL_THRESHOLD: float | None = None
