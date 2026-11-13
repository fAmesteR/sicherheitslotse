"""The answer schema, and the check that decides whether an answer is shown.

The check is deterministic. It catches an invented quote or number, but it can't tell whether a real
quote supports its sentence. The faithfulness metric measures that.

Known limit: "ein" and "eine" are not read as the number 1, and "Acht" in "außer Acht lassen" is
read as 8.
"""

import re
import unicodedata

from pydantic import BaseModel, Field

from lotse.chunking import passage

NO_ANSWER = "Dazu finde ich in den Quellen keine Antwort."
UNVERIFIED = "Keine belegte Antwort möglich."


class Citation(BaseModel):
    source: int = Field(description="Nummer n der Quelle [n]")
    quote: str = Field(description="Wörtlicher Ausschnitt aus dem Text dieser Quelle")


class Answer(BaseModel):
    answerable: bool = Field(description="false, wenn die Antwort nicht in den Quellen steht")
    answer: str = Field(description="Antwort auf Deutsch, jeder Satz endet mit [n]")
    citations: list[Citation]


# Characters that differ between a PDF, a law's XML and a model's output without changing the wording.
SAME = str.maketrans(
    {
        "\u00ad": None,  # soft hyphen
        **dict.fromkeys("„“”«»", '"'),
        **dict.fromkeys("‚‘’", "'"),
        **dict.fromkeys("‐‑‒–—−", "-"),
    }
)
MARKER = re.compile(r"\[(\d+(?:, ?\d+)*)\]")  # "[1]", and "[1, 2]" as some models write it
NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
# Laws write small numbers as words, so "8 Stunden" and "acht Stunden" are the same value.
WORDS = "zwei drei vier fünf sechs sieben acht neun zehn elf zwölf"
WORDS = {word: str(value) for value, word in enumerate(WORDS.split(), start=2)}
NUMBER_WORD = re.compile(rf"\b(?:{'|'.join(WORDS)})\b")
UNIT_POWERS = str.maketrans("", "", "²³")  # the 2 in "m²" is no value
# "jedes dritte Jahr" is "alle drei Jahre". Lower case only, because "Dritte" are third parties,
# and without the eighth, because "achten" is a verb.
ORDINALS = {"zweit": "2", "dritt": "3", "viert": "4", "fünft": "5", "sechst": "6", "siebt": "7"}
ORDINALS |= {"neunt": "9", "zehnt": "10", "elft": "11", "zwölft": "12"}
ORDINAL = re.compile(rf"\b({'|'.join(ORDINALS)})e[rnsm]?\b")
# The labels of a list are layout: "dass\na) die ...\nb) die" may be quoted as "dass die ... die".
# In a passage only the start of a line counts, because "am 1. Mai" inside a line is a date.
LABEL = r"(?:\(\d{1,2}[a-z]?\)|\d{1,2}\.|[a-z]{1,2}\))"
LINE_LABEL = re.compile(rf"(?m)^\s*{LABEL}(?=\s)")
QUOTED_LABEL = re.compile(rf"(?<!\S){LABEL}(?=\s|$)")

TRAILING_MARKERS = re.compile(rf"([.!?])((?:\s*{MARKER.pattern})+)")
MONTHS = "Januar|Februar|März|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember"
# Full stops that end no sentence: "Abs. 1", "z. B.", "am 1. Januar".
INNER_STOP = re.compile(
    r"\b(?:Abs|Nr|Art|S|ca|mind|max|min)\.(?= [\d§])"
    r"|\b(?:bzw|ggf|vgl|sog|inkl|einschl|gem|evtl|z\. ?B|d\. ?h|u\. ?a|o\. ?Ä|i\. ?d\. ?R"
    r"|i\. ?V\. ?m)\.(?= )"
    rf"|(?<=\d)\.(?= (?:[a-zäöü]|{MONTHS}))"
)
SENTENCE_END = re.compile(r"""(?:(?<=[.!?])|(?<=[.!?]["')]))\s+""")


def normalise(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text.translate(SAME)).split())


def numbers(text: str) -> set[str]:
    text = normalise(MARKER.sub(" ", text.translate(UNIT_POWERS)))
    found = {ORDINALS[stem] for stem in ORDINAL.findall(text)}
    text = text.lower()
    return found | set(NUMBER.findall(text)) | {WORDS[word] for word in NUMBER_WORD.findall(text)}


def squeeze(text: str) -> str:
    """For comparing a quote with its passage: case and spacing don't count."""
    return "".join(normalise(text).casefold().split())


def sentences(text: str) -> list[str]:
    """When in doubt, split: a piece cut off by mistake only fails the check, while two sentences
    read as one would let the second go without a source."""
    found = []
    for line in text.splitlines():
        line = TRAILING_MARKERS.sub(r"\2\1", normalise(line))  # "Satz. [1]" is "Satz [1]."
        line = INNER_STOP.sub(lambda match: match[0].replace(".", "\0"), line)
        for piece in SENTENCE_END.split(line):
            if re.search(r"\w", MARKER.sub("", piece)):
                found.append(piece.replace("\0", "."))
    return found


def check(answer: Answer, chunks: list[dict], question: str = "") -> str | None:
    """Why an answer can't be shown, in words the model can act on. None if every rule holds."""
    if not answer.citations:
        return "Es fehlen Zitate."
    for citation in answer.citations:
        if not 1 <= citation.source <= len(chunks):
            return f"Quelle [{citation.source}] gibt es nicht."
        quote = normalise(citation.quote).strip(" \"'.,;:!?")
        words = QUOTED_LABEL.sub(" ", quote)
        if len(words.split()) < 3:
            return f"Das Zitat „{citation.quote}“ hat weniger als drei Wörter."
        text = chunks[citation.source - 1]["text"]
        texts = squeeze(text), squeeze(LINE_LABEL.sub(" ", text))
        if not any(squeeze(q) in t for q in (quote, words) for t in texts):
            return (
                f"Das Zitat „{citation.quote}“ steht nicht wörtlich im Text von Quelle "
                f"[{citation.source}]. Kopiere einen kürzeren, zusammenhängenden Ausschnitt, ohne "
                "Wörter auszulassen oder zu ergänzen."
            )

    cited = {citation.source for citation in answer.citations}
    found = sentences(answer.answer)
    if not found:
        return "Die Antwort ist leer."
    for sentence in found:
        markers = {int(n) for group in MARKER.findall(sentence) for n in group.split(",")}
        if not markers:
            return f"Der Satz „{sentence}“ hat keine Quellenangabe [n]."
        if markers - cited:
            return f"Zu [{min(markers - cited)}] fehlt ein Zitat."
        # A sentence's numbers must come from the sources that sentence names, prefix included,
        # or from the question, which the user wrote.
        known = numbers(question).union(*(numbers(passage(chunks[n - 1])) for n in markers))
        if unknown := numbers(sentence) - known:
            return (
                f"Die Zahl {min(unknown)} im Satz „{sentence}“ steht nicht in den dort "
                "genannten Quellen."
            )
    return None
