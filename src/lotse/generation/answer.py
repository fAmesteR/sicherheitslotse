"""The answer schema, and the check that decides whether an answer is shown.

The check is deterministic. It catches an invented quote or number, but it can't tell whether a real
quote supports its sentence: „darf acht Stunden nicht überschreiten“ can back „Sie müssen acht
Stunden arbeiten [1].“ The faithfulness metric measures how often that happens, and the page shows
each quote under the answer so a reader can compare.

Known limit: "ein" and "eine" count as the number 1 only before a span of time, and "Acht" in
"außer Acht lassen" is read as 8. In a list of references such as "§§ 3, 4 und 5" only the first
number is seen as a reference, so 4 and 5 count as values a sentence may state.
"""

import re
import unicodedata

from pydantic import BaseModel, Field

from lotse.chunking import context_prefix, passage

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
# Not glued to a letter: Unicode normalisation turns "m²" into "m2", and neither that 2 nor
# the 2 in a name like "M1.2" is a value.
NUMBER = re.compile(r"(?<![^\W_])(?<!\d[.,])\d+(?:[.,]\d+)*")
# Laws write numbers as words too, so "8 Stunden" and "acht Stunden" are the same value. A number
# word the check can't read lets any value through, so this goes up to hundred.
ONES = ["ein", "zwei", "drei", "vier", "fünf", "sechs", "sieben", "acht", "neun"]
TENS = ["zwanzig", "dreißig", "vierzig", "fünfzig", "sechzig", "siebzig", "achtzig", "neunzig"]
TEENS = ["dreizehn", "vierzehn", "fünfzehn", "sechzehn", "siebzehn", "achtzehn", "neunzehn"]
WORDS = {word: value for value, word in enumerate(ONES[1:] + ["zehn", "elf", "zwölf"], start=2)}
WORDS |= {word: value for value, word in enumerate(TEENS, start=13)}
WORDS |= {word: 10 * tens for tens, word in enumerate(TENS, start=2)}
WORDS |= {f"{one}und{ten}": n + WORDS[ten] for n, one in enumerate(ONES, 1) for ten in TENS}
WORDS = {word: str(value) for word, value in WORDS.items()} | {"hundert": "100"}
NUMBER_WORD = re.compile(rf"\b(?:{'|'.join(sorted(WORDS, key=len, reverse=True))})\b")
# "ein" is an article, so it counts as 1 only before a span of time: „ein Jahr“, „einer Woche“.
# „eine halbe Stunde“ is 0,5, as a law would write it.
ONE_SPAN = re.compile(
    r"\bein(?:e|en|em|er|es)? (?:jahr(?:e?s)?|monat(?:e?s)?|woche|tag(?:e?s)?|stunde|minute)"
    r"(?![\w-])"  # „einem Tages-Lärmexpositionspegel“ is no day
)
HALF = re.compile(r"\bhalbe?[mnrs]?\b")
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
# Numbers that name a place in a text and state nothing: "§ 5", "Abs. 2", "Anhang 1 Nr. 3.7".
REFERENCE = re.compile(
    r"(?:§+|\b(?:Absatz|Absätze|Abs\.|Satz|Sätze|Nummer|Nummern|Nr\.|Buchstabe|Anhang|Abschnitt"
    r"|Teil|Tabelle|Artikel|Art\.))\s*\d+(?:\.\d+)*[a-z]?\b"
)

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
    text = normalise(MARKER.sub(" ", text))
    found = {ORDINALS[stem] for stem in ORDINAL.findall(text)}
    text = text.lower()
    found |= {WORDS[word] for word in NUMBER_WORD.findall(text)}
    found |= {"1" for _ in ONE_SPAN.finditer(text)} | {"0,5" for _ in HALF.finditer(text)}
    return found | set(NUMBER.findall(text))


def values(text: str) -> set[str]:
    """The numbers a text states. Left out are those that only name a place in it: the label
    "(2)" at the start of a line, "§ 5", "Abs. 1". Otherwise "(2)" in a source would allow an
    invented "zwei Ersthelfer"."""
    return numbers(REFERENCE.sub(" ", LINE_LABEL.sub(" ", text)))


def squeeze(text: str) -> str:
    """For comparing a quote with its passage: case and spacing don't count."""
    return "".join(normalise(text).casefold().split())


def whole_words(text: str, start: int, end: int) -> bool:
    """Whether text[start:end] begins and ends at a word's edge: „zulässig“ is not in „unzulässig“."""
    cut_before = 0 < start and text[start - 1].isalnum() and text[start].isalnum()
    cut_after = end < len(text) and text[end - 1].isalnum() and text[end].isalnum()
    return not (cut_before or cut_after)


def locate(text: str, quote: str) -> tuple[int, int] | None:
    """Where a quote sits in its passage text, as (start, end). The check accepts a quote only if
    this finds it, and the demo highlights that part. Spacing doesn't count inside a quote, since
    PDFs split and join words, but a quote must begin and end with whole words."""
    quote = normalise(quote).strip(" \"'.,;:!?")
    needles = [squeeze(quote), squeeze(QUOTED_LABEL.sub(" ", quote))]
    labels = {at for match in LINE_LABEL.finditer(text) for at in range(*match.span())}
    for skipped in (set(), labels):
        squeezed, positions = "", []  # positions[i] is where squeezed[i] sits in text
        for at, char in enumerate(text):
            if at not in skipped:
                piece = squeeze(char)
                squeezed += piece
                positions += [at] * len(piece)
        for needle in filter(None, needles):
            found = squeezed.find(needle)
            while found >= 0:
                span = positions[found], positions[found + len(needle) - 1] + 1
                if whole_words(text, *span):
                    return span
                found = squeezed.find(needle, found + 1)
    return None


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


def check(answer: Answer, chunks: list[dict]) -> str | None:
    """Why an answer can't be shown, in words the model can act on. None if every rule holds."""
    if not answer.citations:
        return "Es fehlen Zitate."
    for citation in answer.citations:
        if not 1 <= citation.source <= len(chunks):
            return f"Quelle [{citation.source}] gibt es nicht."
        quote = normalise(citation.quote).strip(" \"'.,;:!?")
        if len(QUOTED_LABEL.sub(" ", quote).split()) < 3:
            return f"Das Zitat „{citation.quote}“ hat weniger als drei Wörter."
        if locate(chunks[citation.source - 1]["text"], citation.quote) is None:
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
        # Every number in a sentence must stand in a source that sentence names, prefix
        # included. A number the sentence states as a value must be a value there too, not
        # only a paragraph label or a section number. The question doesn't count: a visitor
        # could ask „Darf ich 14 Stunden arbeiten?“ and get „Sie dürfen 14 Stunden arbeiten [1].“
        sources = [chunks[n - 1] for n in markers]
        named = set().union(*(numbers(passage(chunk)) for chunk in sources))
        stated = set().union(
            *(values(chunk["text"]) for chunk in sources),
            *(values(context_prefix(c["doc"], c["section"], c["heading"])) for c in sources),
        )
        if unknown := (numbers(sentence) - named) | (values(sentence) - stated):
            return (
                f"Die Zahl {min(unknown)} im Satz „{sentence}“ steht nicht in den dort "
                "genannten Quellen."
            )
    return None
