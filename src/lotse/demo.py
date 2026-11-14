"""What the demo does around one question: limits, a cache, and the answer as HTML.

Nothing here knows Gradio, so the tests run without a browser.
"""

import html
import json
import logging
import time
from collections import defaultdict, deque
from collections.abc import Callable
from datetime import date

from lotse import pipeline
from lotse.config import API_DOCS, MANIFEST
from lotse.generation.answer import locate
from lotse.pipeline import Result

MAX_QUESTION_CHARS = 300
QUESTIONS_PER_WINDOW = 10
WINDOW_SECONDS = 600
CACHED_ANSWERS = 1000
REPO_URL = "https://github.com/fAmesteR/sicherheitslotse"

TOO_LONG = f"Die Frage ist zu lang. Erlaubt sind höchstens {MAX_QUESTION_CHARS} Zeichen."
TOO_MANY = (
    f"Sie haben in den letzten {WINDOW_SECONDS // 60} Minuten {QUESTIONS_PER_WINDOW} Fragen "
    "gestellt. Bitte versuchen Sie es später noch einmal."
)
FAILED = "Die Frage konnte gerade nicht beantwortet werden. Bitte versuchen Sie es später erneut."
HINTS = {
    "no_answer": "Durchsucht werden nur die unten genannten Gesetze und Verordnungen.",
    "unverified": (
        "Das Sprachmodell hat geantwortet, aber seine Zitate oder Zahlen stehen so nicht in den "
        "Quellen. Deshalb wird die Antwort nicht angezeigt."
    ),
}
MARK = '<mark style="background:#ffe58a;color:#111">'

log = logging.getLogger(__name__)


class RateLimit:
    """At most `limit` questions per visitor in `window` seconds. Counters live in memory only."""

    def __init__(
        self,
        limit: int = QUESTIONS_PER_WINDOW,
        window: float = WINDOW_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.limit, self.window, self.clock = limit, window, clock
        self.asked: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, visitor: str) -> bool:
        now = self.clock()
        # Forget visitors whose last question is older than the window, so the table stays small.
        for gone in [v for v, times in self.asked.items() if now - times[-1] > self.window]:
            del self.asked[gone]
        times = self.asked[visitor]
        while times and now - times[0] > self.window:
            times.popleft()
        if len(times) >= self.limit:
            return False
        times.append(now)
        return True


def german_date(value: str) -> str:
    try:
        return date.fromisoformat(value).strftime("%d.%m.%Y")
    except ValueError:
        return value  # an ASR names its GMBl issue instead of a date


def highlighted(text: str, quotes: list[str]) -> str:
    """The passage text as HTML, with every quote that is found in it marked."""
    parts, end = [], 0
    for start, stop in sorted(span for quote in quotes if (span := locate(text, quote))):
        start = max(start, end)  # two quotes may overlap
        if start < stop:
            parts += [html.escape(text[end:start]), MARK, html.escape(text[start:stop]), "</mark>"]
            end = stop
    return "".join(parts) + html.escape(text[end:])


def render(result: Result) -> str:
    """The answer and, per cited source, where it stands and what was quoted from it."""
    parts = [f"<p>{html.escape(result.text)}</p>"]
    if hint := HINTS.get(result.status):
        parts.append(f"<p>{hint}</p>")
    quotes: dict[int, list[str]] = defaultdict(list)
    for citation in result.citations:
        quotes[citation.source].append(citation.quote)
    for n in sorted(quotes):
        chunk = result.passages[n - 1]
        name = html.escape(" ".join(filter(None, [chunk["doc"], chunk["section"]])))
        details = [html.escape(chunk["heading"]), f"Stand {german_date(chunk['version_date'])}"]
        link = f'<a href="{html.escape(chunk["url"])}" target="_blank" rel="noopener">Original</a>'
        parts.append(
            f"<p><b>[{n}] {name}</b> · {' · '.join(filter(None, details))} · {link}</p>"
            f'<blockquote style="white-space:pre-wrap">'
            f"{highlighted(chunk['text'], quotes[n])}</blockquote>"
        )
    return "\n".join(parts)


class Demo:
    """Answers one visitor's question, within the limits a public demo with a paid API needs."""

    def __init__(self, ask: Callable[[str], Result] = pipeline.ask, limit: RateLimit | None = None):
        self.ask = ask
        self.limit = limit or RateLimit()
        self.answers: dict[str, Result] = {}

    def respond(self, question: str, visitor: str) -> str:
        question = " ".join(question.split())
        if not question:
            return ""
        if len(question) > MAX_QUESTION_CHARS:
            return f"<p>{TOO_LONG}</p>"
        key = question.casefold()
        if key in self.answers:  # costs nothing, so it doesn't count against the limit
            return render(self.answers[key])
        if not self.limit.allow(visitor):
            return f"<p>{TOO_MANY}</p>"
        try:
            result = self.ask(question)
        except Exception:
            log.exception("A question failed")  # the question itself is not logged
            return f"<p>{FAILED}</p>"
        if result.status != "unverified":  # the model may pass on a second try, so don't keep it
            if len(self.answers) >= CACHED_ANSWERS:
                del self.answers[next(iter(self.answers))]  # the oldest one
            self.answers[key] = result
        return render(result)


def footer() -> str:
    """The notices under the demo, as Markdown."""
    docs = [
        doc for doc in json.loads(MANIFEST.read_text(encoding="utf-8")) if doc["doc"] in API_DOCS
    ]
    names = ", ".join(doc["doc"] for doc in docs)
    loaded = german_date(max(doc["downloaded"] for doc in docs))
    return (
        "**Keine Rechtsberatung.** Sicherheitslotse ist ein Studienprojekt. Antworten können "
        "falsch oder unvollständig sein. Maßgeblich ist der verlinkte Originaltext.\n\n"
        f"**{len(docs)} Dokumente, Stand {loaded}:** {names}. "
        "Quelle der Texte: gesetze-im-internet.de.\n\n"
        "**Datenschutz:** Ihre Frage und die gefundenen Textstellen gehen zur Beantwortung an die "
        "API von Mistral AI. Bitte geben Sie keine personenbezogenen Daten ein.\n\n"
        f"Code und Messwerte: [{REPO_URL.removeprefix('https://')}]({REPO_URL})"
    )
