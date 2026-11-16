"""What the demo does around one question: limits, a cache, and the answer as HTML.

Nothing here knows Gradio, so the tests run without a browser.
"""

import html
import ipaddress
import json
import logging
import os
import threading
import time
from collections import Counter, defaultdict, deque
from collections.abc import Callable
from contextlib import AbstractContextManager
from datetime import UTC, date, datetime

from lotse import pipeline
from lotse.config import API_DOCS, MANIFEST
from lotse.generation import llm
from lotse.generation.answer import locate
from lotse.metrics import percentile
from lotse.pipeline import Result

MAX_QUESTION_CHARS = 300
QUESTIONS_PER_WINDOW = 10
WINDOW_SECONDS = 600
# What the demo may spend on the API per UTC day, about 3,000 questions. A visitor who changes
# address for every question gets past the rate limit, but not past this.
DAILY_BUDGET_USD = 1.0
CACHED_ANSWERS = 1000
KEPT_QUERIES = 10_000  # for /metrics; the log lines themselves go to the server's output
REPO_URL = "https://github.com/fAmesteR/sicherheitslotse"
# Read X-Forwarded-For only where a proxy is known to stand in front: on a Hugging Face Space,
# which sets SPACE_ID, or where LOTSE_TRUST_PROXY=1 says so. Set it only if the app can't be
# reached around the proxy, since anyone who reaches it directly can write that header.
TRUST_PROXY = os.environ.get("LOTSE_TRUST_PROXY", "1" if os.environ.get("SPACE_ID") else "0") == "1"

TOO_LONG = f"Die Frage ist zu lang. Erlaubt sind höchstens {MAX_QUESTION_CHARS} Zeichen."
TOO_MANY = (
    f"Sie haben in den letzten {WINDOW_SECONDS // 60} Minuten {QUESTIONS_PER_WINDOW} Fragen "
    "gestellt. Bitte versuchen Sie es später noch einmal."
)
OVER_BUDGET = "Das Tagesbudget der Demo ist aufgebraucht. Bitte versuchen Sie es morgen wieder."
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
query_log = logging.getLogger("lotse.queries")


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
        self.lock = threading.Lock()  # the page and the API ask from several threads

    def allow(self, visitor: str) -> bool:
        with self.lock:
            now = self.clock()
            # Forget visitors whose last question is older than the window, so the table stays
            # small.
            for gone in [v for v, times in self.asked.items() if now - times[-1] > self.window]:
                del self.asked[gone]
            times = self.asked[visitor]
            while times and now - times[0] > self.window:
                times.popleft()
            if len(times) >= self.limit:
                return False
            times.append(now)
            return True


def public(address: str) -> bool:
    try:
        return ipaddress.ip_address(address).is_global
    except ValueError:
        return False


def visitor(host: str | None, forwarded: str = "", trust_proxy: bool | None = None) -> str:
    """Who asks, for the rate limit. Behind a proxy, as on Hugging Face, the connection comes from
    the proxy's private address and the visitor's is in X-Forwarded-For. Each proxy appends the
    address it saw, so read from the right and take the first public one: anything a visitor wrote
    there stands further left. A direct connection's header is ignored, since anyone can set it,
    and so is every header unless a proxy is trusted (see TRUST_PROXY)."""
    global warned
    host = host or ""  # an ASGI server may not know the peer
    if public(host):
        return host
    if TRUST_PROXY if trust_proxy is None else trust_proxy:
        for hop in reversed(forwarded.split(",")):
            if public(hop := hop.strip()):
                return hop
    if not warned:
        log.warning("No public visitor address: everyone from %r shares one rate limit.", host)
        warned = True
    return host


warned = False


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

    def __init__(
        self,
        ask: Callable[[str], Result] = pipeline.ask,
        limit: RateLimit | None = None,
        meter: Callable[[], AbstractContextManager[Counter]] = llm.meter,
        clock: Callable[[], float] = time.perf_counter,
        budget: float = DAILY_BUDGET_USD,
    ):
        self.ask, self.meter, self.clock, self.budget = ask, meter, clock, budget
        self.limit = limit or RateLimit()
        self.answers: dict[str, Result] = {}
        self.queries: deque[dict] = deque(maxlen=KEPT_QUERIES)
        self.spent = ("", 0.0)  # UTC day and USD spent on it
        # Questions run at the same time, so the cache, the query log and the spending are
        # guarded. The question itself runs outside: pipeline.retrieve guards the models, and
        # each question's cost comes from its own meter.
        self.guard = threading.Lock()

    def answer(self, question: str, visitor: str) -> Result | str:
        """The checked result, or a notice if the question never reached the pipeline."""
        question = " ".join(question.split())
        if not question:
            return ""
        if len(question) > MAX_QUESTION_CHARS:
            return TOO_LONG
        key = question.casefold()
        with self.guard:
            cached = self.answers.get(key)
        if cached:  # costs nothing, so it doesn't count against the limit
            self.record(cached.status, cached=True)
            return cached
        if not self.limit.allow(visitor):
            self.record("rate_limited")
            return TOO_MANY
        if self.over_budget():
            self.record("over_budget")
            return OVER_BUDGET
        start = self.clock()
        with self.meter() as spent:
            try:
                result = self.ask(question)
            except Exception:
                log.exception("A question failed")  # the question itself is not logged
                result = None
        cost = spent["prompt"], spent["completion"], spent["usd"]
        if result is None:  # a first call may have been paid before the second one failed
            self.record("failed", self.clock() - start, *cost)
            return FAILED
        self.record(result.status, self.clock() - start, *cost)
        if result.status != "unverified":  # the model may pass on a second try, so don't keep it
            with self.guard:
                if key not in self.answers and len(self.answers) >= CACHED_ANSWERS:
                    del self.answers[next(iter(self.answers))]  # the oldest one
                self.answers[key] = result
        return result

    def over_budget(self) -> bool:
        today = datetime.now(UTC).date().isoformat()
        with self.guard:
            return self.spent[0] == today and self.spent[1] >= self.budget

    def respond(self, question: str, visitor: str) -> str:
        """The same as HTML, for the page."""
        outcome = self.answer(question, visitor)
        if isinstance(outcome, Result):
            return render(outcome)
        return f"<p>{outcome}</p>" if outcome else ""

    def record(
        self,
        status: str,
        seconds: float = 0.0,
        prompt: int = 0,
        completion: int = 0,
        usd: float = 0.0,
        cached: bool = False,
    ) -> None:
        """One log line per query: when, what happened, how long, how many tokens, what it cost.
        Neither the question nor the visitor's address is in it."""
        entry = {
            "time": datetime.now(UTC).isoformat(timespec="seconds"),
            "status": status,
            "cached": cached,
            "seconds": round(seconds, 3),
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "usd": round(usd, 6),
        }
        with self.guard:
            self.queries.append(entry)
            day = entry["time"][:10]
            self.spent = (day, (self.spent[1] if self.spent[0] == day else 0.0) + usd)
        query_log.info(json.dumps(entry))

    def summary(self) -> dict:
        """Latency and cost since the start, over at most the last KEPT_QUERIES queries."""
        with self.guard:
            queries = list(self.queries)
        served = [q for q in queries if q["status"] in ("answered", "no_answer", "unverified")]
        fresh = sorted(q["seconds"] for q in served if not q["cached"])
        usd = sum(q["usd"] for q in queries)
        return {
            "queries": len(queries),
            "by_status": dict(Counter(q["status"] for q in queries)),
            "from_cache": sum(q["cached"] for q in served),
            # Latency of the queries that ran the pipeline. A cached answer takes no time.
            "p50_seconds": percentile(fresh, 0.50),
            "p95_seconds": percentile(fresh, 0.95),
            "usd_total": round(usd, 6),
            "usd_per_1000_queries": round(1000 * usd / len(served), 4) if served else None,
        }


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
        # Each source shows its own version date; this is when the texts were downloaded.
        f"**{len(docs)} Dokumente, abgerufen am {loaded}:** {names}. "
        "Quelle der Texte: gesetze-im-internet.de.\n\n"
        "**Datenschutz:** Ihre Frage und die gefundenen Textstellen gehen zur Beantwortung an die "
        "API von Mistral AI. Bitte geben Sie keine personenbezogenen Daten ein.\n\n"
        f"Code und Messwerte: [{REPO_URL.removeprefix('https://')}]({REPO_URL})"
    )
