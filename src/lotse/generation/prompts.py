"""What the answer model is told. The check in answer.py enforces the same rules."""

from lotse.chunking import passage
from lotse.generation.answer import Answer

SYSTEM = """\
Du beantwortest Fragen zum deutschen Arbeitsschutz ausschließlich anhand der nummerierten Quellen.
- Belege jeden Satz mit mindestens einem [n] am Satzende, zum Beispiel [1] oder [1][2].
- Jedes Zitat (quote) ist ein zusammenhängender Ausschnitt aus dem Text von Quelle [n], Zeichen \
für Zeichen kopiert und mindestens drei Wörter lang.
- Kürze ein Zitat nie mit „...“ und setze es nie aus mehreren Stellen zusammen. Brauchst du zwei \
Stellen, gib zwei kurze Zitate an.
- Steht die Antwort nicht in den Quellen, setze answerable=false.
- Prüfe vorher, ob eine Quelle genau das regelt, wonach gefragt wird: dieselbe Personengruppe, \
dieselbe Funktion, denselben Gegenstand. Eine Regel für einen nur ähnlichen Fall ist keine \
Antwort: setze dann answerable=false.
- Übernimm Zahlen, Fristen und Grenzwerte wörtlich.
- Keine Rechtsberatung, keine Vermutungen.
- Antworte in höchstens fünf kurzen Sätzen, ohne Aufzählungen."""

INVALID = "Die Ausgabe war kein vollständiges JSON im vorgegebenen Schema."


def build(question: str, chunks: list[dict]) -> list[dict]:
    sources = "\n\n".join(f"[{n}] {passage(chunk)}" for n, chunk in enumerate(chunks, start=1))
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Quellen:\n\n{sources}\n\nFrage: {question}"},
    ]


def retry(messages: list[dict], reply: Answer | None, reason: str) -> list[dict]:
    """The same request plus what was wrong, because temperature 0 would repeat the mistake."""
    hint = (
        f"Diese Antwort wird nicht angezeigt: {reason} Antworte erneut und halte alle Regeln ein."
    )
    if reply is None:  # nothing to show the model, and two user turns in a row may be refused
        last = messages[-1]
        return [*messages[:-1], {**last, "content": f"{last['content']}\n\n{hint}"}]
    previous = {"role": "assistant", "content": reply.model_dump_json()}
    return [*messages, previous, {"role": "user", "content": hint}]
