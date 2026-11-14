"""The demo: a question box, the checked answer and its sources.

uv run --env-file .env python app/app.py
"""

import logging
import sys
from pathlib import Path

import gradio as gr
import spaces  # ZeroGPU needs this imported before torch sets up CUDA

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # a Space installs no package

from lotse import pipeline
from lotse.demo import MAX_QUESTION_CHARS, Demo, footer
from lotse.generation import llm
from lotse.retrieval import dense, rerank
from lotse.retrieval.index import fetch_index

# One per question type: everyday, fact, duties, and one the sources don't answer.
EXAMPLES = [
    "Muss unser Pausenraum ein Fenster haben?",
    (
        "Wie lange müssen wir die Liste der Leute aufheben, die mit krebserzeugenden Stoffen "
        "gearbeitet haben?"
    ),
    "Muss auch ein Kleinbetrieb die Gefährdungsbeurteilung für Gefahrstoffe aufschreiben?",
    "Wie viele Urlaubstage stehen mir im Jahr zu?",
]
INTRO = (
    "# Sicherheitslotse\n"
    "Fragen zum deutschen Arbeitsschutzrecht. Jede Antwort nennt ihre Quelle mit wörtlichem "
    "Zitat. Ein Programm prüft jedes Zitat und jede Zahl gegen den Gesetzestext, bevor die "
    "Antwort angezeigt wird."
)

# Load everything at startup, so a missing key or a stale index stops the app here and not on a
# visitor's first question. On ZeroGPU the models also have to be on the GPU before any request.
fetch_index()
pipeline.index()
llm.client()
dense.load()
rerank.load()


@spaces.GPU(duration=20)
def retrieve(question: str) -> pipeline.Ranked:
    """Embedding and reranking, the only steps that need the GPU. A visitor's daily GPU quota pays
    for this function alone: the API call and the check run outside it."""
    return pipeline.retrieve(pipeline.index(), question)


answers = Demo(lambda question: pipeline.answer(question, retrieve(question)))


def respond(question: str, request: gr.Request) -> str:
    # Known limit: behind a proxy all visitors may share this address. Check it on the Space.
    return answers.respond(question, request.client.host)


with gr.Blocks(title="Sicherheitslotse", analytics_enabled=False) as demo:
    gr.Markdown(INTRO)
    question = gr.Textbox(
        label="Ihre Frage",
        placeholder="Zum Beispiel: Wie lange darf ich am Tag arbeiten?",
        max_length=MAX_QUESTION_CHARS,
    )
    button = gr.Button("Fragen", variant="primary")
    answer = gr.HTML()
    gr.Examples(
        EXAMPLES,
        question,
        answer,
        respond,
        cache_examples=False,
        run_on_click=True,
        label="Beispielfragen",
    )
    gr.Markdown(footer())
    gr.on([question.submit, button.click], respond, question, answer)
    gr.api(answers.summary, api_name="metrics")  # latency and cost so far, as JSON

demo.queue(default_concurrency_limit=1, max_size=20)  # one question at a time, twenty waiting

# System fonts, so a visitor's browser loads nothing from Google.
THEME = gr.themes.Default(font=["system-ui", "sans-serif"])

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")  # shows the query log
    demo.launch(theme=THEME, footer_links=[])
