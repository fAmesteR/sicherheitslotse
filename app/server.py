"""The demo as one web service, as the Docker image runs it: the page at /, and /ask, /health and
/metrics as JSON. Every query writes one log line with its latency and cost.

uv run --env-file .env uvicorn server:api --app-dir app --port 7860
"""

import logging

import gradio as gr

from app import THEME, answers, demo  # app.py loads the index, the key and both models
from lotse.service import create_app

logging.basicConfig(level=logging.INFO, format="%(message)s")  # the query log goes to the output

api = gr.mount_gradio_app(create_app(answers), demo, path="/", theme=THEME, footer_links=[])
