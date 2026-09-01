# Inference Service — anomaly scoring over canonical telemetry features.
#
# The model artifact is built here, not committed and not trained at runtime:
#
#     source code -> docker build -> ml/train.py -> model.joblib in the image
#                 -> container start -> artifact loaded once -> serve predictions
#
# Training is deterministic (fixed seeds, explicit hyperparameters), so the same
# source produces the same model. This keeps the repository free of a 20 MB
# generated binary while letting a fresh clone build and run with no manual step.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# `ml/` is needed at build time to train, and `app/` both to train (it owns the
# feature vocabulary and artifact format) and to serve.
COPY app ./app
COPY ml ./ml

# Train once, at build time. The service never trains — not at startup, and
# certainly not in a request handler.
RUN python ml/train.py

RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 8001

# Reports whether a model is actually loaded, which is what makes this instance
# able to serve. A process that started but failed to load its artifact is alive
# but useless, and must not be reported healthy.
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=5 \
    CMD python -c "import json,sys,urllib.request; sys.exit(0 if json.load(urllib.request.urlopen('http://127.0.0.1:8001/health', timeout=4))['model_loaded'] else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
