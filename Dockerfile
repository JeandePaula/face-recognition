FROM ghcr.io/astral-sh/uv:0.12.23 AS uv
FROM python:3.14.8-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

COPY --from=uv /uv /usr/local/bin/uv
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --create-home app

WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --locked --no-dev --no-install-project
COPY src ./src
COPY licenses ./licenses
COPY THIRD_PARTY.md config.example.toml ./
RUN uv sync --locked --no-dev --no-editable \
    && mkdir -p models known_faces detected_faces_output \
    && chown -R app:app /app

USER app
ENTRYPOINT ["face-camera"]
CMD ["run", "--headless"]
