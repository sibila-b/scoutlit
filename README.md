# ScoutLit — Academic Literature Assistant

An AI system that autonomously retrieves papers from arXiv and Semantic Scholar, classifies them as seminal, foundational, or recent, synthesises a structured literature review with inline citations, and surfaces research gaps.

## Task board

All tasks, sprints, and delivered features are tracked on the project Jira board:

**[ScoutLit Agile Board (Jira)](https://sibila-s-workspace.atlassian.net/jira/software/projects/SCALA/boards/1/backlog)**

## Local setup

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (includes Docker Compose)
- Git

### Steps

```bash
# 1. Clone
git clone https://github.com/sibila-b/scoutlit.git
cd scoutlit

# 2. Configure environment
cp .env.example .env
# Open .env and fill in your API keys (ANTHROPIC_API_KEY is required)

# 3. Start all services
docker compose up
```

All three services start automatically:

| Service | URL | Notes |
|---|---|---|
| Frontend | http://localhost:3000 | React UI |
| Backend API | http://localhost:8000/health | Returns `{"status":"ok"}` — no route exists at `/` |
| Backend docs | http://localhost:8000/docs | Swagger UI listing all endpoints |
| ChromaDB | http://localhost:8001 | Vector store — no browser UI |

> The backend waits for ChromaDB to pass its health check before starting. First run pulls images and builds containers (~60 s on a fast connection).

### Stopping

```bash
docker compose down          # stop containers
docker compose down -v       # stop and delete ChromaDB volume
```

## Quickstart (CLI)

```bash
cp .env.example .env
# fill in your API keys

pip install -e ".[dev]"
scoutlit "attention mechanisms in transformer architectures"
```

## Project structure

```
backend/           — FastAPI app (env-var validation, ChromaDB connection)
frontend/          — Vite + React UI (served by nginx in Docker)
src/
  retrieval/       — arXiv + Semantic Scholar clients
  classification/  — Claude-powered paper categorisation
  synthesis/       — structured review generation with streaming
  gaps/            — research gap detection
tests/             — unit tests (mocked API calls)
.github/workflows/ — CI (lint + test matrix) and release pipeline
docker-compose.yml — three-service local stack
```

## Environment variables

| Variable | Required | Description | Example |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | Claude API key | `sk-ant-…` |
| `SEMANTIC_SCHOLAR_API_KEY` | No | Higher rate limits on Semantic Scholar | `your_key_here` |
| `CHROMA_HOST` | No | ChromaDB hostname (default: `chromadb` in Docker) | `chromadb` |
| `CHROMA_PORT` | No | ChromaDB port (default: `8000`) | `8000` |

## Development

```bash
pytest tests/ -v          # run tests
ruff check src tests       # lint
ruff format src tests      # format
```
