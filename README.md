# ScoutLit — Academic Literature Assistant

An AI system that autonomously retrieves papers from arXiv and Semantic Scholar, classifies them as seminal, foundational, or recent, synthesises a structured literature review with inline citations, and surfaces research gaps.

## Task board

All tasks, sprints, and delivered features are tracked on the project Jira board:

**[ScoutLit Agile Board (Jira)](https://sibila-s-workspace.atlassian.net/jira/software/projects/SCALA/boards/1/backlog)**

## Quickstart

```bash
cp .env.example .env
# fill in your API keys

pip install -e ".[dev]"
scoutlit "attention mechanisms in transformer architectures"
```

## Project structure

```
src/
  retrieval/       — arXiv + Semantic Scholar clients
  classification/  — Claude-powered paper categorisation
  synthesis/       — structured review generation with streaming
  gaps/            — research gap detection
tests/             — unit tests (mocked API calls)
.github/workflows/ — CI (lint + test matrix) and release pipeline
```

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | Claude API key |
| `SEMANTIC_SCHOLAR_API_KEY` | No | Higher rate limits |

## Development

```bash
pytest tests/ -v          # run tests
ruff check src tests       # lint
ruff format src tests      # format
```
