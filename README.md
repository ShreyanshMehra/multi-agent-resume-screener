---
title: multi-agent-resume-screener
emoji: 📄
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
license: mit
---

# multi-agent-resume-screener

**Live demo:** https://bit.ly/multi_agent-resume-screener

> Multi-agent resume screening pipeline built with **LangGraph** and **Gemini**, served via **FastAPI**.

multi-agent-resume-screener screens resumes against a job description using a pipeline of
specialized agents (a **parser**, a **JD parser**, a **matcher**, a
deterministic **scorer**, a deterministic **hygiene** checker, and a **critic**),
with a self-correction loop and a full audit trail for explainability.

It serves two personas from one engine:

- **Candidate mode**: "How well does my CV fit this job, and how do I improve it?"
  Feedback comes in two tracks: **build over time** (skills/experience to develop
  for this role) and **fix right now** (immediate, deterministic CV edits).
- **Recruiter mode**: "Rank these resumes for this job, with reasons."
  Every candidate carries the matcher's per-section reasoning + evidence as the
  explicit reason for the ranking.

## Why a multi-agent design?

A single LLM call that "scores a resume" is a black box: you can't tell *why* a
candidate ranked where they did, and you can't improve one stage without
risking the others. multi-agent-resume-screener splits the job into focused stages, using a
plain function where the task is deterministic and an LLM agent only where the
task needs judgment:

| Stage | Type | Responsibility |
|-------|------|----------------|
| Parser | agent | Resume PDF text → structured fields (skills, projects, experience) |
| JD Parser | agent | Job description → structured requirements |
| Matcher | agent | Per-section sub-scores + quoted evidence, vs **this** JD |
| Scorer | function | Deterministic weighted score from sub-scores |
| Hygiene | function | Deterministic "fix right now" rules: links, quantified bullets, weak verbs, over-long bullets, first-person pronouns, buzzwords, generic names… |
| Critic | agent | JD gaps, "build over time" skill-building advice, verdict, and a confidence check |

If the critic is not confident the score is well-supported, it loops back to the
matcher once for a re-evaluation (capped to avoid infinite loops).

```
                resume.pdf          job description
                    │                     │
                    ▼                     ▼
                 parser ─────────────► jd_parser
                    └──────────┬──────────┘
                               ▼
                    ┌────► matcher ──► scorer ──► hygiene ──► critic ─┐
                    │                                                  │
                    └──────── self-correction (≤1 retry) ◄────────────┘
                                          │
                                          ▼
                          ranked, explainable results
```

## Candidate feedback: two tracks

When you screen your own CV (candidate mode), the feedback is deliberately split
into two buckets, because the two kinds of improvement have very different time
horizons:

- **Build over time, for this role.** JD requirements you don't clearly meet
  (`gaps`) plus the critic's forward-looking, skill-building advice
  (`suggestions`): technologies to learn and the kind of experience/projects to
  build next. These are things you *grow into*, not edits you make today.
- **Fix right now, quick CV edits.** Objective, rule-based issues from the
  deterministic hygiene checker (`hygiene_issues`): add impact numbers, lead with
  strong action verbs, condense over-long bullets, drop first-person pronouns and
  buzzwords, add missing links. These are instant, no-LLM, and reproducible, and
  the UI shows them in a separate section below.

Because the "fix right now" bucket is deterministic, it's cheap and defensible
("we combine objective rule-based checks with LLM judgment, not one big prompt")
and adds no latency.

## Status

✅ Core engine + HTTP API complete and tested (78 tests, all live-verified
against Gemini). See `docs/architecture.md` for the full design.

## Tech stack

- Python 3.12
- FastAPI + Uvicorn
- LangGraph + LangChain (agent orchestration)
- Google Gemini (free tier), provider-abstracted so Groq is swappable
- pypdf (PDF text extraction)
- SQLite (run history / explainability)

## Getting started

```bash
# 1. Clone
git clone https://github.com/ShreyanshMehra/multi-agent-resume-screener.git
cd multi-agent-resume-screener

# 2. Create & activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\Activate.ps1
# macOS/Linux
# source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
copy .env.example .env        # Windows  (use `cp` on macOS/Linux)
# then edit .env and add your free Gemini key from
# https://aistudio.google.com/apikey
```

## Usage

### Run the API

```bash
uvicorn multi_agent_resume_screener.api.main:app --reload
# open http://127.0.0.1:8000/docs for interactive Swagger UI
```

Screen resumes against a job description:

```bash
# Recruiter mode: rank multiple resumes
curl -X POST http://127.0.0.1:8000/screen \
  -F "jd=Backend engineer. Required: Python, Go, PostgreSQL, 3+ years." \
  -F "mode=recruiter" \
  -F "resumes=@alice.pdf" \
  -F "resumes=@bob.pdf"

# Candidate mode: feedback to improve one CV
curl -X POST http://127.0.0.1:8000/screen \
  -F "jd=Backend engineer..." \
  -F "mode=candidate" \
  -F "resumes=@my_cv.pdf"

# Retrieve a stored run later
curl http://127.0.0.1:8000/runs/<run_id>
```

### Command-line scripts

```bash
python scripts/check_llm.py                       # verify your LLM key works
python scripts/parse_resume.py resume.pdf         # PDF → structured resume
python scripts/screen.py resume.pdf job.txt       # full pipeline + agent trace
```

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness check |
| POST | `/screen` | Screen resume PDF(s) against a JD; returns ranked results |
| GET | `/runs/{id}` | Retrieve a stored screening run |
| GET | `/runs` | List recent runs |

`/screen` form fields: `jd` (text), `mode` (`candidate`\|`recruiter`),
`critic_mode` (`fast`\|`full`), `critic_top_k` (int), `resumes` (PDF file(s)).

## Testing

```bash
pip install -e ".[dev]"
pytest -q                       # 78 tests, fully offline (fake LLM)
ruff check src tests scripts    # lint
```

## Project layout

```
src/multi_agent_resume_screener/
├── state.py          # Shared Pydantic state + public result models
├── settings.py       # Typed config from .env
├── pdf.py            # Deterministic PDF → text
├── llm/              # Provider-abstracted LLM client (Gemini/Groq)
├── agents/           # parser, jd_parser, matcher, critic
├── pipeline/         # scorer, hygiene (deterministic), graph, screen
├── api/              # FastAPI app
└── storage/          # SQLite run persistence
```

## Deployment

The app ships with a `Dockerfile` (used by Hugging Face Spaces) and a Render
blueprint (`render.yaml`).

```bash
# Build and run locally with Docker
docker build -t multi-agent-resume-screener .
docker run -p 8000:8000 -e GOOGLE_API_KEY=your-key multi-agent-resume-screener
```

**Hugging Face Spaces (recommended, free):** the `README.md` front-matter
configures a Docker Space (`app_port: 8000`). Create a new **Docker** Space,
push this repo to it, and add `GOOGLE_API_KEY` as a Space **secret**. The Space
sleeps only after 48h idle, so a portfolio link is usually warm.

**Render (free tier):** create a new Blueprint from the repo and set
`GOOGLE_API_KEY` in the dashboard. Note: free services spin down after 15 min
idle. Either platform keeps the SQLite file in `/tmp` (ephemeral, so run history
resets on redeploy, which is fine for a demo).

## License

[MIT](LICENSE) © 2026 Shreyansh Dutt Mehra
