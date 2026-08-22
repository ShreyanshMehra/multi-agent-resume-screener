# Architecture — multi-agent-resume-screener

## Problem

Given one or more resumes (PDF) and a single job description (JD), produce a
ranked shortlist where every candidate's score is **explainable**: which
sections matched, what evidence supported the score, what gaps exist, and what
the candidate could improve.

The earlier single-pass design returned a number with no traceable reasoning.
This version decomposes the work into specialized agents so each stage is
independently inspectable, testable, and improvable.

## Function vs. agent

We use a plain Python **function** when the task is deterministic, and an
**agent** (LLM call) only when the task needs judgment. This keeps cost down and
keeps behavior reproducible wherever possible.

| Task | Implementation |
|------|----------------|
| PDF → text | function (`pypdf`) |
| Text → structured resume | agent |
| JD text → structured requirements | agent |
| Resume ↔ JD section matching | agent |
| Sub-scores → final score | function (weighted sum) |
| Objective resume-quality checks (hygiene) | function (rule-based) |
| Gaps / suggestions / verdict | agent |

## Pipeline

```
┌─────────────────┐     ┌─────────────────┐
│  Resume PDF     │     │ Job Description │
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       ▼
   ┌──────────┐            ┌──────────┐
   │  Parser  │            │   JD     │
   │  Agent   │            │  Parser  │
   └────┬─────┘            └────┬─────┘
        │  StructuredResume     │  StructuredJD
        └───────────┬───────────┘
                    ▼
              ┌──────────┐
              │ Matcher  │  per-section sub-scores + evidence
              │  Agent   │ ◄──────────────┐
              └────┬─────┘                │
                   ▼                      │ (self-correction,
              ┌──────────┐                │  max 1 retry)
              │  Scorer  │  weighted      │
              │ Function │  final score   │
              └────┬─────┘                │
                   ▼                      │
              ┌──────────┐                │
              │  Critic  │  gaps,         │
              │  Agent   │  suggestions,  │
              └────┬─────┘  needs_rescore─┘
                   ▼
              ┌──────────┐
              │  Report  │  score + rationale + gaps + verdict
              └──────────┘
```

> The deterministic **hygiene** function runs between the scorer and the critic;
> it is omitted from the box diagram above for space but feeds the critic and the
> final report.

## Orchestration & personas

A higher layer (`pipeline/screen.py`) serves two personas from one engine:

- **candidate** — one resume, full critic feedback to improve the CV.
- **recruiter** — many resumes ranked against one JD. In `full` critic mode every
  candidate is critiqued; in `fast` mode the resumes are scored first and only
  the **top-K** are critiqued, saving LLM calls at scale.

The JD is parsed once and reused across all resumes (the parser nodes are
idempotent), and resumes are processed concurrently with `asyncio`. The
`enable_critic` flag drives a conditional edge after the hygiene node so fast
mode can skip the inline critic and critique only the top-K ranked candidates.

## Persistence

Each screening run is stored by `storage/runs.py` (standard-library `sqlite3`,
no extra dependency) as a row containing the full JSON result, retrievable via
`GET /runs/{id}` and listed via `GET /runs`. A fresh connection is opened per
operation, keeping the store safe to call from FastAPI's worker threads.

## Design decisions

- **Per-resume pipeline, parallelized.** Each resume runs the full pipeline
  independently (`asyncio.gather`), then a final pass ranks them. This is
  debuggable, retry-friendly, and avoids long-context recency bias from stuffing
  many resumes into one call.
- **Matcher returns sub-scores + evidence, not a final number.** Each section
  (skills, experience, projects, education) is scored independently with quoted
  evidence and short reasoning.
- **Scorer is deterministic.** `final = Σ weightᵢ · sub_scoreᵢ`. Weights are
  hardcoded defaults with an optional per-request override. The math is provable
  and tunable without touching prompts.
- **Critic drives a single self-correction loop.** If the critic's confidence in
  the scoring is low, it sets `needs_rescore` and the graph loops back to the
  matcher once (`retry_count < max_retries`). One conditional edge justifies
  using LangGraph over plain function calls.
- **Configurable critic.** `fast` mode runs the critic only on the top-K ranked
  resumes; `full` mode runs it on every resume.

## Shared state

All nodes read from and write to a single `PipelineState` (Pydantic). It
accumulates structured outputs as agents run, tracks retry bookkeeping, and
maintains a `trace` list — one entry per agent invocation — which becomes the
explainability/audit log persisted to SQLite.

See `src/multi_agent_resume_screener/state.py` (added in Step 4) for the concrete schema.

## Scoring weights (defaults)

| Section | Weight |
|---------|--------|
| Skills | 0.35 |
| Experience | 0.30 |
| Projects | 0.25 |
| Education | 0.10 |

Overridable per request via the API `config`.
