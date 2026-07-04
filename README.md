# Autonomous Document Agent

A FastAPI service that takes a natural-language request, **plans its own TODO
list**, **executes each step**, **self-checks its own draft**, and returns a
polished **.docx** document.

## Quick start

```bash
pip install -r requirements.txt

# Get a free key at https://console.groq.com/keys, then:
export GROQ_API_KEY="your_key_here"

# Optional - defaults to llama-3.3-70b-versatile
export GROQ_MODEL="llama-3.3-70b-versatile"

python3 -m uvicorn main:app --reload
```

Server runs at `http://localhost:8000`. Interactive docs at
`http://localhost:8000/docs`.

> **No API key?** The agent still runs — `llm_client.py` falls back to a
> deterministic mock mode so you can test the full pipeline (plan → execute →
> reflect → docx) offline. Useful for development; use a real key for the
> demo video so the content is genuinely LLM-generated.

## Try it

```bash
curl -X POST http://localhost:8000/agent \
  -H "Content-Type: application/json" \
  -d '{"request": "Create a project plan for launching a mobile app in 3 months"}'
```

Response includes the generated plan (with per-step status), the
assumptions/notes from self-check, and a `download_url` for the `.docx`:

```bash
curl -O http://localhost:8000/download/<filename>.docx
```

Or run `python3 test_pipeline.py` to exercise both required test cases
directly (no server needed) and inspect the console output showing the
agent's plan being created and executed step by step.

## Architecture

```
Request  ->  planner.py   ->  executor.py   ->  reflect.py    ->  docgen.py  ->  .docx
             (LLM call #1)    (LLM call per      (LLM call:        (python-docx)
             decides doc      step, generates     compares draft
             type + section   that section's      vs original
             list = TODO)     content)            request, lists
                                                   assumptions/gaps)
```

- **`planner.py`** — the agent decides, per request, what document type fits
  and what sections are needed. Nothing is hardcoded per-request; ambiguous
  requests just get more assumption-laden plans, not a different code path.
- **`executor.py`** — walks the plan in order, one LLM call per section,
  flips each step's status `pending -> in_progress -> done` (or
  `failed_recovered` if a single section's generation throws, so one bad
  call doesn't kill the whole document).
- **`reflect.py`** — **the mandatory engineering improvement**
  (Reflection/Self-check). After drafting, the agent re-reads its own output
  next to the original request and explicitly lists what it had to assume
  or what's still missing. This is what makes the "complex/ambiguous"
  test case work honestly instead of silently guessing.
- **`docgen.py`** — python-docx build: real `Heading` styles (so the
  document has a proper outline, not just bold text), title block, bullet
  lists, and a final "Assumptions & Notes" section sourced directly from
  the reflection step.
- **`llm_client.py`** — single choke point for all LLM calls; owns
  retry-with-backoff and the offline mock fallback, so that logic isn't
  duplicated in three files.
- **`main.py`** — FastAPI app, `POST /agent` + `GET /download/{file}`, plus
  basic request validation (rejects empty/garbage input with a 422 before
  any LLM call is made).

## The two required test cases

1. **Standard**: *"Create a project plan for launching a mobile app in 3
   months"* — clean, linear plan, no major assumptions needed.
2. **Complex/ambiguous**: *"We need something for tomorrow's client meeting
   about our new product, not sure exactly what format, and the budget
   numbers aren't finalized yet."* — agent has to pick a document type
   itself (e.g. proposal), and the self-check step will explicitly flag the
   missing budget and format as assumptions rather than failing.

Both are already wired up in `test_pipeline.py`.

## Talking points for the video

**Debugging insight (real, not invented):** LLMs don't reliably return pure
JSON for the plan — they wrap it in ```json fences or add a sentence before
it. `planner._extract_json()` strips fences and falls back to regex-matching
the outermost `{...}` block; if that *still* fails, one retry is issued with
a stricter prompt, and if that fails too, `FALLBACK_PLAN` is used so the
request completes instead of 500ing. That's your error-handling story.

**Tradeoff — Autonomous planning vs. deterministic workflow:** the plan step
lets the LLM freely decide section count/type per request, which is exactly
what makes the ambiguous test case work — but it also means plan shape is
non-deterministic across runs of the *same* request, which would matter for
things like automated regression tests. A deterministic template per
doc_type would be more predictable and cheaper (no planning LLM call) but
couldn't gracefully handle a genuinely novel request type.
