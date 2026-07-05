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

Server runs at `http://localhost:8000`.

- **Browser UI**: open `http://localhost:8000/` — type a request (or click
  one of the two example buttons), hit "Run agent", and watch the plan
  execute step by step with a live-feeling console log, then get the
  assumptions/notes and a download button for the `.docx`. This is the
  easiest way to demo the two required test cases on camera.
- **Interactive API docs**: `http://localhost:8000/docs`

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

## Meeting recording → Minutes of Meeting

Beyond typed requests, you can upload a meeting recording (video or audio)
and the agent will transcribe it and generate Minutes of Meeting through
the exact same pipeline.

- **Endpoint**: `POST /agent/from-recording` (multipart file upload)
- **UI**: click the "Upload meeting recording" tab, drop in a file, hit Run
- **How it works** (`transcribe.py`):
  1. Audio is extracted from the video using `imageio-ffmpeg` — a pip
     package that bundles a pre-compiled ffmpeg binary, so there's no
     separate system install needed on Windows or on a deployment server
  2. Long recordings are automatically **split into ~10-minute chunks**
     (with a small overlap so words at chunk boundaries aren't lost),
     since hosted transcription APIs cap file size — this means it scales
     to long meetings, not just short clips
  3. Each chunk is transcribed via Groq's hosted **Whisper large-v3**
     model, then stitched back into one transcript
  4. The transcript becomes the "request" text and is fed into the exact
     same `generate_plan -> execute_plan -> reflect_on_draft -> build_docx`
     pipeline used for typed requests — no separate code path to maintain
- **Scope decision**: no speaker diarization (no "Speaker 1 / Speaker 2"
  labels) — output is a clean transcript-based summary. Real diarization
  (e.g. `pyannote.audio`) is possible but adds heavy ML dependencies and
  meaningfully slower processing; left out to keep the service lightweight
  and fast to deploy.

## Tool orchestration

`executor.py` doesn't just make plain text-generation calls — it gives the
LLM a real callable tool via function/tool calling
(`tools.py` + `llm_client.call_llm_with_tools`):

- **`lookup_benchmark_data(topic)`** — a mock benchmark-data lookup
  (typical budget range, timeline, team size for a given project type).
  Stands in for what would be a real internal database or pricing API call.

The model is given the tool's schema and description, and **decides for
itself, per section, whether it needs it** — e.g. it calls the tool when
writing a "Budget" or "Timeline" section, but not for an "Introduction"
section. This is genuine tool orchestration (the agent choosing among
available actions), not just another LLM text call. Which sections
actually invoked the tool is tracked per-step (`tools_used`), surfaced in
the API response, shown live in the UI's execution log, and noted in the
generated `.docx`.

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
  call doesn't kill the whole document). Each call is made via
  `call_llm_with_tools`, giving the model the option to invoke
  `lookup_benchmark_data` when a section needs numbers (see "Tool
  orchestration" below).
- **`tools.py`** — the tool definition (JSON schema for function calling)
  and its mock implementation.
- **`transcribe.py`** — turns an uploaded meeting recording into text
  (audio extraction, chunking, transcription) so it can feed into the
  same pipeline as a typed request.
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
