"""
main.py
-------
FastAPI entrypoint. Exposes:

  POST /agent                 -> runs the full pipeline from a typed text request
  POST /agent/from-recording  -> runs the full pipeline from an uploaded meeting
                                  video/audio file (transcribed first)
  GET  /download/{name}       -> serves the generated .docx file
  GET  /health                -> simple liveness check

Pipeline per request (see README for the architecture diagram):
  0. transcribe.transcribe_recording() -> (recording path only) turns the
                                            uploaded video/audio into text,
                                            which then becomes the "request"
  1. planner.generate_plan()   -> agent decides its own TODO list
  2. executor.execute_plan()   -> agent works through the list, step by step,
                                   with tool-calling for sections that need data
  3. reflect.reflect_on_draft()-> agent self-checks its own output (mandatory
                                   engineering improvement: reflection/self-check)
  4. docgen.build_docx()       -> polished Word document written to disk
"""

import os
import uuid
import logging

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from planner import generate_plan
from executor import execute_plan
from reflect import reflect_on_draft
from docgen import build_docx, OUTPUT_DIR
from transcribe import transcribe_recording

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("agent")

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

app = FastAPI(
    title="Autonomous Document Agent",
    description="Plans, executes, self-checks, and produces a Word document from a natural-language request.",
    version="1.0.0",
)

# Serve the browser UI. Mounted at /ui so it doesn't shadow the JSON routes below;
# "/" redirects to it for convenience.
app.mount("/ui", StaticFiles(directory=STATIC_DIR, html=True), name="ui")


@app.get("/")
def root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


class AgentRequest(BaseModel):
    request: str = Field(..., min_length=3, description="Natural language request describing the document needed")


class AgentResponse(BaseModel):
    doc_type: str
    title: str
    plan: list
    assumptions: list
    summary: str
    download_url: str
    source: str = "text"  # "text" or "recording", so the UI can show provenance


@app.get("/health")
def health():
    return {"status": "ok"}


def _run_pipeline(user_request: str, source: str = "text", source_label: str = "") -> AgentResponse:
    """
    Shared core: everything from here down is identical whether the request
    came from typed text or from a transcribed recording. Keeping this in
    one place means /agent and /agent/from-recording can never drift apart.
    """
    logger.info(f"[REQUEST RECEIVED] source={source} len={len(user_request)} chars")

    # 1. Autonomous planning
    plan = generate_plan(user_request)
    logger.info(f"[PLAN CREATED] doc_type={plan['doc_type']} steps={[s['section_heading'] for s in plan['steps']]}")

    # 2. Execute each step in the plan (with tool-calling where needed)
    plan = execute_plan(user_request, plan)

    # 3. Reflection / self-check (mandatory improvement)
    assumptions = reflect_on_draft(user_request, plan)

    # 4. Generate the Word document
    filename = f"{uuid.uuid4().hex[:8]}_{plan['doc_type'].replace(' ', '_')}.docx"
    try:
        filepath = build_docx(plan, assumptions, filename)
    except Exception as e:
        logger.error(f"Document generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Document generation failed: {e}")

    logger.info(f"[DOCUMENT READY] {filepath}")

    summary = (
        f"Generated a {plan['doc_type']} titled '{plan['title']}' with "
        f"{len(plan['steps'])} sections"
        + (f" from {source_label}" if source_label else ", based on your request")
        + f". {len(assumptions)} assumption(s)/note(s) were flagged during self-check."
    )

    return AgentResponse(
        doc_type=plan["doc_type"],
        title=plan["title"],
        plan=[{k: v for k, v in s.items() if k != "content"} | {
            "content_preview": s.get("content", "")[:120],
            "tools_used": s.get("tools_used", []),
        } for s in plan["steps"]],
        assumptions=assumptions,
        summary=summary,
        download_url=f"/download/{filename}",
        source=source,
    )


@app.post("/agent", response_model=AgentResponse)
def run_agent(payload: AgentRequest):
    user_request = payload.request.strip()

    # --- Request validation & guardrail: reject empty/garbage input early ---
    if len(user_request) < 3:
        raise HTTPException(status_code=422, detail="Request is too short to plan against.")

    return _run_pipeline(user_request, source="text")


MAX_UPLOAD_BYTES = 300 * 1024 * 1024  # 300 MB guardrail on recording uploads


@app.post("/agent/from-recording", response_model=AgentResponse)
async def run_agent_from_recording(file: UploadFile = File(...)):
    file_bytes = await file.read()

    # --- Guardrails: reject empty or oversized uploads before doing any work ---
    if len(file_bytes) == 0:
        raise HTTPException(status_code=422, detail="Uploaded file is empty.")
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 300MB for this demo).")

    logger.info(f"[RECORDING RECEIVED] {file.filename} ({len(file_bytes) / 1_000_000:.1f} MB)")

    try:
        transcript = transcribe_recording(file_bytes, file.filename)
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")

    if len(transcript.strip()) < 20:
        raise HTTPException(
            status_code=422,
            detail="Couldn't extract meaningful speech from this recording.",
        )

    # Cap transcript length fed to the LLM stages - keeps prompt size sane
    # for very long meetings without needing a separate summarization pass.
    MAX_TRANSCRIPT_CHARS = 15000
    transcript_for_agent = transcript[:MAX_TRANSCRIPT_CHARS]

    user_request = (
        "Generate professional meeting minutes (Minutes of Meeting) based on the "
        "following meeting transcript. Identify the key discussion points, any "
        "decisions made, and action items with owners where mentioned.\n\n"
        f"TRANSCRIPT:\n{transcript_for_agent}"
    )

    return _run_pipeline(user_request, source="recording", source_label=f"the uploaded recording '{file.filename}'")


@app.get("/download/{filename}")
def download(filename: str):
    filepath = os.path.join(OUTPUT_DIR, filename)
    if not os.path.isfile(filepath):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(
        filepath,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        filename=filename,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
