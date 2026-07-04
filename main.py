"""
main.py
-------
FastAPI entrypoint. Exposes:

  POST /agent            -> runs the full autonomous pipeline, returns JSON
                             (plan, assumptions, download URL)
  GET  /download/{name}  -> serves the generated .docx file
  GET  /health           -> simple liveness check

Pipeline per request (see README for the architecture diagram):
  1. planner.generate_plan()   -> agent decides its own TODO list
  2. executor.execute_plan()   -> agent works through the list, step by step
  3. reflect.reflect_on_draft()-> agent self-checks its own output (mandatory
                                   engineering improvement: reflection/self-check)
  4. docgen.build_docx()       -> polished Word document written to disk
"""

import os
import uuid
import logging

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from planner import generate_plan
from executor import execute_plan
from reflect import reflect_on_draft
from docgen import build_docx, OUTPUT_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("agent")

app = FastAPI(
    title="Autonomous Document Agent",
    description="Plans, executes, self-checks, and produces a Word document from a natural-language request.",
    version="1.0.0",
)


class AgentRequest(BaseModel):
    request: str = Field(..., min_length=3, description="Natural language request describing the document needed")


class AgentResponse(BaseModel):
    doc_type: str
    title: str
    plan: list
    assumptions: list
    summary: str
    download_url: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/agent", response_model=AgentResponse)
def run_agent(payload: AgentRequest):
    user_request = payload.request.strip()

    # --- Request validation & guardrail: reject empty/garbage input early ---
    if len(user_request) < 3:
        raise HTTPException(status_code=422, detail="Request is too short to plan against.")

    logger.info(f"[REQUEST RECEIVED] {user_request}")

    # 1. Autonomous planning
    plan = generate_plan(user_request)
    logger.info(f"[PLAN CREATED] doc_type={plan['doc_type']} steps={[s['section_heading'] for s in plan['steps']]}")

    # 2. Execute each step in the plan
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
        f"{len(plan['steps'])} sections, based on your request. "
        f"{len(assumptions)} assumption(s)/note(s) were flagged during self-check."
    )

    return AgentResponse(
        doc_type=plan["doc_type"],
        title=plan["title"],
        plan=[{k: v for k, v in s.items() if k != "content"} | {"content_preview": s.get("content", "")[:120]} for s in plan["steps"]],
        assumptions=assumptions,
        summary=summary,
        download_url=f"/download/{filename}",
    )


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
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
