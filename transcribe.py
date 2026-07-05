"""
transcribe.py
-------------
Turns an uploaded meeting recording (video or audio, any length) into plain
text, so it can be fed into the same planner -> executor -> reflect ->
docgen pipeline used for typed requests.

Design notes:
- Uses `imageio-ffmpeg`, which ships a pre-compiled ffmpeg binary as part of
  the pip package. No system-level ffmpeg install needed, on Windows or on
  a Linux server - this is what makes it safe to deploy without extra setup.
- Groq's Whisper endpoint (like most hosted transcription APIs) has a file
  size limit, so long meetings are split into fixed-length chunks with a
  small overlap (so a sentence spoken right at a chunk boundary isn't cut
  in half and lost), each chunk is transcribed independently, and the
  results are stitched back together in order.
- No speaker diarization (by design/scope decision) - output is a clean
  transcript, not per-speaker labeled dialogue.
"""

import os
import math
import subprocess
import tempfile
import logging
import shutil

import imageio_ffmpeg

logger = logging.getLogger("agent")

FFMPEG_BIN = imageio_ffmpeg.get_ffmpeg_exe()

CHUNK_SECONDS = 600      # 10 minutes per chunk
OVERLAP_SECONDS = 5      # small overlap so words at the boundary aren't lost
AUDIO_EXTS = (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac")


def _run_ffmpeg(args: list):
    cmd = [FFMPEG_BIN, "-y"] + args
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr[-800:]}")


def _get_duration_seconds(path: str) -> float:
    """Uses ffmpeg itself (via stderr parsing) to get media duration - avoids
    needing a separate ffprobe dependency."""
    cmd = [FFMPEG_BIN, "-i", path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    for line in result.stderr.splitlines():
        line = line.strip()
        if line.startswith("Duration:"):
            time_str = line.split(",")[0].replace("Duration:", "").strip()
            h, m, s = time_str.split(":")
            return int(h) * 3600 + int(m) * 60 + float(s)
    raise RuntimeError("Could not determine media duration")


def extract_audio(input_path: str, output_path: str):
    """Extracts mono 16kHz wav audio - small file size, ideal for speech models."""
    _run_ffmpeg(["-i", input_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", output_path])


def split_into_chunks(audio_path: str, work_dir: str) -> list:
    """
    Splits a wav file into CHUNK_SECONDS-long pieces with OVERLAP_SECONDS
    overlap between consecutive chunks. Returns a list of chunk file paths
    in order.
    """
    duration = _get_duration_seconds(audio_path)
    if duration <= CHUNK_SECONDS:
        return [audio_path]  # short enough, no splitting needed

    n_chunks = math.ceil(duration / (CHUNK_SECONDS - OVERLAP_SECONDS))
    chunk_paths = []

    for i in range(n_chunks):
        start = max(0, i * (CHUNK_SECONDS - OVERLAP_SECONDS))
        if start >= duration:
            break
        chunk_path = os.path.join(work_dir, f"chunk_{i:03d}.wav")
        _run_ffmpeg([
            "-i", audio_path,
            "-ss", str(start), "-t", str(CHUNK_SECONDS),
            "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            chunk_path,
        ])
        chunk_paths.append(chunk_path)

    logger.info(f"[TRANSCRIBE] Split {duration:.0f}s audio into {len(chunk_paths)} chunk(s)")
    return chunk_paths


def _transcribe_one_chunk(chunk_path: str) -> str:
    """Sends a single audio chunk to Groq's hosted Whisper model."""
    from llm_client import _client  # reuse the already-configured Groq client

    if _client is None:
        logger.warning("[TRANSCRIBE] No GROQ_API_KEY configured - returning mock transcript for this chunk")
        return (
            "[MOCK TRANSCRIPT CHUNK - no GROQ_API_KEY configured] This would normally "
            "contain real transcribed speech from this segment of the recording."
        )

    with open(chunk_path, "rb") as f:
        result = _client.audio.transcriptions.create(
            file=(os.path.basename(chunk_path), f.read()),
            model="whisper-large-v3",
        )
    return result.text.strip()


def transcribe_recording(file_bytes: bytes, original_filename: str) -> str:
    """
    End-to-end entry point: raw uploaded file bytes (video or audio, any
    length) in, full plain-text transcript out. Handles temp file cleanup
    internally regardless of success/failure.
    """
    work_dir = tempfile.mkdtemp(prefix="agent_transcribe_")
    suffix = os.path.splitext(original_filename)[1].lower() or ".mp4"
    input_path = os.path.join(work_dir, f"input{suffix}")

    try:
        with open(input_path, "wb") as f:
            f.write(file_bytes)

        # Extract audio unless the upload was already a plain audio file
        if suffix in AUDIO_EXTS:
            audio_path = input_path
        else:
            audio_path = os.path.join(work_dir, "audio.wav")
            logger.info(f"[TRANSCRIBE] Extracting audio from {original_filename}")
            extract_audio(input_path, audio_path)

        chunk_paths = split_into_chunks(audio_path, work_dir)

        transcripts = []
        for idx, chunk_path in enumerate(chunk_paths, 1):
            logger.info(f"[TRANSCRIBE] Transcribing chunk {idx}/{len(chunk_paths)}")
            transcripts.append(_transcribe_one_chunk(chunk_path))

        full_transcript = _stitch_transcripts(transcripts)
        logger.info(f"[TRANSCRIBE] Done - {len(full_transcript)} characters total")
        return full_transcript

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def _stitch_transcripts(chunks: list) -> str:
    """
    Joins per-chunk transcripts into one text. Because consecutive chunks
    overlap by OVERLAP_SECONDS of audio, there's a small amount of
    duplicated words at each boundary - for a first working version we
    join with a clear separator rather than attempting fuzzy de-duplication,
    which keeps behavior predictable. The LLM stages downstream are
    tolerant of this (they're summarizing/planning from the content, not
    doing exact-match processing), so a few duplicated words at chunk
    boundaries don't meaningfully affect output quality.
    """
    return "\n\n".join(chunks)
