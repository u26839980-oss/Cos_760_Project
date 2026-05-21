"""
asr_module.py
=============
Handles transcription via three ASR engines:
  1. OpenAI Whisper large-v3
  2. Lelapa AI ASR API
  3. wav2vec-2.0 / AfroXLSR checkpoint

Each function returns a dict:
  {"engine": str, "file": str, "transcript": str, "segments": list}
"""

import os
import json
import time
import logging
import requests
import numpy as np
import torch
import librosa
import soundfile as sf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# 0.  AUDIO UTILITY
# ─────────────────────────────────────────────────────────────

def load_audio(file_path: str, target_sr: int = 16_000) -> tuple[np.ndarray, int]:
    """
    Load and resample audio to mono 16kHz (required by all three ASR engines).

    Parameters
    ----------
    file_path : str
        Path to any audio format supported by librosa (.mp3, .wav, .ogg …)
    target_sr : int
        Target sample rate (default 16 000 Hz)

    Returns
    -------
    audio : np.ndarray   float32 mono waveform
    sr    : int          sample rate (always target_sr)
    """
    logger.info(f"Loading audio: {file_path}")
    audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
    logger.info(f"  Duration: {len(audio)/sr:.1f}s  |  SR: {sr} Hz")
    return audio, sr


def save_wav(audio: np.ndarray, sr: int, out_path: str) -> str:
    """Save numpy array as 16-bit WAV (required by Lelapa API and wav2vec)."""
    sf.write(out_path, audio, sr, subtype="PCM_16")
    logger.info(f"  Saved WAV → {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────
# 1.  WHISPER
# ─────────────────────────────────────────────────────────────

def transcribe_whisper(audio_path: str, model_size: str = "large-v3",
                       language: str = "tn") -> dict:
    """
    Transcribe with OpenAI Whisper.

    Parameters
    ----------
    audio_path  : str   Path to audio file
    model_size  : str   "large-v3" (recommended), "medium", "small"
    language    : str   ISO-639-1 code. "tn" = Setswana; pass None for auto-detect.

    Returns
    -------
    dict with keys: engine, file, transcript, segments, language_detected
    """
    import whisper  # lazy import so other engines work if whisper not installed

    logger.info(f"[Whisper] Loading model '{model_size}' …")
    model = whisper.load_model(model_size)

    logger.info(f"[Whisper] Transcribing {os.path.basename(audio_path)} …")
    result = model.transcribe(
        audio_path,
        language=language,
        verbose=False,
        word_timestamps=False,
    )

    segments = [
        {"start": s["start"], "end": s["end"], "text": s["text"].strip()}
        for s in result.get("segments", [])
    ]

    return {
        "engine": "whisper",
        "file": audio_path,
        "transcript": result["text"].strip(),
        "segments": segments,
        "language_detected": result.get("language", language),
    }


# ─────────────────────────────────────────────────────────────
# 2.  LELAPA AI API
# ─────────────────────────────────────────────────────────────

LELAPA_API_BASE = "https://vulavula-services.lelapa.ai/api/v1"
# Set LELAPA_API_KEY in your .env or environment before running.

def transcribe_lelapa(audio_path: str, api_key: str | None = None,
                      language_code: str = "sot") -> dict:
    """
    Transcribe via the Lelapa AI Vulavula ASR REST API.

    Steps
    -----
    1. Upload audio file  → get upload_id
    2. Trigger transcription → get job_id
    3. Poll until complete  → retrieve transcript

    API docs: https://lelapa.ai/vulavula-api-docs
    Note: Setswana (tn / Tswana) may map to "sot" or "tsn" depending on
    the Lelapa endpoint version. Check their docs and update language_code.

    Parameters
    ----------
    audio_path    : str   Path to 16kHz mono WAV file
    api_key       : str   Your Lelapa API key (or set env var LELAPA_API_KEY)
    language_code : str   Lelapa language code for Setswana

    Returns
    -------
    dict with keys: engine, file, transcript, segments
    """
    key = api_key or os.environ.get("LELAPA_API_KEY", "")
    if not key:
        raise EnvironmentError(
            "Lelapa API key not found. Set LELAPA_API_KEY env variable or pass api_key=..."
        )

    headers = {"X-CLIENT-TOKEN": key}

    # ── Step 1: Upload audio ──────────────────────────────────
    logger.info("[Lelapa] Uploading audio …")
    with open(audio_path, "rb") as f:
        upload_resp = requests.post(
            f"{LELAPA_API_BASE}/transcribe/upload",
            headers=headers,
            files={"file": (os.path.basename(audio_path), f, "audio/wav")},
            timeout=120,
        )
    upload_resp.raise_for_status()
    upload_id = upload_resp.json()["upload_id"]
    logger.info(f"  upload_id: {upload_id}")

    # ── Step 2: Start transcription job ──────────────────────
    logger.info("[Lelapa] Starting transcription job …")
    job_resp = requests.post(
        f"{LELAPA_API_BASE}/transcribe/process",
        headers={**headers, "Content-Type": "application/json"},
        json={"upload_id": upload_id, "language": language_code},
        timeout=60,
    )
    job_resp.raise_for_status()
    job_id = job_resp.json()["job_id"]
    logger.info(f"  job_id: {job_id}")

    # ── Step 3: Poll for result ───────────────────────────────
    logger.info("[Lelapa] Polling for result (may take 1–5 min) …")
    for attempt in range(60):          # max ~10 minutes
        time.sleep(10)
        status_resp = requests.get(
            f"{LELAPA_API_BASE}/transcribe/status/{job_id}",
            headers=headers,
            timeout=30,
        )
        status_resp.raise_for_status()
        data = status_resp.json()
        status = data.get("status", "unknown")
        logger.info(f"  Attempt {attempt+1}: status = {status}")

        if status == "completed":
            transcript = data.get("transcription", "")
            segments = data.get("segments", [])
            return {
                "engine": "lelapa",
                "file": audio_path,
                "transcript": transcript.strip(),
                "segments": segments,
            }
        elif status in ("failed", "error"):
            raise RuntimeError(f"Lelapa job failed: {data}")

    raise TimeoutError("Lelapa transcription timed out after 10 minutes.")


# ─────────────────────────────────────────────────────────────
# 3.  wav2vec-2.0 / AfroXLSR
# ─────────────────────────────────────────────────────────────

# Community checkpoint trained on multiple African languages including Setswana
AFROXLSR_CHECKPOINT = "lighteternal/asr-wav2vec2-xls-r-afr-tn"
# Alternative: "chrisjay/fonxlsr" if above unavailable

def transcribe_wav2vec(audio_path: str, checkpoint: str = AFROXLSR_CHECKPOINT,
                       chunk_length_s: int = 30) -> dict:
    """
    Transcribe with wav2vec-2.0 using an AfroXLSR Setswana checkpoint.

    The model is loaded from HuggingFace Hub. On first run it will be
    downloaded (~1.2 GB). Subsequent runs use the local cache.

    Parameters
    ----------
    audio_path     : str   Path to audio file
    checkpoint     : str   HuggingFace model hub checkpoint
    chunk_length_s : int   Chunk size for long-form transcription (seconds)

    Returns
    -------
    dict with keys: engine, file, transcript, segments
    """
    from transformers import pipeline as hf_pipeline

    logger.info(f"[wav2vec] Loading model: {checkpoint} …")
    device = 0 if torch.cuda.is_available() else -1
    logger.info(f"  Device: {'GPU' if device==0 else 'CPU'}")

    asr_pipe = hf_pipeline(
        "automatic-speech-recognition",
        model=checkpoint,
        device=device,
        chunk_length_s=chunk_length_s,
        stride_length_s=5,
    )

    logger.info(f"[wav2vec] Transcribing {os.path.basename(audio_path)} …")
    audio, sr = load_audio(audio_path, target_sr=16_000)

    result = asr_pipe({"array": audio, "sampling_rate": sr},
                      return_timestamps="word")

    # Build flat transcript + crude segments from word-level timestamps
    chunks = result.get("chunks", [])
    full_text = " ".join(c["text"] for c in chunks) if chunks else result.get("text", "")

    segments = []
    current_segment = {"start": None, "end": None, "words": []}
    for chunk in chunks:
        ts = chunk.get("timestamp", (None, None))
        if current_segment["start"] is None:
            current_segment["start"] = ts[0]
        current_segment["end"] = ts[1]
        current_segment["words"].append(chunk["text"])

        # Split into ~10-word segments for readability
        if len(current_segment["words"]) >= 10:
            segments.append({
                "start": current_segment["start"],
                "end": current_segment["end"],
                "text": " ".join(current_segment["words"]),
            })
            current_segment = {"start": None, "end": None, "words": []}

    if current_segment["words"]:
        segments.append({
            "start": current_segment["start"],
            "end": current_segment["end"],
            "text": " ".join(current_segment["words"]),
        })

    return {
        "engine": "wav2vec",
        "file": audio_path,
        "transcript": full_text.strip(),
        "segments": segments,
    }


# ─────────────────────────────────────────────────────────────
# 4.  BATCH RUNNER
# ─────────────────────────────────────────────────────────────

def run_all_engines(audio_files: list[str], output_dir: str,
                    lelapa_api_key: str | None = None,
                    skip_lelapa: bool = False,
                    skip_wav2vec: bool = False) -> dict[str, list[dict]]:
    """
    Run all three ASR engines on every audio file in the list.

    Parameters
    ----------
    audio_files    : list of paths to audio files
    output_dir     : directory where JSON transcripts will be saved
    lelapa_api_key : optional API key (falls back to env var)
    skip_lelapa    : set True if you don't have API access yet
    skip_wav2vec   : set True to skip wav2vec (slow on CPU)

    Returns
    -------
    results : {"whisper": [...], "lelapa": [...], "wav2vec": [...]}
    """
    os.makedirs(output_dir, exist_ok=True)
    results = {"whisper": [], "lelapa": [], "wav2vec": []}

    # Pre-process all audio to 16kHz WAV once
    wav_files = []
    for af in audio_files:
        wav_path = os.path.join(output_dir, os.path.splitext(os.path.basename(af))[0] + "_16k.wav")
        if not os.path.exists(wav_path):
            audio, sr = load_audio(af)
            save_wav(audio, sr, wav_path)
        wav_files.append(wav_path)

    for wav in wav_files:
        base = os.path.splitext(os.path.basename(wav))[0]

        # ── Whisper ──────────────────────────────────────────
        w_path = os.path.join(output_dir, f"{base}_whisper.json")
        if os.path.exists(w_path):
            logger.info(f"[Whisper] Cache hit: {w_path}")
            with open(w_path) as f:
                w_res = json.load(f)
        else:
            try:
                w_res = transcribe_whisper(wav)
                with open(w_path, "w", encoding="utf-8") as f:
                    json.dump(w_res, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.error(f"[Whisper] Failed on {wav}: {e}")
                w_res = {"engine": "whisper", "file": wav, "transcript": "", "segments": [], "error": str(e)}
        results["whisper"].append(w_res)

        # ── Lelapa ───────────────────────────────────────────
        l_path = os.path.join(output_dir, f"{base}_lelapa.json")
        if skip_lelapa:
            logger.info("[Lelapa] Skipped (skip_lelapa=True)")
            results["lelapa"].append({"engine": "lelapa", "file": wav, "transcript": "SKIPPED", "segments": []})
        elif os.path.exists(l_path):
            logger.info(f"[Lelapa] Cache hit: {l_path}")
            with open(l_path) as f:
                results["lelapa"].append(json.load(f))
        else:
            try:
                l_res = transcribe_lelapa(wav, api_key=lelapa_api_key)
                with open(l_path, "w", encoding="utf-8") as f:
                    json.dump(l_res, f, ensure_ascii=False, indent=2)
                results["lelapa"].append(l_res)
            except Exception as e:
                logger.error(f"[Lelapa] Failed on {wav}: {e}")
                results["lelapa"].append({"engine": "lelapa", "file": wav, "transcript": "", "segments": [], "error": str(e)})

        # ── wav2vec ───────────────────────────────────────────
        v_path = os.path.join(output_dir, f"{base}_wav2vec.json")
        if skip_wav2vec:
            logger.info("[wav2vec] Skipped (skip_wav2vec=True)")
            results["wav2vec"].append({"engine": "wav2vec", "file": wav, "transcript": "SKIPPED", "segments": []})
        elif os.path.exists(v_path):
            logger.info(f"[wav2vec] Cache hit: {v_path}")
            with open(v_path) as f:
                results["wav2vec"].append(json.load(f))
        else:
            try:
                v_res = transcribe_wav2vec(wav)
                with open(v_path, "w", encoding="utf-8") as f:
                    json.dump(v_res, f, ensure_ascii=False, indent=2)
                results["wav2vec"].append(v_res)
            except Exception as e:
                logger.error(f"[wav2vec] Failed on {wav}: {e}")
                results["wav2vec"].append({"engine": "wav2vec", "file": wav, "transcript": "", "segments": [], "error": str(e)})

    return results