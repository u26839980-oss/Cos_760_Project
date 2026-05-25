"""
asr_module.py
=============
Three ASR engines for TWO language options.

SETSWANA (--language tsn):
  Engine 1 - intronhealth/afrispeech-whisper-medium-all   [Whisper-based]
  Engine 2 - sitwala/whisper-large-v3-turbo-anv-tsn-50h   [Whisper fine-tuned]
  Engine 3 - openai/whisper-large-v3  language=auto       [Whisper off-the-shelf]

ISIZULU (--language zul):
  Engine 1 - TheirStory/whisper-medium-zulu               [Whisper fine-tuned]
  Engine 2 - facebook/mms-1b-all  adapter="zul"           [wav2vec2-based]
  Engine 3 - Lelapa AI Vulavula API                       [Commercial API]

All three engines use the HuggingFace pipeline() API consistently.
Every result is cached as JSON - rerunning is instant.
"""

import os, json, time, logging
import numpy as np
import torch
import librosa, soundfile as sf
import requests

try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    truststore = None

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HF_CACHE_DIR = os.path.join(PROJECT_ROOT, "src", ".hf_cache")
os.makedirs(HF_CACHE_DIR, exist_ok=True)
os.environ.setdefault("HF_HOME", HF_CACHE_DIR)
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(HF_CACHE_DIR, "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(HF_CACHE_DIR, "transformers"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# ── Model IDs (confirmed working, do not change) ────────────────────────────
MODELS = {
    "tsn": {
        "whisper": "intronhealth/afrispeech-whisper-medium-all",
        "whisper_ft": "sitwala/whisper-large-v3-turbo-anv-tsn-50h",
        "whisper3": "openai/whisper-large-v3",
        "mms_lang": None,
    },
    "zul": {
        "whisper": "TheirStory/whisper-medium-zulu",
        "mms_lang": "zul",
    },
}

LELAPA_BASE = "https://vulavula-services.lelapa.ai/api/v1"


# ============================================================
# AUDIO HELPERS
# ============================================================

def load_and_resample(path: str, sr: int = 16_000):
    """Load any audio file and resample to 16 kHz mono float32."""
    audio, _ = librosa.load(path, sr=sr, mono=True)
    logger.info(f"  Loaded {os.path.basename(path)}: "
                f"{len(audio)/sr:.1f}s at {sr} Hz")
    return audio, sr


def ensure_wav(audio_path: str, out_dir: str) -> str:
    """Convert any audio file to 16-bit mono 16 kHz WAV if needed."""
    stem    = os.path.splitext(os.path.basename(audio_path))[0]
    wav_out = os.path.join(out_dir, f"{stem}_16k.wav")
    if not os.path.exists(wav_out):
        audio, sr = load_and_resample(audio_path)
        sf.write(wav_out, audio, sr, subtype="PCM_16")
        logger.info(f"  Saved WAV -> {wav_out}")
    return wav_out


def _hf_pipeline(model_id: str, **kwargs):
    """Create a HuggingFace ASR pipeline on GPU if available, else CPU."""
    device = 0 if torch.cuda.is_available() else -1
    logger.info(f"  Loading {model_id} on {'GPU' if device==0 else 'CPU'} ...")
    from transformers import pipeline
    return pipeline(
        "automatic-speech-recognition",
        model=model_id,
        device=device,
        chunk_length_s=30,
        stride_length_s=5,
        **kwargs,
    )


def _run_pipeline(pipe, wav_path: str) -> str:
    """Run a loaded pipeline on a WAV file, return transcript string."""
    audio, sr = load_and_resample(wav_path)
    result    = pipe({"array": audio, "sampling_rate": sr},
                     return_timestamps=False)
    text = result.get("text", "").strip()
    logger.info(f"  -> {len(text.split())} words transcribed")
    return text


def _make_result(engine: str, wav: str, text: str) -> dict:
    return {"engine": engine, "file": wav, "transcript": text}


# ============================================================
# ENGINE 1a: AfriSpeech-Whisper  (Setswana)
# ============================================================

def transcribe_afrispeech_whisper(wav_path: str) -> dict:
    """
    intronhealth/afrispeech-whisper-medium-all
    Fine-tuned on AfriSpeech-200: 200+ hours of African-accented
    and African-language speech. Handles Setswana without needing
    a language code (auto-detects from the audio).
    """
    pipe = _hf_pipeline(MODELS["tsn"]["whisper"])
    text = _run_pipeline(pipe, wav_path)
    return _make_result("afrispeech_whisper", wav_path, text)


# ============================================================
# ENGINE 1b: Whisper fine-tuned on Zulu  (isiZulu)
# ============================================================

def transcribe_whisper_zulu(wav_path: str) -> dict:
    """
    TheirStory/whisper-medium-zulu
    Fine-tuned from openai/whisper-medium on the Zulu audio dataset.
    WER = 0.1993 on evaluation set.
    No language code needed - model is Zulu-only.
    """
    pipe = _hf_pipeline(MODELS["zul"]["whisper"])
    text = _run_pipeline(pipe, wav_path)
    return _make_result("whisper_zulu", wav_path, text)


# ============================================================
# ENGINE 2a: Setswana Whisper fine-tune
# ============================================================

def transcribe_setswana_whisper_ft(wav_path: str) -> dict:
    """
    sitwala/whisper-large-v3-turbo-anv-tsn-50h
    Fine-tuned Whisper-large-v3-turbo checkpoint for Setswana (`tsn`)
    on the African Next Voices dataset family.
    """
    pipe = _hf_pipeline(
        MODELS["tsn"]["whisper_ft"],
        generate_kwargs={"language": None, "task": "transcribe"},
    )
    text = _run_pipeline(pipe, wav_path)
    return _make_result("setswana_whisper_ft", wav_path, text)


# ============================================================
# ENGINE 2b: Meta MMS  (isiZulu only)
# ============================================================

def transcribe_mms(wav_path: str, lang_code: str) -> dict:
    """
    facebook/mms-1b-all
    Meta's Massively Multilingual Speech model (1B params, 1107 languages).
    Uses small language adapter weights (~2MB) loaded on the fly.
    Confirmed language codes:  tsn = Setswana,  zul = isiZulu

    NOTE: ignore_mismatched_sizes=True is REQUIRED by the official docs
    because the adapter head shape differs from the base model.
    """
    if not lang_code:
        raise ValueError("MMS adapter not available for this language.")
    logger.info(f"  [MMS] Loading facebook/mms-1b-all with adapter={lang_code} ...")
    pipe = _hf_pipeline(
        "facebook/mms-1b-all",
        model_kwargs={
            "target_lang":            lang_code,
            "ignore_mismatched_sizes": True,
        },
    )
    text = _run_pipeline(pipe, wav_path)
    return _make_result("mms", wav_path, text)


# ============================================================
# ENGINE 3a: Whisper large-v3 auto-detect  (Setswana fallback)
# ============================================================

def transcribe_whisper_large_v3(wav_path: str) -> dict:
    """
    openai/whisper-large-v3  with language=None (auto-detect).
    Used as the third Setswana engine because Lelapa does NOT support
    Setswana STT. Whisper large-v3 is the 'off-the-shelf' reference
    point the proposal specifically names.

    NOTE: We pass language=None. Do NOT pass language='tn' - that code
    is not in Whisper's tokenizer and causes 'Unsupported language' error.
    """
    logger.info("  [Whisper-v3] Loading openai/whisper-large-v3 (auto-detect) ...")
    pipe = _hf_pipeline(
        "openai/whisper-large-v3",
        generate_kwargs={"language": None, "task": "transcribe"},
    )
    text = _run_pipeline(pipe, wav_path)
    return _make_result("whisper_large_v3", wav_path, text)


# ============================================================
# ENGINE 3b: Lelapa AI API  (isiZulu only)
# ============================================================

def transcribe_lelapa_zulu(wav_path: str, api_key: str | None = None) -> dict:
    """
    Lelapa AI Vulavula API - isiZulu ASR.
    Lelapa supports isiZulu (zul) but NOT Setswana STT natively.

    Get your API key at: https://lelapa.ai
    Set it:  $env:LELAPA_API_KEY="your_key"   (PowerShell)
    """
    key = api_key or os.environ.get("LELAPA_API_KEY", "").strip()
    if not key:
        raise EnvironmentError(
            "LELAPA_API_KEY not set.\n"
            "  PowerShell: $env:LELAPA_API_KEY='paste_key_here'\n"
            "  Then rerun WITHOUT --skip-lelapa"
        )

    h = {"X-CLIENT-TOKEN": key}

    # Upload
    logger.info("[Lelapa] Uploading ...")
    with open(wav_path, "rb") as f:
        r = requests.post(f"{LELAPA_BASE}/transcribe/upload", headers=h,
                          files={"file": (os.path.basename(wav_path), f, "audio/wav")},
                          timeout=120)
    if r.status_code == 401:
        raise PermissionError("Lelapa key rejected (401). Double-check LELAPA_API_KEY.")
    r.raise_for_status()
    upload_id = r.json()["upload_id"]

    # Trigger
    logger.info(f"[Lelapa] Triggering job (upload_id={upload_id}) ...")
    r2 = requests.post(f"{LELAPA_BASE}/transcribe/process",
                       headers={**h, "Content-Type": "application/json"},
                       json={"upload_id": upload_id, "language": "zul"},
                       timeout=60)
    r2.raise_for_status()
    job_id = r2.json()["job_id"]
    logger.info(f"[Lelapa] job_id={job_id} - polling ...")

    # Poll
    for i in range(72):
        time.sleep(10)
        r3 = requests.get(f"{LELAPA_BASE}/transcribe/status/{job_id}",
                          headers=h, timeout=30)
        r3.raise_for_status()
        data   = r3.json()
        status = data.get("status", "unknown")
        logger.info(f"  Attempt {i+1:02d}: {status}")
        if status == "completed":
            text = data.get("transcription", "").strip()
            logger.info(f"  -> {len(text.split())} words")
            return _make_result("lelapa", wav_path, text)
        if status in ("failed", "error"):
            raise RuntimeError(f"Lelapa job failed: {data}")

    raise TimeoutError("Lelapa timed out after 12 min.")


# ============================================================
# BATCH RUNNER  (call this from pipeline.py)
# ============================================================

def run_all_engines(audio_files: list[str],
                    out_dir: str,
                    language: str = "tsn",
                    lelapa_api_key: str | None = None,
                    skip_lelapa: bool = False,
                    skip_mms: bool = False,
                    skip_engine3: bool = False) -> dict[str, list[dict]]:
    """
    Run all three engines on every audio file. Results cached as JSON.

    Parameters
    ----------
    audio_files    : list of paths to audio files in data/audio/
    out_dir        : where to cache JSON transcript files
    language       : "tsn" (Setswana) or "zul" (isiZulu)
    lelapa_api_key : Lelapa key (isiZulu only); or set LELAPA_API_KEY env var
    skip_lelapa    : skip Lelapa (isiZulu only)
    skip_mms       : skip MMS engine (slow on CPU without GPU)
    skip_engine3   : skip Engine 3 entirely

    Returns
    -------
    dict keyed by engine name, each value is a list of result dicts
    """
    assert language in ("tsn", "zul"), "language must be 'tsn' or 'zul'"
    os.makedirs(out_dir, exist_ok=True)

    # Engine names depend on language
    if language == "tsn":
        eng1_name = "afrispeech_whisper"
        eng2_name = "setswana_whisper_ft"
        eng3_name = "whisper_large_v3"
    else:
        eng1_name = "whisper_zulu"
        eng2_name = "mms"
        eng3_name = "lelapa"

    results = {eng1_name: [], eng2_name: [], eng3_name: []}

    for audio_path in audio_files:
        wav = ensure_wav(audio_path, out_dir)
        stem = os.path.splitext(os.path.basename(wav))[0]

        # ── Engine 1 ────────────────────────────────────────────────────────
        cache = os.path.join(out_dir, f"{stem}_{eng1_name}.json")
        if os.path.exists(cache):
            logger.info(f"[{eng1_name}] Cache hit")
            r = json.load(open(cache, encoding="utf-8"))
        else:
            logger.info(f"\n[{eng1_name}] Running ...")
            try:
                r = (transcribe_afrispeech_whisper(wav)
                     if language == "tsn"
                     else transcribe_whisper_zulu(wav))
                json.dump(r, open(cache, "w", encoding="utf-8"),
                          ensure_ascii=False, indent=2)
            except Exception as e:
                logger.error(f"[{eng1_name}] FAILED: {e}")
                r = _make_result(eng1_name, wav, "")
                r["error"] = str(e)
        results[eng1_name].append(r)

        # ── Engine 2: MMS ───────────────────────────────────────────────────
        cache = os.path.join(out_dir, f"{stem}_{eng2_name}.json")
        if language == "tsn":
            if skip_mms:
                logger.info(f"[{eng2_name}] Skipped (--skip-mms mapped to engine 2)")
                results[eng2_name].append(_make_result(eng2_name, wav, ""))
            elif os.path.exists(cache):
                logger.info(f"[{eng2_name}] Cache hit")
                results[eng2_name].append(json.load(open(cache, encoding="utf-8")))
            else:
                logger.info(f"\n[{eng2_name}] Running ...")
                try:
                    r = transcribe_setswana_whisper_ft(wav)
                    json.dump(r, open(cache, "w", encoding="utf-8"),
                              ensure_ascii=False, indent=2)
                    results[eng2_name].append(r)
                except Exception as e:
                    logger.error(f"[{eng2_name}] FAILED: {e}")
                    r = _make_result(eng2_name, wav, "")
                    r["error"] = str(e)
                    results[eng2_name].append(r)
        else:
            if skip_mms:
                logger.info("[MMS] Skipped (--skip-mms)")
                results[eng2_name].append(_make_result(eng2_name, wav, ""))
            elif not MODELS[language]["mms_lang"]:
                logger.warning("[MMS] Skipped: no supported MMS adapter is available for this language.")
                results[eng2_name].append(_make_result(eng2_name, wav, ""))
            elif os.path.exists(cache):
                logger.info("[MMS] Cache hit")
                results[eng2_name].append(json.load(open(cache, encoding="utf-8")))
            else:
                logger.info(f"\n[MMS] Running (lang={MODELS[language]['mms_lang']}) ...")
                try:
                    r = transcribe_mms(wav, MODELS[language]["mms_lang"])
                    json.dump(r, open(cache, "w", encoding="utf-8"),
                              ensure_ascii=False, indent=2)
                    results[eng2_name].append(r)
                except Exception as e:
                    logger.error(f"[MMS] FAILED: {e}")
                    r = _make_result(eng2_name, wav, "")
                    r["error"] = str(e)
                    results[eng2_name].append(r)

        # ── Engine 3 ────────────────────────────────────────────────────────
        cache = os.path.join(out_dir, f"{stem}_{eng3_name}.json")
        # Auto-skip Lelapa if no key
        api_key = lelapa_api_key or os.environ.get("LELAPA_API_KEY", "").strip()
        if language == "zul" and not api_key:
            skip_lelapa = True
            logger.warning(
                "[Lelapa] Auto-skipping: LELAPA_API_KEY not set.\n"
                "  Set it in PowerShell: $env:LELAPA_API_KEY='your_key'\n"
                "  Then rerun without --skip-lelapa"
            )

        _skip = skip_engine3 or (language == "zul" and skip_lelapa)
        if _skip:
            logger.info(f"[{eng3_name}] Skipped")
            results[eng3_name].append(_make_result(eng3_name, wav, ""))
        elif os.path.exists(cache):
            logger.info(f"[{eng3_name}] Cache hit")
            results[eng3_name].append(json.load(open(cache, encoding="utf-8")))
        else:
            logger.info(f"\n[{eng3_name}] Running ...")
            try:
                if language == "tsn":
                    r = transcribe_whisper_large_v3(wav)
                else:
                    r = transcribe_lelapa_zulu(wav, api_key=api_key)
                json.dump(r, open(cache, "w", encoding="utf-8"),
                          ensure_ascii=False, indent=2)
                results[eng3_name].append(r)
            except Exception as e:
                logger.error(f"[{eng3_name}] FAILED: {e}")
                r = _make_result(eng3_name, wav, "")
                r["error"] = str(e)
                results[eng3_name].append(r)

    # Summary
    logger.info("\n  === ASR Summary ===")
    for eng, res_list in results.items():
        words  = sum(len(r.get("transcript","").split()) for r in res_list)
        errors = sum(1 for r in res_list if "error" in r)
        logger.info(f"  {eng:<25} {words:>5} words  |  {errors} error(s)")

    return results
