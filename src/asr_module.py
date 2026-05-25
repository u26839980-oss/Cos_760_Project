"""
asr_module.py
=============
Three confirmed working ASR engines for TWO language options.

SETSWANA (--language tsn):
  Engine 1 - intronhealth/afrispeech-whisper-medium-all   [Whisper-based]
  Engine 2 - facebook/mms-1b-all  adapter="tsn"           [wav2vec2-based]
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

logger = logging.getLogger(__name__)

# # ── Model IDs (confirmed working, do not change) ────────────────────────────
# MODELS = {
#     "tsn": {
#         "whisper":  "intronhealth/afrispeech-whisper-medium-all",
#         "mms_lang": "tsn",                      # ISO 639-3 for Setswana
#         "whisper3": "openai/whisper-large-v3",  # fallback off-the-shelf
#     },
#     "zul": {
#         "whisper":  "TheirStory/whisper-medium-zulu",
#         "mms_lang": "zul",                      # ISO 639-3 for isiZulu
#         # Lelapa is used instead of whisper3 for Zulu
#     },
# }

MODELS = {
   "tsn": {
        "whisper":  "intronhealth/afrispeech-whisper-medium-all",
        "whisper3": "openai/whisper-large-v3",
        "kesego":   "models/kesego_whisper_setswana_merged",   # now a local folder
    },
    "zul": {
        "whisper":  "TheirStory/whisper-medium-zulu",
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
# ENGINE 2: Meta MMS  (both languages via adapter)
# ============================================================

# def transcribe_mms(wav_path: str, lang_code: str) -> dict:
#     # """
#     # facebook/mms-1b-all
#     # Meta's Massively Multilingual Speech model (1B params, 1107 languages).
#     # Uses small language adapter weights (~2MB) loaded on the fly.
#     # Confirmed language codes:  tsn = Setswana,  zul = isiZulu

#     # NOTE: ignore_mismatched_sizes=True is REQUIRED by the official docs
#     # because the adapter head shape differs from the base model.
#     # """
#     # logger.info(f"  [MMS] Loading facebook/mms-1b-all with adapter={lang_code} ...")
#     # pipe = _hf_pipeline(
#     #     "facebook/mms-1b-all",
#     #     model_kwargs={
#     #         "target_lang":            lang_code,
#     #         "ignore_mismatched_sizes": True,
#     #     },
#     # )
#     # text = _run_pipeline(pipe, wav_path)
#     # return _make_result("mms", wav_path, text)
#     """
#     facebook/mms-1b-all with language adapter loaded manually.
#     Works around the pipeline’s automatic adapter loading bug.
#     """
#     from transformers import Wav2Vec2ForCTC, AutoProcessor, pipeline

#     logger.info(f"  [MMS] Loading facebook/mms-1b-all with adapter={lang_code} ...")
#     processor = AutoProcessor.from_pretrained("facebook/mms-1b-all")
#     model = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all")

#     # Set the target language and load the corresponding adapter weights
#     processor.tokenizer.set_target_lang(lang_code)
#     model.load_adapter(lang_code)

#     # Move to GPU if available
#     device = 0 if torch.cuda.is_available() else -1
#     if device >= 0:
#         model = model.to("cuda")

#     # Build a pipeline with the already‑loaded model & processor
#     pipe = pipeline(
#         "automatic-speech-recognition",
#         model=model,
#         tokenizer=processor.tokenizer,
#         feature_extractor=processor.feature_extractor,
#         device=device,
#         chunk_length_s=30,
#         stride_length_s=5,
#     )

#     text = _run_pipeline(pipe, wav_path)
#     return _make_result("mms", wav_path, text)


# # ============================================================
# # ENGINE 3a: Whisper large-v3 auto-detect  (Setswana fallback)
# # ============================================================

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

def transcribe_kesego_tsn(wav_path: str) -> dict:
    """
    kesego/whisper-large-v3-setswana
    A Whisper large-v3 model fine-tuned specifically on Setswana
    (including code‑switched speech). Should outperform the off‑the‑shelf
    large-v3 on pure Setswana audio.

    No language tag is needed – the model is already specialised to Setswana.
    """
    pipe = _hf_pipeline(MODELS["tsn"]["kesego"])
    text = _run_pipeline(pipe, wav_path)
    return _make_result("kesego_tsn", wav_path, text)

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
    """
    assert language in ("tsn", "zul"), "language must be 'tsn' or 'zul'"
    os.makedirs(out_dir, exist_ok=True)

    # ── Define the three engines for each language ──────────────────────────
    if language == "tsn":
        engines = {
            "afrispeech_whisper": lambda wav: transcribe_afrispeech_whisper(wav),
            "kesego_tsn":         lambda wav: transcribe_kesego_tsn(wav),
            "whisper_large_v3":   lambda wav: transcribe_whisper_large_v3(wav),
        }
        # Override skip flags for Setswana (no MMS, no Lelapa)
        skip_mms = True
        skip_lelapa = True
    else:  # Zulu
        engines = {
            "whisper_zulu": lambda wav: transcribe_whisper_zulu(wav),
            #"mms":          lambda wav: transcribe_mms(wav, MODELS["zul"]["mms_lang"]),
            "lelapa":       lambda wav: transcribe_lelapa_zulu(wav, api_key=api_key),
        }

    # Apply user‑requested skip flags
    if skip_mms and "mms" in engines:
        del engines["mms"]
    if (skip_lelapa or skip_engine3) and "lelapa" in engines:
        del engines["lelapa"]
    if skip_engine3 and "whisper_large_v3" in engines:
        del engines["whisper_large_v3"]

    results = {name: [] for name in engines}

    # ── Process each audio file ─────────────────────────────────────────────
    for audio_path in audio_files:
        wav = ensure_wav(audio_path, out_dir)
        stem = os.path.splitext(os.path.basename(wav))[0]

        for eng_name, eng_func in engines.items():
            cache = os.path.join(out_dir, f"{stem}_{eng_name}.json")
            if os.path.exists(cache):
                logger.info(f"[{eng_name}] Cache hit")
                r = json.load(open(cache, encoding="utf-8"))
            else:
                logger.info(f"\n[{eng_name}] Running ...")
                try:
                    r = eng_func(wav)
                    json.dump(r, open(cache, "w", encoding="utf-8"),
                              ensure_ascii=False, indent=2)
                except Exception as e:
                    logger.error(f"[{eng_name}] FAILED: {e}")
                    r = _make_result(eng_name, wav, "")
                    r["error"] = str(e)
            results[eng_name].append(r)

    # ── Summary ─────────────────────────────────────────────────────────────
    logger.info("\n  === ASR Summary ===")
    for eng, res_list in results.items():
        words  = sum(len(r.get("transcript","").split()) for r in res_list)
        errors = sum(1 for r in res_list if "error" in r)
        logger.info(f"  {eng:<25} {words:>5} words  |  {errors} error(s)")

    return results

# def run_all_engines(audio_files: list[str],
#                     out_dir: str,
#                     language: str = "tsn",
#                     lelapa_api_key: str | None = None,
#                     skip_lelapa: bool = False,
#                     skip_mms: bool = False,
#                     skip_engine3: bool = False) -> dict[str, list[dict]]:
#     """
#     Run all three engines on every audio file. Results cached as JSON.

#     Parameters
#     ----------
#     audio_files    : list of paths to audio files in data/audio/
#     out_dir        : where to cache JSON transcript files
#     language       : "tsn" (Setswana) or "zul" (isiZulu)
#     lelapa_api_key : Lelapa key (isiZulu only); or set LELAPA_API_KEY env var
#     skip_lelapa    : skip Lelapa (isiZulu only)
#     skip_mms       : skip MMS engine (slow on CPU without GPU)
#     skip_engine3   : skip Engine 3 entirely

#     Returns
#     -------
#     dict keyed by engine name, each value is a list of result dicts
#     """
#     assert language in ("tsn", "zul"), "language must be 'tsn' or 'zul'"
#     os.makedirs(out_dir, exist_ok=True)

#     # Engine names depend on language
#     if language == "tsn":
#         eng1_name = "afrispeech_whisper"
#         eng3_name = "whisper_large_v3"
#     else:
#         eng1_name = "whisper_zulu"
#         eng3_name = "lelapa"

#     results = {eng1_name: [], "mms": [], eng3_name: []}

#     for audio_path in audio_files:
#         wav = ensure_wav(audio_path, out_dir)
#         stem = os.path.splitext(os.path.basename(wav))[0]

#         # ── Engine 1 ────────────────────────────────────────────────────────
#         cache = os.path.join(out_dir, f"{stem}_{eng1_name}.json")
#         if os.path.exists(cache):
#             logger.info(f"[{eng1_name}] Cache hit")
#             r = json.load(open(cache, encoding="utf-8"))
#         else:
#             logger.info(f"\n[{eng1_name}] Running ...")
#             try:
#                 r = (transcribe_afrispeech_whisper(wav)
#                      if language == "tsn"
#                      else transcribe_whisper_zulu(wav))
#                 json.dump(r, open(cache, "w", encoding="utf-8"),
#                           ensure_ascii=False, indent=2)
#             except Exception as e:
#                 logger.error(f"[{eng1_name}] FAILED: {e}")
#                 r = _make_result(eng1_name, wav, "")
#                 r["error"] = str(e)
#         results[eng1_name].append(r)

#         # ── Engine 2: MMS ───────────────────────────────────────────────────
#         cache = os.path.join(out_dir, f"{stem}_mms.json")
#         if skip_mms:
#             logger.info("[MMS] Skipped (--skip-mms)")
#             results["mms"].append(_make_result("mms", wav, ""))
#         elif os.path.exists(cache):
#             logger.info("[MMS] Cache hit")
#             results["mms"].append(json.load(open(cache, encoding="utf-8")))
#         else:
#             logger.info(f"\n[MMS] Running (lang={MODELS[language]['mms_lang']}) ...")
#             try:
#                 r = transcribe_mms(wav, MODELS[language]["mms_lang"])
#                 json.dump(r, open(cache, "w", encoding="utf-8"),
#                           ensure_ascii=False, indent=2)
#                 results["mms"].append(r)
#             except Exception as e:
#                 logger.error(f"[MMS] FAILED: {e}")
#                 r = _make_result("mms", wav, "")
#                 r["error"] = str(e)
#                 results["mms"].append(r)

#         # ── Engine 3 ────────────────────────────────────────────────────────
#         cache = os.path.join(out_dir, f"{stem}_{eng3_name}.json")
#         # Auto-skip Lelapa if no key
#         api_key = lelapa_api_key or os.environ.get("LELAPA_API_KEY", "").strip()
#         if language == "zul" and not api_key:
#             skip_lelapa = True
#             logger.warning(
#                 "[Lelapa] Auto-skipping: LELAPA_API_KEY not set.\n"
#                 "  Set it in PowerShell: $env:LELAPA_API_KEY='your_key'\n"
#                 "  Then rerun without --skip-lelapa"
#             )

#         _skip = skip_engine3 or (language == "zul" and skip_lelapa)
#         if _skip:
#             logger.info(f"[{eng3_name}] Skipped")
#             results[eng3_name].append(_make_result(eng3_name, wav, ""))
#         elif os.path.exists(cache):
#             logger.info(f"[{eng3_name}] Cache hit")
#             results[eng3_name].append(json.load(open(cache, encoding="utf-8")))
#         else:
#             logger.info(f"\n[{eng3_name}] Running ...")
#             try:
#                 if language == "tsn":
#                     r = transcribe_whisper_large_v3(wav)
#                 else:
#                     r = transcribe_lelapa_zulu(wav, api_key=api_key)
#                 json.dump(r, open(cache, "w", encoding="utf-8"),
#                           ensure_ascii=False, indent=2)
#                 results[eng3_name].append(r)
#             except Exception as e:
#                 logger.error(f"[{eng3_name}] FAILED: {e}")
#                 r = _make_result(eng3_name, wav, "")
#                 r["error"] = str(e)
#                 results[eng3_name].append(r)

#     # Summary
#     logger.info("\n  === ASR Summary ===")
#     for eng, res_list in results.items():
#         words  = sum(len(r.get("transcript","").split()) for r in res_list)
#         errors = sum(1 for r in res_list if "error" in r)
#         logger.info(f"  {eng:<25} {words:>5} words  |  {errors} error(s)")

#     return results