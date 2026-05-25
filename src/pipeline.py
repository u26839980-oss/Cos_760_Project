#!/usr/bin/env python3
"""
pipeline.py  -  Setswana edition
==================================
COS760 NLP Project - Group 37
ASR + Topic Modelling for Setswana COVID-19 Podcasts

Three Setswana-compatible ASR engines:
  1. afrispeech_whisper  - intronhealth/afrispeech-whisper-medium-all
  2. mms                 - facebook/mms-1b-all  (tsn adapter)
  3. lelapa              - Lelapa AI Vulavula API

Usage
-----
  # Demo (no audio needed):
  python pipeline.py --mode demo

  # Real audio, all 3 engines:
  python pipeline.py --mode real

  # Skip Lelapa (no API key yet):
  python pipeline.py --mode real --skip-lelapa

  # Skip MMS (too slow on CPU):
  python pipeline.py --mode real --skip-mms

  # Skip both, only AfriSpeech-Whisper:
  python pipeline.py --mode real --skip-lelapa --skip-mms
"""

import io, os, sys, json, glob, logging, argparse, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# ── Windows-safe UTF-8 logging ────────────────────────────────────────────────
def _utf8_stream(stream):
    try:
        if hasattr(stream, "buffer"):
            return io.TextIOWrapper(stream.buffer, encoding="utf-8",
                                    errors="replace", line_buffering=True)
    except Exception:
        pass
    return stream

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("pipeline.log", encoding="utf-8"),
        logging.StreamHandler(_utf8_stream(sys.stdout)),
    ],
)
logger = logging.getLogger(__name__)

TRANSCRIPT_DIR = "data/transcripts"
FIGURE_DIR     = "outputs/figures"
RESULT_DIR     = "outputs/results"
COHERENCE_DIR  = "outputs/coherence"
for _d in [TRANSCRIPT_DIR, FIGURE_DIR, RESULT_DIR, COHERENCE_DIR]:
    os.makedirs(_d, exist_ok=True)

# Engine names used throughout
ENGINES = ["afrispeech_whisper", "mms", "lelapa"]


# =============================================================================
# TRANSCRIPT SEGMENTATION
# Topic models need multiple documents; we slide a word window over the
# full transcript to produce overlapping pseudo-documents.
# =============================================================================

def segment_transcript(transcript: str,
                       window_words: int = 150,
                       step_words: int = 75) -> list[str]:
    words = transcript.split()
    if not words:
        return []
    if len(words) <= window_words:
        return [transcript]

    segs, start = [], 0
    while start < len(words):
        segs.append(" ".join(words[start: start + window_words]))
        start += step_words
        if start + window_words // 2 > len(words):
            break
    return segs


def segment_all(transcripts_by_engine: dict[str, list[str]],
                window_words: int, step_words: int) -> dict[str, list[str]]:
    out = {}
    for engine, transcripts in transcripts_by_engine.items():
        all_segs = []
        for i, t in enumerate(transcripts):
            segs = segment_transcript(t, window_words, step_words)
            logger.info(f"  [{engine}] file {i+1}: "
                        f"{len(t.split())} words -> {len(segs)} segments")
            all_segs.extend(segs)

        if not all_segs:
            logger.warning(f"  [{engine}] No segments - transcript was empty")
        out[engine] = all_segs or [""]
        logger.info(f"  [{engine}] Total segments for topic modelling: {len(out[engine])}")
    return out


# =============================================================================
# STAGE 1: ASR
# =============================================================================

def stage_asr(args) -> dict[str, list[str]]:
    logger.info("=" * 60)
    logger.info("STAGE 1: ASR TRANSCRIPTION")
    logger.info("=" * 60)

    if args.mode == "demo":
        from demo_data import generate_demo_transcripts
        return generate_demo_transcripts(n_episodes=args.n_episodes)

    from asr_module import run_all_engines

    audio_files = sorted(
        glob.glob(os.path.join("data/audio", "*.mp3")) +
        glob.glob(os.path.join("data/audio", "*.wav")) +
        glob.glob(os.path.join("data/audio", "*.ogg")) +
        glob.glob(os.path.join("data/audio", "*.m4a"))
    )
    if not audio_files:
        logger.error("No audio files found in data/audio/. "
                     "Add files or use --mode demo")
        sys.exit(1)

    logger.info(f"Found {len(audio_files)} audio file(s).")
    logger.info("  Engine 1: AfriSpeech-Whisper  (intronhealth/afrispeech-whisper-medium-all)")
    logger.info("  Engine 2: MMS                 (facebook/mms-1b-all, tsn adapter)")
    logger.info("  Engine 3: Lelapa API          (Vulavula, tsn)")

    raw_results = run_all_engines(
        audio_files=audio_files,
        out_dir=TRANSCRIPT_DIR,
        lelapa_api_key=os.environ.get("LELAPA_API_KEY", "").strip() or None,
        skip_lelapa=args.skip_lelapa,
        skip_mms=args.skip_mms,
    )

    # Convert to plain transcript strings per engine
    transcripts_by_engine = {
        engine: [r.get("transcript", "") for r in results]
        for engine, results in raw_results.items()
    }

    for eng, trs in transcripts_by_engine.items():
        n_words = sum(len(t.split()) for t in trs)
        logger.info(f"  [{eng}] {n_words} total words across {len(trs)} file(s)")
        if n_words < 30:
            logger.warning(
                f"  [{eng}] Very few words. Possible reasons:\n"
                f"    - ASR model failed (check data/transcripts/*.json for error field)\n"
                f"    - Audio too short or too noisy\n"
                f"    - Model download incomplete"
            )

    logger.info("\n  Segmenting transcripts into topic-model documents ...")
    return segment_all(transcripts_by_engine, args.window_words, args.step_words)


# =============================================================================
# STAGE 2: PREPROCESSING
# =============================================================================

def stage_preprocessing(transcripts_by_engine: dict[str, list[str]]) -> dict:
    logger.info("=" * 60)
    logger.info("STAGE 2: TEXT PREPROCESSING")
    logger.info("=" * 60)

    from preprocessing import preprocess_transcripts

    preprocessed = {}
    for engine, transcripts in transcripts_by_engine.items():
        logger.info(f"\n  Engine: {engine.upper()}")
        transcripts = [t for t in transcripts if t.strip()]

        if not transcripts:
            logger.warning(f"  [{engine}] All transcripts empty - skipping")
            _empty = {"raw_tokens": [], "corrected_tokens": [],
                      "joined_strings": [], "vocab": set(), "correction_rate": 0}
            preprocessed[engine] = {"raw": _empty, "corrected": _empty}
            continue

        raw_prep = preprocess_transcripts(transcripts, apply_correction=False, min_freq=2)
        cor_prep = preprocess_transcripts(transcripts, apply_correction=True,  min_freq=2)
        preprocessed[engine] = {"raw": raw_prep, "corrected": cor_prep}

        out = {
            "raw_texts":       raw_prep["joined_strings"],
            "corrected_texts": cor_prep["joined_strings"],
            "correction_rate": cor_prep["correction_rate"],
        }
        with open(os.path.join(RESULT_DIR, f"preprocessed_{engine}.json"),
                  "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

        logger.info(f"  Correction rate: {cor_prep['correction_rate']:.2%}")

    return preprocessed


# =============================================================================
# STAGE 3: TOPIC MODELLING
# =============================================================================

def stage_topic_modelling(preprocessed: dict, n_topics: int = 7) -> dict:
    logger.info("=" * 60)
    logger.info("STAGE 3: TOPIC MODELLING  (LDA | BERTopic | NMF)")
    logger.info("=" * 60)

    from topic_modeling import run_all_topic_models

    all_results = {}
    for engine, prep_data in preprocessed.items():
        logger.info(f"\n  Engine: {engine.upper()}")
        cor         = prep_data.get("corrected", {})
        token_lists = cor.get("corrected_tokens", [])
        joined      = cor.get("joined_strings", [])

        valid = [(t, j) for t, j in zip(token_lists, joined) if len(t) > 2]
        if len(valid) < 3:
            logger.warning(
                f"  [{engine}] Only {len(valid)} non-empty segments (need >= 3). "
                f"Skipping topic modelling."
            )
            continue

        token_lists_f, joined_f = zip(*valid)
        k = min(n_topics, max(2, len(valid) // 2))
        if k != n_topics:
            logger.info(f"  [{engine}] n_topics capped: {n_topics} -> {k} "
                        f"(only {len(valid)} segments available)")

        results = run_all_topic_models(
            token_lists=list(token_lists_f),
            joined_strings=list(joined_f),
            n_topics=k,
        )
        for res in results.values():
            res["engine"] = engine
        all_results[engine] = results

        topics_out = {
            m: {"n_topics": r["n_topics"], "topics": r["topics"]}
            for m, r in results.items()
        }
        out_path = os.path.join(RESULT_DIR, f"topics_{engine}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(topics_out, f, ensure_ascii=False, indent=2)
        logger.info(f"  Saved topics -> {out_path}")

    return all_results


# =============================================================================
# STAGE 4: EVALUATION
# =============================================================================

def stage_evaluation(all_results: dict, preprocessed: dict,
                     n_topics: int = 7):
    logger.info("=" * 60)
    logger.info("STAGE 4: EVALUATION  (NPMI | UMass | Diversity)")
    logger.info("=" * 60)

    from evaluation import (evaluate_all_conditions, compute_error_propagation,
                             save_evaluation_results)
    from topic_modeling import run_lda

    token_lists_by_engine = {
        eng: data["corrected"]["corrected_tokens"]
        for eng, data in preprocessed.items()
        if data.get("corrected", {}).get("corrected_tokens")
    }

    eval_df = evaluate_all_conditions(all_results, token_lists_by_engine)
    # Print evaluation summary
    print("\n" + "="*65)
    print("  EVALUATION SUMMARY")
    print("="*65)
    print(eval_df.to_string(index=False, float_format="%.4f"))
    if not eval_df.empty:
        best = eval_df.loc[eval_df["npmi"].idxmax()]
        print(f"\n  * Best NPMI: {best['engine'].upper()} x {best['method'].upper()} ({best['npmi']:.4f})")
    print("="*65 + "\n")
    save_evaluation_results(eval_df,
                            os.path.join(COHERENCE_DIR, "coherence_scores.csv"))

    # Error propagation
    logger.info("\n  Error Propagation Analysis ...")
    raw_tokens = {eng: data["raw"]["raw_tokens"]
                  for eng, data in preprocessed.items()
                  if data.get("raw", {}).get("raw_tokens")}
    cor_tokens = {eng: data["corrected"]["corrected_tokens"]
                  for eng, data in preprocessed.items()
                  if data.get("corrected", {}).get("corrected_tokens")}

    def _lda_wrap(tl, n_topics=n_topics):
        k = min(n_topics, max(2, len(tl) // 2))
        return run_lda(tl, n_topics=k, passes=10)

    error_df = pd.DataFrame()
    if raw_tokens and cor_tokens:
        try:
            error_df = compute_error_propagation(
                raw_tokens, cor_tokens, _lda_wrap, n_topics
            )
            error_df.to_csv(
                os.path.join(COHERENCE_DIR, "error_propagation.csv"), index=False
            )
            print(error_df.to_string(index=False, float_format="%.4f"))
        except Exception as e:
            logger.warning(f"  Error propagation failed: {e}")

    return eval_df, error_df


# =============================================================================
# STAGE 5: VISUALISATION
# =============================================================================

def stage_visualisation(all_results, eval_df, error_df, preprocessed):
    logger.info("=" * 60)
    logger.info("STAGE 5: VISUALISATION")
    logger.info("=" * 60)

    from visualization import (plot_coherence_comparison, plot_diversity,
                                plot_error_propagation, save_pyldavis,
                                save_bertopic_visuals, plot_word_clouds,
                                plot_doc_topic_heatmap)

    if eval_df.empty:
        logger.warning("  Evaluation DataFrame empty - no charts to produce")
        return

    plot_coherence_comparison(eval_df, metric="npmi",
        out_path=os.path.join(FIGURE_DIR, "npmi_comparison.png"))
    plot_coherence_comparison(eval_df, metric="umass",
        out_path=os.path.join(FIGURE_DIR, "umass_comparison.png"))
    plot_diversity(eval_df,
        out_path=os.path.join(FIGURE_DIR, "diversity.png"))
    if not error_df.empty:
        plot_error_propagation(error_df,
            out_path=os.path.join(FIGURE_DIR, "error_propagation.png"))

    for engine, methods in all_results.items():
        n_segs     = len(preprocessed[engine]["corrected"]["joined_strings"])
        doc_labels = [f"Seg{i+1}" for i in range(n_segs)]

        for method, res in methods.items():
            if res.get("topics"):
                plot_word_clouds(res, out_dir=FIGURE_DIR)
            mat = res.get("doc_topic_matrix")
            if mat is not None and len(mat) > 0:
                plot_doc_topic_heatmap(
                    res, doc_labels=doc_labels,
                    out_path=os.path.join(FIGURE_DIR,
                                          f"heatmap_{engine}_{method}.png"))

        lda_res = methods.get("lda", {})
        if lda_res.get("model"):
            try:
                save_pyldavis(lda_res,
                    out_path=os.path.join(FIGURE_DIR, f"pyldavis_{engine}.html"))
            except Exception as e:
                logger.warning(f"  pyLDAvis failed [{engine}]: {e}")

        bert_res = methods.get("bertopic", {})
        if bert_res.get("model"):
            save_bertopic_visuals(bert_res,
                out_dir=os.path.join(FIGURE_DIR, f"bertopic_{engine}"))

    logger.info(f"  Figures saved -> {FIGURE_DIR}/")


# =============================================================================
# NATIVE SPEAKER TEMPLATE
# =============================================================================

def generate_native_speaker_template(all_results, eval_df):
    if eval_df.empty:
        return
    lines = [
        "=" * 70,
        "NATIVE SPEAKER QUALITATIVE REVIEW",
        "COS760 Group 37 - Setswana COVID-19 Topic Modelling",
        "Reviewers: Kagiso Tsiane & Shaun Seabo",
        "=" * 70, "",
        "For each topic: rate interpretability, identify theme, flag errors.",
        "Error codes: M=morpheme split, T=tonal confusion, E=English loanword,",
        "             S=code-switch error, R=repetition, N=none", "",
    ]
    for engine in eval_df["engine"].unique():
        sub         = eval_df[eval_df["engine"] == engine]
        best_method = sub.loc[sub["npmi"].idxmax(), "method"]
        res         = all_results.get(engine, {}).get(best_method, {})

        lines += [
            "=" * 70,
            f"ENGINE : {engine.upper()}",
            f"Method : {best_method.upper()}  (highest NPMI = {sub['npmi'].max():.4f})",
            "=" * 70,
        ]
        for t in res.get("topics", []):
            lines += [
                f"\nTopic {t['id']:02d}: {', '.join(t['words'][:10])}",
                "  Interpretability [ Clear / Partial / Unclear ]: ___",
                "  Theme: ___________________________________",
                "  Error types (codes above): ________________",
                "  Example error words: ______________________",
            ]
        lines.append("")

    out_path = os.path.join(RESULT_DIR, "native_speaker_review_template.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"  Native speaker template -> {out_path}")


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="COS760 Group 37 - Setswana ASR + Topic Modelling"
    )
    p.add_argument("--mode", choices=["real", "demo"], default="demo",
                   help="'real' = audio files in data/audio/,  'demo' = synthetic data")
    p.add_argument("--n-episodes", type=int, default=12,
                   help="Synthetic episodes in demo mode")
    p.add_argument("--n-topics",   type=int, default=7)
    p.add_argument("--skip-lelapa", action="store_true",
                   help="Skip Lelapa API (no key available)")
    p.add_argument("--skip-mms", action="store_true",
                   help="Skip MMS/wav2vec engine (slow on CPU)")
    p.add_argument("--window-words", type=int, default=150,
                   help="Words per transcript window for topic modelling docs")
    p.add_argument("--step-words",   type=int, default=75,
                   help="Word stride between windows (50%% overlap by default)")
    return p.parse_args()


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("  COS760 Group 37: Setswana ASR + Topic Modelling")
    logger.info("=" * 60)
    logger.info(f"  Mode      : {args.mode.upper()}")
    logger.info(f"  N topics  : {args.n_topics}")
    logger.info(f"  Skip Lelapa: {args.skip_lelapa}")
    logger.info(f"  Skip MMS  : {args.skip_mms}")

    transcripts   = stage_asr(args)
    preprocessed  = stage_preprocessing(transcripts)
    all_results   = stage_topic_modelling(preprocessed, n_topics=args.n_topics)

    if not all_results:
        logger.error(
            "No topic results produced. Check:\n"
            "  1. data/transcripts/*.json - did ASR produce text?\n"
            "  2. Run --mode demo to verify the full pipeline works.\n"
            "  3. Try --skip-mms if MMS is failing."
        )
        sys.exit(1)

    eval_df, error_df = stage_evaluation(
        all_results, preprocessed, n_topics=args.n_topics
    )
    stage_visualisation(all_results, eval_df, error_df, preprocessed)
    generate_native_speaker_template(all_results, eval_df)

    logger.info("\n" + "=" * 60)
    logger.info("  PIPELINE COMPLETE")
    logger.info(f"  Figures   : {FIGURE_DIR}/")
    logger.info(f"  Results   : {RESULT_DIR}/")
    logger.info(f"  Coherence : {COHERENCE_DIR}/")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()