#!/usr/bin/env python3
"""
pipeline.py
===========
COS760 NLP Project – Group 37
ASR + Topic Modelling for Setswana COVID-19 Podcasts

Run this script to execute the complete pipeline:
  1. ASR Transcription (Whisper / Lelapa / wav2vec)
  2. Text Preprocessing
  3. Topic Modelling (LDA / BERTopic / NMF)
  4. Coherence Evaluation (NPMI / UMass)
  5. Error Propagation Analysis
  6. Visualisations

Usage
-----
  # Full pipeline on real audio (ensure audio files are in data/audio/):
  python pipeline.py --mode real

  # Demo mode (synthetic data, no audio files needed):
  python pipeline.py --mode demo

  # Skip Lelapa (no API key yet):
  python pipeline.py --mode real --skip-lelapa

  # Skip wav2vec (too slow on CPU):
  python pipeline.py --mode real --skip-wav2vec
"""

import os
import sys
import json
import glob
import logging
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# ── Adjust path so we can import from src/ ─────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("pipeline.log"),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)

# ── Output directories ──────────────────────────────────────────
TRANSCRIPT_DIR = "data/transcripts"
FIGURE_DIR     = "outputs/figures"
RESULT_DIR     = "outputs/results"
COHERENCE_DIR  = "outputs/coherence"

for d in [TRANSCRIPT_DIR, FIGURE_DIR, RESULT_DIR, COHERENCE_DIR]:
    os.makedirs(d, exist_ok=True)


# ═══════════════════════════════════════════════════════════════
# STAGE 1: ASR TRANSCRIPTION
# ═══════════════════════════════════════════════════════════════

def stage_asr(args) -> dict[str, list[str]]:
    """
    Run ASR engines and return transcripts keyed by engine name.
    Returns: { "whisper": [str, ...], "lelapa": [...], "wav2vec": [...] }
    """
    logger.info("="*60)
    logger.info("STAGE 1: ASR TRANSCRIPTION")
    logger.info("="*60)

    if args.mode == "demo":
        from demo_data import generate_demo_transcripts
        return generate_demo_transcripts(n_episodes=args.n_episodes)

    # --- Real audio mode ---
    from asr_module import run_all_engines

    audio_files = sorted(glob.glob(os.path.join("data/audio", "*.mp3")) +
                         glob.glob(os.path.join("data/audio", "*.wav")) +
                         glob.glob(os.path.join("data/audio", "*.ogg")) +
                         glob.glob(os.path.join("data/audio", "*.m4a")))

    if not audio_files:
        logger.error("No audio files found in data/audio/. "
                     "Add audio files or run with --mode demo")
        sys.exit(1)

    logger.info(f"Found {len(audio_files)} audio file(s).")

    api_key = os.environ.get("LELAPA_API_KEY", "")

    raw_results = run_all_engines(
        audio_files=audio_files,
        output_dir=TRANSCRIPT_DIR,
        lelapa_api_key=api_key or None,
        skip_lelapa=args.skip_lelapa or (not api_key),
        skip_wav2vec=args.skip_wav2vec,
    )

    # Convert to plain strings (one per file per engine)
    return {
        engine: [r["transcript"] for r in results]
        for engine, results in raw_results.items()
    }


# ═══════════════════════════════════════════════════════════════
# STAGE 2: PREPROCESSING
# ═══════════════════════════════════════════════════════════════

def stage_preprocessing(transcripts_by_engine: dict[str, list[str]]) -> dict:
    """
    Preprocess all engine transcripts.
    Returns nested dict: { engine: preprocess_result_dict }
    """
    logger.info("="*60)
    logger.info("STAGE 2: TEXT PREPROCESSING")
    logger.info("="*60)

    from preprocessing import preprocess_transcripts

    preprocessed = {}
    for engine, transcripts in transcripts_by_engine.items():
        logger.info(f"\n  Engine: {engine.upper()}")

        # Run WITHOUT correction first (for error propagation baseline)
        raw_prep = preprocess_transcripts(
            transcripts,
            apply_correction=False,
        )

        # Run WITH correction (main pipeline)
        cor_prep = preprocess_transcripts(
            transcripts,
            apply_correction=True,
        )

        preprocessed[engine] = {
            "raw": raw_prep,
            "corrected": cor_prep,
        }

        # Save joined strings for inspection
        out = {
            "raw_texts": raw_prep["joined_strings"],
            "corrected_texts": cor_prep["joined_strings"],
            "correction_rate": cor_prep["correction_rate"],
        }
        with open(os.path.join(RESULT_DIR, f"preprocessed_{engine}.json"), "w") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

        logger.info(f"  Correction rate: {cor_prep['correction_rate']:.2%}")

    return preprocessed


# ═══════════════════════════════════════════════════════════════
# STAGE 3: TOPIC MODELLING
# ═══════════════════════════════════════════════════════════════

def stage_topic_modelling(preprocessed: dict, n_topics: int = 7) -> dict:
    """
    Run LDA, BERTopic, NMF for each engine's corrected transcripts.
    Returns: { engine: { method: result_dict } }
    """
    logger.info("="*60)
    logger.info("STAGE 3: TOPIC MODELLING")
    logger.info("="*60)

    from topic_modeling import run_all_topic_models

    all_results = {}
    for engine, prep_data in preprocessed.items():
        logger.info(f"\n  Engine: {engine.upper()}")
        cor = prep_data["corrected"]
        token_lists   = cor["corrected_tokens"]
        joined_strings = cor["joined_strings"]

        # Skip if no content
        if not any(joined_strings):
            logger.warning(f"  No content for {engine} – skipping")
            continue

        # Filter out empty documents
        valid_pairs = [(t, j) for t, j in zip(token_lists, joined_strings)
                       if len(t) > 3]
        if not valid_pairs:
            logger.warning(f"  All documents empty after filtering for {engine}")
            continue
        token_lists_f, joined_f = zip(*valid_pairs)

        results = run_all_topic_models(
            token_lists=list(token_lists_f),
            joined_strings=list(joined_f),
            n_topics=n_topics,
        )

        # Attach engine label to each result
        for method, res in results.items():
            res["engine"] = engine

        all_results[engine] = results

        # Save topic words to JSON for inspection / native speaker review
        topics_out = {}
        for method, res in results.items():
            topics_out[method] = {
                "n_topics": res["n_topics"],
                "topics": res["topics"],
            }
        with open(os.path.join(RESULT_DIR, f"topics_{engine}.json"), "w") as f:
            json.dump(topics_out, f, ensure_ascii=False, indent=2)

        logger.info(f"  Saved topics → {RESULT_DIR}/topics_{engine}.json")

    return all_results


# ═══════════════════════════════════════════════════════════════
# STAGE 4: EVALUATION
# ═══════════════════════════════════════════════════════════════

def stage_evaluation(all_results: dict, preprocessed: dict,
                     n_topics: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute coherence scores and error propagation analysis.
    Returns: (eval_df, error_df)
    """
    logger.info("="*60)
    logger.info("STAGE 4: EVALUATION")
    logger.info("="*60)

    from evaluation import (evaluate_all_conditions, compute_error_propagation,
                            save_evaluation_results, print_summary_table)
    from topic_modeling import run_lda

    # ── 4a. Coherence for all 9 conditions ─────────────────────
    token_lists_by_engine = {
        eng: data["corrected"]["corrected_tokens"]
        for eng, data in preprocessed.items()
    }
    eval_df = evaluate_all_conditions(all_results, token_lists_by_engine)
    print_summary_table(eval_df)
    save_evaluation_results(eval_df, os.path.join(COHERENCE_DIR, "coherence_scores.csv"))

    # ── 4b. Error propagation (using LDA as reference method) ──
    logger.info("\n  Error Propagation Analysis …")
    raw_tokens   = {eng: data["raw"]["raw_tokens"] for eng, data in preprocessed.items()}
    cor_tokens   = {eng: data["corrected"]["corrected_tokens"] for eng, data in preprocessed.items()}

    def _lda_wrapper(token_lists, n_topics=n_topics):
        return run_lda(token_lists, n_topics=n_topics, passes=10)

    error_df = compute_error_propagation(raw_tokens, cor_tokens, _lda_wrapper, n_topics)
    error_df.to_csv(os.path.join(COHERENCE_DIR, "error_propagation.csv"), index=False)
    logger.info(f"  Error propagation saved → {COHERENCE_DIR}/error_propagation.csv")
    print(error_df.to_string(index=False, float_format="%.4f"))

    return eval_df, error_df


# ═══════════════════════════════════════════════════════════════
# STAGE 5: VISUALISATION
# ═══════════════════════════════════════════════════════════════

def stage_visualisation(all_results: dict, eval_df: pd.DataFrame,
                         error_df: pd.DataFrame,
                         preprocessed: dict) -> None:
    """Produce all charts and save to outputs/figures/."""
    logger.info("="*60)
    logger.info("STAGE 5: VISUALISATION")
    logger.info("="*60)

    from visualization import (
        plot_coherence_comparison, plot_diversity, plot_error_propagation,
        save_pyldavis, save_bertopic_visuals, plot_word_clouds,
        plot_doc_topic_heatmap, print_summary_table,
    )

    # 1. NPMI comparison chart
    plot_coherence_comparison(
        eval_df, metric="npmi",
        out_path=os.path.join(FIGURE_DIR, "npmi_comparison.png")
    )

    # 2. UMass comparison chart
    plot_coherence_comparison(
        eval_df, metric="umass",
        out_path=os.path.join(FIGURE_DIR, "umass_comparison.png")
    )

    # 3. Diversity chart
    plot_diversity(
        eval_df,
        out_path=os.path.join(FIGURE_DIR, "diversity.png")
    )

    # 4. Error propagation chart
    if not error_df.empty:
        plot_error_propagation(
            error_df,
            out_path=os.path.join(FIGURE_DIR, "error_propagation.png")
        )

    # Per-engine visualisations
    doc_labels_map = {}
    for engine, methods in all_results.items():
        # Try to infer doc labels from preprocessed data
        n_docs = len(preprocessed[engine]["corrected"]["joined_strings"])
        doc_labels_map[engine] = [f"Ep{i+1}" for i in range(n_docs)]

        # 5. Word clouds for LDA
        lda_res = methods.get("lda", {})
        if lda_res.get("topics"):
            plot_word_clouds(lda_res, out_dir=FIGURE_DIR)

        # 6. Word clouds for BERTopic
        bert_res = methods.get("bertopic", {})
        if bert_res.get("topics"):
            plot_word_clouds(bert_res, out_dir=FIGURE_DIR)

        # 7. Document-topic heatmaps
        for method, res in methods.items():
            if res.get("doc_topic_matrix") is not None and len(res["doc_topic_matrix"]) > 0:
                plot_doc_topic_heatmap(
                    res,
                    doc_labels=doc_labels_map[engine],
                    out_path=os.path.join(FIGURE_DIR, f"heatmap_{engine}_{method}.png")
                )

        # 8. pyLDAvis interactive (LDA only)
        if lda_res.get("model") is not None:
            try:
                save_pyldavis(
                    lda_res,
                    out_path=os.path.join(FIGURE_DIR, f"pyldavis_{engine}.html")
                )
            except Exception as e:
                logger.warning(f"  pyLDAvis failed for {engine}: {e}")

        # 9. BERTopic HTML visuals
        if bert_res.get("model") is not None:
            save_bertopic_visuals(
                bert_res,
                out_dir=os.path.join(FIGURE_DIR, f"bertopic_{engine}")
            )

    logger.info(f"\n  All figures saved to: {FIGURE_DIR}/")


# ═══════════════════════════════════════════════════════════════
# NATIVE SPEAKER REVIEW TEMPLATE
# ═══════════════════════════════════════════════════════════════

def generate_native_speaker_template(all_results: dict,
                                      eval_df: pd.DataFrame) -> str:
    """
    Generate a structured text file for native speaker qualitative review.
    """
    logger.info("  Generating native speaker review template …")

    # Find best configuration per engine (highest NPMI)
    out_lines = [
        "=" * 70,
        "NATIVE SPEAKER QUALITATIVE REVIEW TEMPLATE",
        "COS760 Group 37 – Setswana COVID-19 Topic Modelling",
        "=" * 70,
        "",
        "INSTRUCTIONS:",
        "For each topic below, please:",
        "  1. Rate interpretability: Clear / Partial / Unclear",
        "  2. Identify the dominant theme (if any)",
        "  3. Flag any meaning-obscuring errors",
        "  4. Note any culturally relevant observations",
        "",
    ]

    for engine in eval_df["engine"].unique():
        engine_df = eval_df[eval_df["engine"] == engine]
        best_method = engine_df.loc[engine_df["npmi"].idxmax(), "method"]
        best_result = all_results.get(engine, {}).get(best_method, {})

        out_lines += [
            "=" * 70,
            f"ENGINE: {engine.upper()}  |  Best Method: {best_method.upper()}",
            f"NPMI: {engine_df['npmi'].max():.4f}",
            "=" * 70,
        ]

        for topic in best_result.get("topics", []):
            words_str = ", ".join(topic["words"][:10])
            out_lines += [
                f"\nTopic {topic['id']:02d}:",
                f"  Words: {words_str}",
                f"  Interpretability: [ Clear / Partial / Unclear ]",
                f"  Theme: ___________________________________",
                f"  Errors flagged: ___________________________",
                f"  Notes: ___________________________________",
            ]

        out_lines.append("")

    content = "\n".join(out_lines)
    out_path = os.path.join(RESULT_DIR, "native_speaker_review_template.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)

    logger.info(f"  Native speaker template saved → {out_path}")
    return out_path


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="COS760 Group 37 – ASR + Topic Modelling Pipeline")
    p.add_argument("--mode", choices=["real", "demo"], default="demo",
                   help="'real' uses audio files in data/audio/, 'demo' uses synthetic data")
    p.add_argument("--n-episodes", type=int, default=12,
                   help="Number of synthetic episodes in demo mode")
    p.add_argument("--n-topics", type=int, default=7,
                   help="Number of topics for NMF and LDA fixed mode")
    p.add_argument("--skip-lelapa", action="store_true",
                   help="Skip Lelapa ASR (use if no API key)")
    p.add_argument("--skip-wav2vec", action="store_true",
                   help="Skip wav2vec (use if running on CPU without time)")
    return p.parse_args()


def main():
    args = parse_args()

    logger.info("╔══════════════════════════════════════════════════════════╗")
    logger.info("║  COS760 Group 37: ASR + Topic Modelling Pipeline         ║")
    logger.info("║  Setswana COVID-19 Podcasts                               ║")
    logger.info("╚══════════════════════════════════════════════════════════╝")
    logger.info(f"  Mode: {args.mode.upper()}")
    logger.info(f"  N topics: {args.n_topics}")

    # Stage 1: ASR
    transcripts = stage_asr(args)

    # Stage 2: Preprocessing
    preprocessed = stage_preprocessing(transcripts)

    # Stage 3: Topic Modelling
    all_results = stage_topic_modelling(preprocessed, n_topics=args.n_topics)

    if not all_results:
        logger.error("No topic modelling results produced. Exiting.")
        sys.exit(1)

    # Stage 4: Evaluation
    eval_df, error_df = stage_evaluation(all_results, preprocessed, n_topics=args.n_topics)

    # Stage 5: Visualisation
    stage_visualisation(all_results, eval_df, error_df, preprocessed)

    # Native speaker template
    generate_native_speaker_template(all_results, eval_df)

    logger.info("\n" + "="*60)
    logger.info("  PIPELINE COMPLETE")
    logger.info(f"  Figures:      {FIGURE_DIR}/")
    logger.info(f"  Results:      {RESULT_DIR}/")
    logger.info(f"  Coherence:    {COHERENCE_DIR}/")
    logger.info(f"  Log file:     pipeline.log")
    logger.info("="*60)


if __name__ == "__main__":
    main()