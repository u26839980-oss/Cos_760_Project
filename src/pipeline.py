#!/usr/bin/env python3
"""
pipeline.py  -  COS760 Group 37
================================
Runs the full ASR + Topic Modelling pipeline for either:
  --language tsn   (Setswana)
  --language zul   (isiZulu)

QUICKSTART - just copy these commands:

  # Test everything works (no audio needed):
  python pipeline.py --mode demo

  # Setswana audio, skip slow engines:
  python pipeline.py --mode real --language tsn --skip-mms

  # Setswana audio, all engines:
  python pipeline.py --mode real --language tsn

  # isiZulu audio, no Lelapa key yet:
  python pipeline.py --mode real --language zul --skip-lelapa

  # isiZulu audio, all engines (need LELAPA_API_KEY set):
  python pipeline.py --mode real --language zul
"""

import io, os, sys, json, glob, logging, argparse, warnings
warnings.filterwarnings("ignore")
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# ── Windows-safe UTF-8 output ──────────────────────────────────────────────
def _utf8(stream):
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
        logging.StreamHandler(_utf8(sys.stdout)),
    ],
)
logger = logging.getLogger(__name__)

DIRS = {
    "transcripts": "data/transcripts",
    "figures":     "outputs/figures",
    "results":     "outputs/results",
    "coherence":   "outputs/coherence",
}
for d in DIRS.values():
    os.makedirs(d, exist_ok=True)


# ── Segment a transcript into overlapping word-windows ─────────────────────
# Topic models need multiple documents. One long transcript = one document
# which is not enough. We slide a window to get 10-30 pseudo-documents.

def segment(text: str, window: int = 150, step: int = 75) -> list[str]:
    words = text.split()
    if not words:
        return []
    if len(words) <= window:
        return [text]
    out, i = [], 0
    while i < len(words):
        out.append(" ".join(words[i: i + window]))
        i += step
        if i + window // 2 > len(words):
            break
    return out


def segment_all(by_engine: dict, window: int, step: int) -> dict:
    out = {}
    for eng, texts in by_engine.items():
        segs = []
        for j, t in enumerate(texts):
            s = segment(t, window, step)
            logger.info(f"  [{eng}] file {j+1}: {len(t.split())} words -> {len(s)} segments")
            segs.extend(s)
        out[eng] = segs or [""]
        logger.info(f"  [{eng}] Total segments: {len(out[eng])}")
    return out


# ===========================================================================
# STAGE 1 - ASR
# ===========================================================================

def stage_asr(args) -> dict:
    logger.info("=" * 60)
    logger.info("STAGE 1: ASR TRANSCRIPTION")
    logger.info(f"  Language: {args.language.upper()}")
    logger.info("=" * 60)

    if args.mode == "demo":
        from demo_data import generate_demo_transcripts
        return generate_demo_transcripts(n_episodes=args.n_episodes)

    from asr_module import run_all_engines

    audio_files = sorted(
        glob.glob("data/audio/*.mp3") + glob.glob("data/audio/*.wav") +
        glob.glob("data/audio/*.ogg") + glob.glob("data/audio/*.m4a")
    )
    if not audio_files:
        logger.error("No files in data/audio/. Add audio or use --mode demo")
        sys.exit(1)

    logger.info(f"  {len(audio_files)} audio file(s) found.")

    raw = run_all_engines(
        audio_files=audio_files,
        out_dir=DIRS["transcripts"],
        language=args.language,
        lelapa_api_key=os.environ.get("LELAPA_API_KEY", "").strip() or None,
        skip_lelapa=args.skip_lelapa,
        skip_mms=args.skip_mms,
        skip_engine3=args.skip_engine3,
    )

    # Convert to plain strings
    by_engine = {eng: [r.get("transcript", "") for r in lst]
                 for eng, lst in raw.items()}

    for eng, texts in by_engine.items():
        w = sum(len(t.split()) for t in texts)
        logger.info(f"  [{eng}] {w} words total")
        if w < 20:
            logger.warning(
                f"  [{eng}] Very few words - check data/transcripts/ for errors.\n"
                f"           Did the model download finish? Is the audio long enough?"
            )

    logger.info("\n  Segmenting into topic-model documents ...")
    return segment_all(by_engine, args.window_words, args.step_words)


# ===========================================================================
# STAGE 2 - PREPROCESSING
# ===========================================================================

def stage_preprocessing(by_engine: dict) -> dict:
    logger.info("=" * 60)
    logger.info("STAGE 2: PREPROCESSING")
    logger.info("=" * 60)

    from preprocessing import preprocess_transcripts

    out = {}
    for eng, texts in by_engine.items():
        logger.info(f"\n  [{eng}]")
        texts = [t for t in texts if t.strip()]
        if not texts:
            logger.warning(f"  [{eng}] No content - skipping")
            _e = {"raw_tokens":[], "corrected_tokens":[], "joined_strings":[], "vocab":set(), "correction_rate":0}
            out[eng] = {"raw": _e, "corrected": _e}
            continue

        raw = preprocess_transcripts(texts, apply_correction=False, min_freq=2)
        cor = preprocess_transcripts(texts, apply_correction=True,  min_freq=2)
        out[eng] = {"raw": raw, "corrected": cor}

        json.dump(
            {"raw_texts": raw["joined_strings"],
             "corrected_texts": cor["joined_strings"],
             "correction_rate": cor["correction_rate"]},
            open(os.path.join(DIRS["results"], f"preprocessed_{eng}.json"),
                 "w", encoding="utf-8"),
            ensure_ascii=False, indent=2
        )
        logger.info(f"  Correction rate: {cor['correction_rate']:.2%}")

    return out


# ===========================================================================
# STAGE 3 - TOPIC MODELLING
# ===========================================================================

def stage_topic_modelling(preprocessed: dict, n_topics: int) -> dict:
    logger.info("=" * 60)
    logger.info("STAGE 3: TOPIC MODELLING  (LDA | BERTopic | NMF)")
    logger.info("=" * 60)

    from topic_modeling import run_all_topic_models
    all_res = {}

    for eng, data in preprocessed.items():
        logger.info(f"\n  [{eng}]")
        tokens = data["corrected"]["corrected_tokens"]
        joined = data["corrected"]["joined_strings"]
        valid  = [(t, j) for t, j in zip(tokens, joined) if len(t) > 2]

        if len(valid) < 3:
            logger.warning(
                f"  [{eng}] Only {len(valid)} non-empty segments. "
                f"Need >= 3 for topic models. Skipping.\n"
                f"  This means ASR produced very little text. "
                f"Check data/transcripts/*.json"
            )
            continue

        tok_f, join_f = zip(*valid)
        k = min(n_topics, max(2, len(valid) // 2))
        if k != n_topics:
            logger.info(f"  [{eng}] Capping topics {n_topics} -> {k} "
                        f"({len(valid)} segments)")

        res = run_all_topic_models(list(tok_f), list(join_f), n_topics=k)
        for r in res.values():
            r["engine"] = eng
        all_res[eng] = res

        out_path = os.path.join(DIRS["results"], f"topics_{eng}.json")
        json.dump(
            {m: {"n_topics": r["n_topics"], "topics": r["topics"]}
             for m, r in res.items()},
            open(out_path, "w", encoding="utf-8"),
            ensure_ascii=False, indent=2
        )
        logger.info(f"  Topics saved -> {out_path}")

    return all_res


# ===========================================================================
# STAGE 4 - EVALUATION
# ===========================================================================

def stage_evaluation(all_res: dict, preprocessed: dict, n_topics: int):
    logger.info("=" * 60)
    logger.info("STAGE 4: EVALUATION  (NPMI | UMass | Diversity)")
    logger.info("=" * 60)

    from evaluation import (evaluate_all_conditions, compute_error_propagation,
                             save_evaluation_results, print_summary_table)
    from topic_modeling import run_lda

    tok_by_eng = {e: d["corrected"]["corrected_tokens"]
                  for e, d in preprocessed.items()
                  if d.get("corrected", {}).get("corrected_tokens")}

    eval_df = evaluate_all_conditions(all_res, tok_by_eng)
    print_summary_table(eval_df)
    save_evaluation_results(eval_df,
        os.path.join(DIRS["coherence"], "coherence_scores.csv"))

    # Error propagation
    raw_tok = {e: d["raw"]["raw_tokens"]
               for e, d in preprocessed.items()
               if d.get("raw", {}).get("raw_tokens")}
    cor_tok = {e: d["corrected"]["corrected_tokens"]
               for e, d in preprocessed.items()
               if d.get("corrected", {}).get("corrected_tokens")}

    def _lda(tl, n_topics=n_topics):
        k = min(n_topics, max(2, len(tl) // 2))
        return run_lda(tl, n_topics=k, passes=10)

    err_df = pd.DataFrame()
    try:
        err_df = compute_error_propagation(raw_tok, cor_tok, _lda, n_topics)
        err_df.to_csv(os.path.join(DIRS["coherence"], "error_propagation.csv"),
                      index=False)
        print(err_df.to_string(index=False, float_format="%.4f"))
    except Exception as e:
        logger.warning(f"  Error propagation skipped: {e}")

    return eval_df, err_df


# ===========================================================================
# STAGE 5 - VISUALISATION
# ===========================================================================

def stage_visualisation(all_res, eval_df, err_df, preprocessed):
    logger.info("=" * 60)
    logger.info("STAGE 5: VISUALISATION")
    logger.info("=" * 60)

    from visualization import (plot_coherence_comparison, plot_diversity,
                            plot_error_propagation, save_pyldavis,
                            save_bertopic_visuals, plot_word_clouds,
                            plot_doc_topic_heatmap)
    if eval_df.empty:
        logger.warning("  Nothing to plot.")
        return

    plot_coherence_comparison(eval_df, "npmi",
        out_path=os.path.join(DIRS["figures"], "npmi_comparison.png"))
    plot_coherence_comparison(eval_df, "umass",
        out_path=os.path.join(DIRS["figures"], "umass_comparison.png"))
    plot_diversity(eval_df,
        out_path=os.path.join(DIRS["figures"], "diversity.png"))
    if not err_df.empty:
        plot_error_propagation(err_df,
            out_path=os.path.join(DIRS["figures"], "error_propagation.png"))

    for eng, methods in all_res.items():
        n = len(preprocessed[eng]["corrected"]["joined_strings"])
        labels = [f"Seg{i+1}" for i in range(n)]

        for method, res in methods.items():
            if res.get("topics"):
                plot_word_clouds(res, out_dir=DIRS["figures"])
            mat = res.get("doc_topic_matrix")
            if mat is not None and len(mat) > 0:
                plot_doc_topic_heatmap(res, doc_labels=labels,
                    out_path=os.path.join(DIRS["figures"],
                                          f"heatmap_{eng}_{method}.png"))

        if methods.get("lda", {}).get("model"):
            try:
                save_pyldavis(methods["lda"],
                    os.path.join(DIRS["figures"], f"pyldavis_{eng}.html"))
            except Exception as e:
                logger.warning(f"  pyLDAvis failed: {e}")

        if methods.get("bertopic", {}).get("model"):
            save_bertopic_visuals(methods["bertopic"],
                os.path.join(DIRS["figures"], f"bertopic_{eng}"))

    logger.info(f"  Figures saved -> {DIRS['figures']}/")


# ===========================================================================
# NATIVE SPEAKER TEMPLATE
# ===========================================================================

def native_speaker_template(all_res, eval_df):
    if eval_df.empty:
        return
    lines = [
        "=" * 70,
        "NATIVE SPEAKER REVIEW - COS760 Group 37",
        "Reviewers: Kagiso Tsiane & Shaun Seabo",
        "=" * 70, "",
        "For each topic list:",
        "  Interpretability: Clear / Partial / Unclear",
        "  Theme: short label (e.g. 'vaccination campaign')",
        "  Errors: M=morpheme split, T=tonal, E=English loanword,",
        "          S=code-switch, R=repetition, N=none", "",
    ]
    for eng in eval_df["engine"].unique():
        sub = eval_df[eval_df["engine"] == eng]
        best = sub.loc[sub["npmi"].idxmax(), "method"]
        res  = all_res.get(eng, {}).get(best, {})
        lines += [
            "=" * 70,
            f"ENGINE : {eng}",
            f"Method : {best.upper()}  (NPMI = {sub['npmi'].max():.4f})",
            "=" * 70,
        ]
        for t in res.get("topics", []):
            lines += [
                f"\nTopic {t['id']:02d}: {', '.join(t['words'][:10])}",
                "  Interpretability: ___",
                "  Theme: ___________",
                "  Errors: __________",
            ]
        lines.append("")

    path = os.path.join(DIRS["results"], "native_speaker_review.txt")
    open(path, "w", encoding="utf-8").write("\n".join(lines))
    logger.info(f"  Native speaker template -> {path}")


# ===========================================================================
# MAIN
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode",     choices=["real", "demo"], default="demo")
    p.add_argument("--language", choices=["tsn", "zul"],   default="tsn",
                   help="tsn=Setswana  zul=isiZulu")
    p.add_argument("--n-episodes",   type=int, default=12)
    p.add_argument("--n-topics",     type=int, default=7)
    p.add_argument("--skip-lelapa",  action="store_true",
                   help="Skip Lelapa API (isiZulu only)")
    p.add_argument("--skip-mms",     action="store_true",
                   help="Skip MMS engine (slow on CPU)")
    p.add_argument("--skip-engine3", action="store_true",
                   help="Skip Engine 3 entirely")
    p.add_argument("--window-words", type=int, default=150)
    p.add_argument("--step-words",   type=int, default=75)
    return p.parse_args()


def main():
    args = parse_args()
    logger.info("=" * 60)
    logger.info("  COS760 Group 37 - ASR + Topic Modelling")
    logger.info(f"  Language : {args.language.upper()}")
    logger.info(f"  Mode     : {args.mode.upper()}")
    logger.info("=" * 60)

    by_engine    = stage_asr(args)
    preprocessed = stage_preprocessing(by_engine)
    all_res      = stage_topic_modelling(preprocessed, args.n_topics)

    if not all_res:
        logger.error(
            "No topic results. Possible causes:\n"
            "  1. ASR transcripts are empty - check data/transcripts/*.json\n"
            "  2. Audio file too short (needs 30+ seconds)\n"
            "  3. Model failed to download - check internet and try again\n"
            "  4. Run --mode demo to confirm the pipeline itself works"
        )
        sys.exit(1)

    eval_df, err_df = stage_evaluation(all_res, preprocessed, args.n_topics)
    stage_visualisation(all_res, eval_df, err_df, preprocessed)
    native_speaker_template(all_res, eval_df)

    logger.info("\n" + "=" * 60)
    logger.info("  DONE. Check outputs/ for all results.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()