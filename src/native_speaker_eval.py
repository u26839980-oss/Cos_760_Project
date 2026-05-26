"""
native_speaker_eval.py
======================
Interactive native speaker evaluation tool.

Kagiso Tsiane & Shaun Seabo can use this to:
  1. Load topic results from any run
  2. Review topic words in a structured format
  3. Rate interpretability and document errors
  4. Export completed evaluation to CSV

Usage
-----
  python src/native_speaker_eval.py --engine whisper --method lda
  python src/native_speaker_eval.py --input src/outputs/results/topics_whisper.json
"""

import os
import sys
import json
import argparse
import csv
from datetime import datetime

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SRC_DIR, "outputs", "results")


# ─────────────────────────────────────────────────────────────
# RATING RUBRICS
# ─────────────────────────────────────────────────────────────

INTERPRETABILITY_SCALE = {
    "3": "Clear – Topic theme is obvious from the keywords alone",
    "2": "Partial – Some keywords make sense together, but topic is vague",
    "1": "Unclear – Keywords appear random / unrelated",
    "0": "Noise – Keywords appear to be pure ASR errors, no discernible meaning",
}

ERROR_TYPES = {
    "M": "Morpheme fragmentation (agglutinative word split incorrectly)",
    "T": "Tonal confusion (similar-sounding words confused)",
    "E": "English loan-word misrecognised",
    "S": "Code-switching error (language boundary missed)",
    "R": "Repetition / disfluency retained as content word",
    "N": "None – no significant errors observed",
}


def load_topics(json_path: str) -> dict:
    """Load topics from a saved JSON file."""
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def interactive_review(engine: str, method: str, topics: list[dict],
                       reviewer_name: str) -> list[dict]:
    """
    Walk the reviewer through each topic interactively.

    Parameters
    ----------
    engine        : ASR engine name
    method        : topic method name
    topics        : list of topic dicts
    reviewer_name : name of native speaker reviewer

    Returns
    -------
    list of completed review records
    """
    print("\n" + "="*65)
    print(f"  NATIVE SPEAKER REVIEW – {engine.upper()} × {method.upper()}")
    print(f"  Reviewer: {reviewer_name}")
    print("="*65)
    print("\nScoring key:")
    for k, v in INTERPRETABILITY_SCALE.items():
        print(f"  {k} = {v}")
    print("\nError type codes:")
    for k, v in ERROR_TYPES.items():
        print(f"  {k} = {v}")

    reviews = []
    for topic in topics:
        tid = topic["id"]
        words = topic["words"]
        weights = topic.get("weights", [1.0]*len(words))

        print(f"\n{'─'*65}")
        print(f"  TOPIC {tid:02d}")
        print(f"{'─'*65}")
        print("  Top words (by weight):")
        for w, wt in zip(words[:10], weights[:10]):
            print(f"    {w:<25} ({wt:.4f})")

        # Interpretability score
        while True:
            score = input("\n  Interpretability score [0-3]: ").strip()
            if score in INTERPRETABILITY_SCALE:
                break
            print("  Please enter 0, 1, 2, or 3")

        # Theme label
        theme = input("  Theme label (brief phrase, e.g. 'vaccination campaign'): ").strip()

        # Error types (can enter multiple, comma-separated)
        print(f"\n  Error types: {list(ERROR_TYPES.keys())} (comma-separated, or N)")
        while True:
            errors_input = input("  Error types observed: ").upper().strip()
            error_codes = [e.strip() for e in errors_input.split(",")]
            if all(e in ERROR_TYPES for e in error_codes):
                break
            print(f"  Valid codes: {list(ERROR_TYPES.keys())}")

        # Specific error examples
        error_examples = input("  Example error words (optional, comma-separated): ").strip()

        # Free notes
        notes = input("  Additional notes (press Enter to skip): ").strip()

        record = {
            "timestamp": datetime.now().isoformat(),
            "reviewer": reviewer_name,
            "engine": engine,
            "method": method,
            "topic_id": tid,
            "top_words": ", ".join(words[:10]),
            "interpretability_score": int(score),
            "interpretability_label": INTERPRETABILITY_SCALE[score],
            "theme": theme,
            "error_types": ", ".join(error_codes),
            "error_examples": error_examples,
            "notes": notes,
        }
        reviews.append(record)
        print(f"  ✓ Topic {tid:02d} recorded.")

    return reviews


def save_reviews(reviews: list[dict], out_path: str):
    """Save completed reviews to CSV."""
    if not reviews:
        return
    fieldnames = list(reviews[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(reviews)
    print(f"\n  Reviews saved → {out_path}")


def compute_review_stats(reviews: list[dict]) -> dict:
    """Compute summary statistics from completed reviews."""
    if not reviews:
        return {}

    scores = [r["interpretability_score"] for r in reviews]
    avg = sum(scores) / len(scores)
    clear = sum(1 for s in scores if s == 3)
    partial = sum(1 for s in scores if s == 2)
    unclear = sum(1 for s in scores if s <= 1)

    all_errors = []
    for r in reviews:
        all_errors.extend(r["error_types"].split(", "))

    error_freq = {}
    for e in all_errors:
        if e and e != "N":
            error_freq[e] = error_freq.get(e, 0) + 1

    return {
        "n_topics_reviewed": len(reviews),
        "avg_interpretability": round(avg, 2),
        "clear_topics": clear,
        "partial_topics": partial,
        "unclear_topics": unclear,
        "most_common_errors": sorted(error_freq.items(), key=lambda x: -x[1]),
    }


def main():
    parser = argparse.ArgumentParser(description="Native Speaker Evaluation Tool")
    parser.add_argument("--input", default=None,
                        help="Path to topics JSON file (e.g. src/outputs/results/topics_whisper.json)")
    parser.add_argument("--engine", default="whisper",
                        help="ASR engine name (if not using --input)")
    parser.add_argument("--method", default="lda",
                        help="Topic method (lda / bertopic / nmf)")
    parser.add_argument("--out", default=None,
                        help="Output CSV path (default: outputs/results/review_<engine>_<method>.csv)")
    args = parser.parse_args()

    # Load topics
    if args.input:
        topics_data = load_topics(args.input)
        # Try to infer engine from filename
        engine = os.path.basename(args.input).replace("topics_", "").replace(".json", "")
    else:
        json_path = os.path.join(RESULTS_DIR, f"topics_{args.engine}.json")
        if not os.path.exists(json_path):
            print(f"  Error: {json_path} not found. Run pipeline.py first.")
            sys.exit(1)
        topics_data = load_topics(json_path)
        engine = args.engine

    method = args.method
    if method not in topics_data:
        print(f"  Method '{method}' not found in results. Available: {list(topics_data.keys())}")
        sys.exit(1)

    topics = topics_data[method]["topics"]

    # Get reviewer name
    reviewer = input("\n  Your name (for records): ").strip() or "Anonymous"

    # Run interactive review
    reviews = interactive_review(engine, method, topics, reviewer)

    # Save
    out_path = args.out or os.path.join(RESULTS_DIR, f"review_{engine}_{method}.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    save_reviews(reviews, out_path)

    # Print stats
    stats = compute_review_stats(reviews)
    print("\n" + "="*65)
    print("  REVIEW SUMMARY")
    print("="*65)
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print("="*65)


if __name__ == "__main__":
    main()
