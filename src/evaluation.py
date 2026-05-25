"""
evaluation.py
=============
Quantitative evaluation of topic models.

Metrics implemented:
  1. NPMI coherence  (c_npmi via Gensim)  – primary metric
  2. UMass coherence (u_mass via Gensim)  – secondary metric
  3. Topic diversity – fraction of unique words across all topic top-words
  4. Error propagation analysis – NPMI delta before/after correction

All results are stored in a pandas DataFrame for easy plotting.
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# 1.  SINGLE COHERENCE SCORE
# ─────────────────────────────────────────────────────────────

def compute_coherence(topics: list[dict],
                      token_lists: list[list[str]],
                      coherence: str = "c_npmi",
                      topn: int = 10) -> float:
    """
    Compute NPMI or UMass coherence for a list of topics.

    Parameters
    ----------
    topics       : list of {"id", "words", "weights"} dicts
    token_lists  : list of token lists (the reference corpus)
    coherence    : "c_npmi" | "u_mass" | "c_v" | "c_uci"
    topn         : number of top words to use per topic

    Returns
    -------
    mean coherence score (float)
    """
    from gensim import corpora
    from gensim.models.coherencemodel import CoherenceModel

    if not topics:
        logger.warning("  No topics to evaluate – returning 0.0")
        return 0.0

    # Rebuild a minimal gensim Dictionary from the token lists
    dictionary = corpora.Dictionary(token_lists)

    topic_word_lists = [t["words"][:topn] for t in topics]

    # Filter topic words to those in the dictionary
    topic_word_lists = [
        [w for w in wl if w in dictionary.token2id]
        for wl in topic_word_lists
    ]
    topic_word_lists = [wl for wl in topic_word_lists if len(wl) >= 2]

    if not topic_word_lists:
        logger.warning("  All topic words filtered from dictionary – returning 0.0")
        return 0.0

    corpus = [dictionary.doc2bow(doc) for doc in token_lists]

    if coherence in ("u_mass",):
        # u_mass only needs corpus + dictionary (no texts needed)
        cm = CoherenceModel(
            topics=topic_word_lists,
            corpus=corpus,
            dictionary=dictionary,
            coherence=coherence,
            processes=1,
        )
    else:
        # c_npmi, c_v, c_uci need raw texts
        cm = CoherenceModel(
            topics=topic_word_lists,
            texts=token_lists,
            dictionary=dictionary,
            coherence=coherence,
            processes=1,
        )

    score = cm.get_coherence()
    return float(score)


# ─────────────────────────────────────────────────────────────
# 2.  TOPIC DIVERSITY
# ─────────────────────────────────────────────────────────────

def compute_topic_diversity(topics: list[dict], topn: int = 10) -> float:
    """
    Topic diversity = unique words / (n_topics × topn).

    A score of 1.0 means all top words are unique across topics.
    A low score means topics are redundant.

    Parameters
    ----------
    topics : list of topic dicts
    topn   : number of top words per topic

    Returns
    -------
    diversity score in [0, 1]
    """
    if not topics:
        return 0.0
    all_words = [w for t in topics for w in t["words"][:topn]]
    unique_words = set(all_words)
    diversity = len(unique_words) / max(len(all_words), 1)
    return float(diversity)


# ─────────────────────────────────────────────────────────────
# 3.  FULL EVALUATION GRID  (3 ASR × 3 Topic models = 9 conditions)
# ─────────────────────────────────────────────────────────────

def evaluate_all_conditions(
    topic_results: dict[str, dict[str, dict]],
    token_lists_by_engine: dict[str, list[list[str]]],
    topn: int = 10,
) -> pd.DataFrame:
    """
    Compute NPMI, UMass, and diversity for all 9 (ASR × method) conditions.

    Parameters
    ----------
    topic_results : nested dict
        {
          "whisper":  {"lda": {...}, "bertopic": {...}, "nmf": {...}},
          "lelapa":   {"lda": {...}, "bertopic": {...}, "nmf": {...}},
          "wav2vec":  {"lda": {...}, "bertopic": {...}, "nmf": {...}},
        }
    token_lists_by_engine : {engine: list of token lists}
        Used as reference corpus for coherence computation.
    topn : top words per topic used in coherence computation

    Returns
    -------
    pd.DataFrame with columns:
      engine, method, n_topics, npmi, umass, diversity
    """
    records = []
    for engine, methods in topic_results.items():
        ref_tokens = token_lists_by_engine.get(engine, [])
        for method, result in methods.items():
            topics = result.get("topics", [])
            logger.info(f"  Evaluating {engine} × {method} ({len(topics)} topics) …")

            npmi = compute_coherence(topics, ref_tokens, coherence="c_npmi", topn=topn)
            umass = compute_coherence(topics, ref_tokens, coherence="u_mass", topn=topn)
            diversity = compute_topic_diversity(topics, topn=topn)

            records.append({
                "engine": engine,
                "method": method,
                "n_topics": result.get("n_topics", 0),
                "npmi": npmi,
                "umass": umass,
                "diversity": diversity,
            })

            logger.info(f"    NPMI={npmi:.4f}  UMass={umass:.4f}  Diversity={diversity:.4f}")

    df = pd.DataFrame(records)
    return df


# ─────────────────────────────────────────────────────────────
# 4.  ERROR PROPAGATION ANALYSIS
# ─────────────────────────────────────────────────────────────

def compute_error_propagation(
    token_lists_raw: dict[str, list[list[str]]],
    token_lists_corrected: dict[str, list[list[str]]],
    topic_fn,
    n_topics: int = 7,
) -> pd.DataFrame:
    """
    Measure how dictionary-based correction affects NPMI.

    For each engine, fit the same topic model on:
      (a) raw preprocessed tokens (no correction)
      (b) corrected tokens

    Then compare NPMI scores to compute:
      Δ NPMI = NPMI_corrected − NPMI_raw
      Correction gain % = (Δ NPMI / |NPMI_raw|) × 100

    Parameters
    ----------
    token_lists_raw       : {engine: list of raw token lists}
    token_lists_corrected : {engine: list of corrected token lists}
    topic_fn              : callable – one of run_lda, run_nmf, run_bertopic
                            Must accept token_lists as first positional arg.
    n_topics              : fixed topic count for comparison

    Returns
    -------
    pd.DataFrame with columns:
      engine, npmi_raw, npmi_corrected, delta_npmi, gain_pct
    """
    records = []
    for engine in token_lists_raw:
        raw = token_lists_raw[engine]
        cor = token_lists_corrected[engine]

        logger.info(f"  Error propagation: {engine} …")

        # Raw
        result_raw = topic_fn(raw, n_topics=n_topics)
        npmi_raw = compute_coherence(result_raw["topics"], raw, "c_npmi")

        # Corrected
        result_cor = topic_fn(cor, n_topics=n_topics)
        npmi_cor = compute_coherence(result_cor["topics"], cor, "c_npmi")

        delta = npmi_cor - npmi_raw
        gain_pct = (delta / abs(npmi_raw) * 100) if npmi_raw != 0 else 0.0

        logger.info(f"    Raw NPMI: {npmi_raw:.4f}  |  Corrected NPMI: {npmi_cor:.4f}  "
                    f"|  Δ={delta:+.4f}  ({gain_pct:+.1f}%)")

        records.append({
            "engine": engine,
            "npmi_raw": npmi_raw,
            "npmi_corrected": npmi_cor,
            "delta_npmi": delta,
            "gain_pct": gain_pct,
        })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────
# 5.  PRETTY PRINT TOPICS
# ─────────────────────────────────────────────────────────────

def print_topics(result: dict, n_words: int = 10):
    """Print topic words in a human-readable table format."""
    engine = result.get("engine", "?")
    method = result.get("method", "?")
    topics = result.get("topics", [])

    print(f"\n{'='*60}")
    print(f"  Engine: {engine.upper()}  |  Method: {method.upper()}")
    print(f"  Topics: {len(topics)}")
    print(f"{'='*60}")

    for t in topics:
        words_str = ", ".join(t["words"][:n_words])
        print(f"  Topic {t['id']:02d}: {words_str}")
    print()


def save_evaluation_results(df: pd.DataFrame, out_path: str):
    """Save evaluation DataFrame to CSV."""
    df.to_csv(out_path, index=False)
    logger.info(f"  Evaluation results saved -> {out_path}")
    return out_path