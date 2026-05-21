"""
visualization.py
================
All plotting and visualisation for the project.

Charts produced:
  1. Comparative NPMI bar chart (3 ASR × 3 methods)
  2. Comparative UMass bar chart
  3. Topic diversity chart
  4. Error propagation Δ NPMI chart
  5. pyLDAvis interactive HTML (for LDA)
  6. BERTopic visualisations (topic map, bar chart)
  7. Word clouds per topic
  8. Document-topic heatmaps
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

logger = logging.getLogger(__name__)

# ── Colour scheme ────────────────────────────────────────────
ENGINE_COLORS = {
    "whisper": "#2196F3",    # blue
    "lelapa":  "#4CAF50",    # green
    "wav2vec": "#FF9800",    # orange
}
METHOD_PATTERNS = {
    "lda":      "",          # solid
    "bertopic": "///",
    "nmf":      "...",
}
PALETTE = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336", "#00BCD4",
           "#FF5722", "#8BC34A", "#3F51B5"]

plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ─────────────────────────────────────────────────────────────
# 1.  COMPARATIVE BAR CHARTS  (NPMI / UMass)
# ─────────────────────────────────────────────────────────────

def plot_coherence_comparison(df: pd.DataFrame, metric: str = "npmi",
                              out_path: str | None = None) -> plt.Figure:
    """
    Grouped bar chart: ASR engine × topic method coherence scores.

    Parameters
    ----------
    df      : evaluation DataFrame (from evaluate_all_conditions)
    metric  : "npmi" or "umass"
    out_path: if provided, save figure here

    Returns
    -------
    matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    engines = df["engine"].unique().tolist()
    methods = df["method"].unique().tolist()
    n_engines = len(engines)
    n_methods = len(methods)

    x = np.arange(n_engines)
    bar_width = 0.25
    offsets = np.linspace(-(n_methods - 1) / 2, (n_methods - 1) / 2, n_methods) * bar_width

    for i, method in enumerate(methods):
        vals = []
        for engine in engines:
            row = df[(df["engine"] == engine) & (df["method"] == method)]
            vals.append(row[metric].values[0] if len(row) else 0.0)

        bars = ax.bar(x + offsets[i], vals, bar_width * 0.9,
                      label=method.upper(),
                      hatch=METHOD_PATTERNS.get(method, ""),
                      color=[ENGINE_COLORS.get(e, "#999") for e in engines],
                      alpha=0.85, edgecolor="white")

        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([e.upper() for e in engines], fontsize=11)
    ax.set_ylabel(f"{metric.upper()} Score", fontsize=11)
    ax.set_title(f"Topic Coherence ({metric.upper()}) by ASR Engine × Topic Method\n"
                 f"Setswana COVID-19 Podcast Corpus", fontsize=13, pad=15)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.legend(title="Topic Method", loc="upper right", framealpha=0.9)
    ax.set_ylim(min(df[metric].min() - 0.05, -0.1), df[metric].max() + 0.08)

    # Engine colour legend
    engine_patches = [mpatches.Patch(color=ENGINE_COLORS.get(e, "#999"), label=e.upper())
                      for e in engines]
    ax2 = ax.twinx()
    ax2.set_yticks([])
    ax2.legend(handles=engine_patches, title="ASR Engine", loc="upper left", framealpha=0.9)

    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, bbox_inches="tight")
        logger.info(f"  Saved coherence chart → {out_path}")
    return fig


def plot_diversity(df: pd.DataFrame, out_path: str | None = None) -> plt.Figure:
    """Bar chart of topic diversity scores."""
    fig, ax = plt.subplots(figsize=(8, 5))
    df_pivot = df.pivot(index="engine", columns="method", values="diversity")

    df_pivot.plot(kind="bar", ax=ax, colormap="Set2", edgecolor="white", rot=0)
    ax.set_title("Topic Diversity by ASR Engine × Method", fontsize=13)
    ax.set_ylabel("Diversity Score (0–1)", fontsize=11)
    ax.set_xlabel("ASR Engine", fontsize=11)
    ax.legend(title="Method", loc="upper right")
    ax.set_ylim(0, 1.1)
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, bbox_inches="tight")
        logger.info(f"  Saved diversity chart → {out_path}")
    return fig


# ─────────────────────────────────────────────────────────────
# 2.  ERROR PROPAGATION CHART
# ─────────────────────────────────────────────────────────────

def plot_error_propagation(err_df: pd.DataFrame,
                           out_path: str | None = None) -> plt.Figure:
    """
    Side-by-side bar chart showing NPMI before/after correction,
    with Δ NPMI annotated.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    x = np.arange(len(err_df))
    engines = err_df["engine"].tolist()
    colors = [ENGINE_COLORS.get(e, "#999") for e in engines]

    # Left: raw vs corrected NPMI
    w = 0.35
    b1 = ax1.bar(x - w/2, err_df["npmi_raw"], w, label="Raw (no correction)",
                 color=colors, alpha=0.5, edgecolor="black")
    b2 = ax1.bar(x + w/2, err_df["npmi_corrected"], w, label="After correction",
                 color=colors, alpha=0.95, edgecolor="black")
    ax1.set_xticks(x)
    ax1.set_xticklabels([e.upper() for e in engines])
    ax1.set_title("NPMI: Before vs After Dictionary Correction", fontsize=12)
    ax1.set_ylabel("NPMI Score", fontsize=11)
    ax1.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax1.legend()

    # Right: Δ NPMI per engine
    delta_colors = ["#4CAF50" if d >= 0 else "#F44336" for d in err_df["delta_npmi"]]
    bars = ax2.bar(x, err_df["delta_npmi"], color=delta_colors, edgecolor="white")
    ax2.set_xticks(x)
    ax2.set_xticklabels([e.upper() for e in engines])
    ax2.set_title("Δ NPMI (Correction Gain)", fontsize=12)
    ax2.set_ylabel("Δ NPMI", fontsize=11)
    ax2.axhline(0, color="black", linewidth=1.0)

    for bar, (_, row) in zip(bars, err_df.iterrows()):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + (0.002 if row["delta_npmi"] >= 0 else -0.008),
                 f"{row['gain_pct']:+.1f}%", ha="center", fontsize=9)

    fig.suptitle("Error Propagation Analysis: Dictionary-Based ASR Correction",
                 fontsize=14, y=1.01)
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, bbox_inches="tight")
        logger.info(f"  Saved error propagation chart → {out_path}")
    return fig


# ─────────────────────────────────────────────────────────────
# 3.  pyLDAvis (interactive)
# ─────────────────────────────────────────────────────────────

def save_pyldavis(lda_result: dict, out_path: str) -> str:
    """
    Export an interactive pyLDAvis HTML for a LDA result.

    Parameters
    ----------
    lda_result : result dict from run_lda
    out_path   : path for the .html output file

    Returns
    -------
    out_path
    """
    import pyLDAvis
    import pyLDAvis.gensim_models as gensimvis

    lda_model  = lda_result["model"]
    dictionary = lda_result["extra"]["dictionary"]
    corpus     = lda_result["extra"]["corpus"]

    if lda_model is None:
        logger.warning("  LDA model is None – skipping pyLDAvis")
        return out_path

    logger.info("  Preparing pyLDAvis …")
    vis_data = gensimvis.prepare(lda_model, corpus, dictionary, mds="mmds")
    pyLDAvis.save_html(vis_data, out_path)
    logger.info(f"  pyLDAvis saved → {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────
# 4.  BERTopic visualisations
# ─────────────────────────────────────────────────────────────

def save_bertopic_visuals(bertopic_result: dict, out_dir: str):
    """
    Save BERTopic inter-topic distance map and topic bar charts as HTML files.

    Parameters
    ----------
    bertopic_result : result dict from run_bertopic
    out_dir         : directory for output HTML files
    """
    model = bertopic_result.get("model")
    if model is None:
        logger.warning("  BERTopic model is None – skipping viz")
        return

    os.makedirs(out_dir, exist_ok=True)

    try:
        # Inter-topic distance map
        fig_topics = model.visualize_topics()
        fig_topics.write_html(os.path.join(out_dir, "bertopic_topic_map.html"))
        logger.info("  Saved BERTopic topic map")

        # Topic word bar charts
        fig_bars = model.visualize_barchart(top_n_topics=min(12, bertopic_result["n_topics"]),
                                             n_words=10)
        fig_bars.write_html(os.path.join(out_dir, "bertopic_barchart.html"))
        logger.info("  Saved BERTopic bar chart")

        # Hierarchy
        if bertopic_result["n_topics"] > 2:
            fig_hier = model.visualize_hierarchy()
            fig_hier.write_html(os.path.join(out_dir, "bertopic_hierarchy.html"))
            logger.info("  Saved BERTopic hierarchy")

    except Exception as e:
        logger.warning(f"  BERTopic viz failed: {e}")


# ─────────────────────────────────────────────────────────────
# 5.  WORD CLOUDS
# ─────────────────────────────────────────────────────────────

def plot_word_clouds(result: dict, max_topics: int = 6,
                     out_dir: str | None = None) -> plt.Figure:
    """
    Word cloud grid for top topics of any model.

    Parameters
    ----------
    result     : standardised result dict
    max_topics : maximum topics to display
    out_dir    : save individual PNGs here (optional)

    Returns
    -------
    matplotlib Figure
    """
    from wordcloud import WordCloud

    topics = result.get("topics", [])[:max_topics]
    if not topics:
        logger.warning("  No topics for word cloud")
        return plt.figure()

    n = len(topics)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    if n == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)

    for i, (topic, ax) in enumerate(zip(topics, axes.flat)):
        freq = {w: max(wt, 0.001) for w, wt in zip(topic["words"], topic["weights"])}
        wc = WordCloud(
            width=400, height=300,
            background_color="white",
            colormap=PALETTE[i % len(PALETTE)],
            prefer_horizontal=0.9,
            max_words=30,
        ).generate_from_frequencies(freq)
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(f"Topic {topic['id']}: {', '.join(topic['words'][:3])}…",
                     fontsize=10, pad=6)

    # Hide empty axes
    for j in range(i + 1, nrows * ncols):
        axes.flat[j].axis("off")

    engine = result.get("engine", "?")
    method = result["method"].upper()
    fig.suptitle(f"Word Clouds – {engine.upper()} × {method}", fontsize=13, y=1.01)
    fig.tight_layout()

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"wordclouds_{engine}_{result['method']}.png")
        fig.savefig(path, bbox_inches="tight")
        logger.info(f"  Saved word clouds → {path}")

    return fig


# ─────────────────────────────────────────────────────────────
# 6.  DOCUMENT-TOPIC HEATMAP
# ─────────────────────────────────────────────────────────────

def plot_doc_topic_heatmap(result: dict, doc_labels: list[str] | None = None,
                           out_path: str | None = None) -> plt.Figure:
    """
    Heatmap showing document-topic proportions.

    Parameters
    ----------
    result     : standardised result dict
    doc_labels : list of document names (e.g. episode file names)
    out_path   : save path

    Returns
    -------
    matplotlib Figure
    """
    matrix = result.get("doc_topic_matrix")
    if matrix is None or len(matrix) == 0:
        return plt.figure()

    n_docs, n_topics = matrix.shape
    doc_labels = doc_labels or [f"Ep{i+1}" for i in range(n_docs)]
    topic_labels = [f"T{t['id']}" for t in result.get("topics", range(n_topics))][:n_topics]

    fig, ax = plt.subplots(figsize=(max(6, n_topics), max(4, n_docs * 0.5 + 1)))
    sns.heatmap(matrix, ax=ax,
                xticklabels=topic_labels,
                yticklabels=doc_labels,
                cmap="YlOrRd", annot=n_docs <= 15, fmt=".2f",
                linewidths=0.5, cbar_kws={"label": "Topic Proportion"})
    ax.set_title(f"Document-Topic Matrix – {result.get('engine','').upper()} × "
                 f"{result['method'].upper()}", fontsize=12, pad=10)
    ax.set_xlabel("Topics")
    ax.set_ylabel("Documents (Episodes)")
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, bbox_inches="tight")
        logger.info(f"  Saved heatmap → {out_path}")
    return fig


# ─────────────────────────────────────────────────────────────
# 7.  SUMMARY TABLE
# ─────────────────────────────────────────────────────────────

def print_summary_table(df: pd.DataFrame):
    """Pretty-print the evaluation DataFrame as a formatted table."""
    print("\n" + "="*70)
    print("  EVALUATION SUMMARY")
    print("="*70)
    best_npmi_idx = df["npmi"].idxmax()
    print(df.to_string(index=False, float_format="%.4f"))
    print("="*70)
    best = df.loc[best_npmi_idx]
    print(f"\n  ★ Best NPMI: {best['engine'].upper()} × {best['method'].upper()} "
          f"({best['npmi']:.4f})")
    print()