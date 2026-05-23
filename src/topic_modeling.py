"""
topic_modeling.py
=================
Three topic modelling approaches:
  1. LDA  (Gensim)       - probabilistic bag-of-words model
  2. BERTopic             - embedding-based neural topic model
  3. NMF  (sklearn)       - matrix factorisation on TF-IDF

Each function accepts preprocessed token lists / joined strings
and returns a standardised result dict for coherence evaluation.

Returned dict shape (all three models):
  {
    "method"  : str,
    "n_topics": int,
    "topics"  : [{"id": int, "words": [str, ...], "weights": [float, ...]}, ...],
    "doc_topic_matrix": np.ndarray (n_docs × n_topics),
    "model"   : <the fitted model object>,   # for pyLDAvis / BERTopic viz
    "extra"   : {...}                         # model-specific extras
  }
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# 1.  LDA  (Gensim)
# ─────────────────────────────────────────────────────────────

def run_lda(token_lists: list[list[str]],
            n_topics: int | None = None,
            topic_range: tuple[int, int] = (3, 15),
            n_top_words: int = 10,
            passes: int = 20,
            random_state: int = 42) -> dict:
    """
    Fit LDA with Gensim.  If n_topics is None, search topic_range and
    choose the number that maximises NPMI coherence (c_v proxy).

    Parameters
    ----------
    token_lists  : list of token lists (corrected, post-stopword removal)
    n_topics     : fixed number of topics, or None for automatic search
    topic_range  : (min, max) topics to search if n_topics is None
    n_top_words  : number of top words per topic to extract
    passes       : LDA training passes
    random_state : reproducibility seed

    Returns
    -------
    Standardised result dict (see module docstring)
    """
    from gensim import corpora, models
    from gensim.models.coherencemodel import CoherenceModel
    import sys
    sys.path.insert(0, "src")
    from preprocessing import build_gensim_corpus

    logger.info("=== LDA ===")
    dictionary, corpus = build_gensim_corpus(token_lists)

    if len(dictionary) < 3:
        logger.warning("  Dictionary too small for LDA - returning empty result.")
        return _empty_result("lda")

    def _fit(k: int):
        return models.LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=k,
            passes=passes,
            random_state=random_state,
            alpha="auto",
            eta="auto",
            minimum_probability=0.0,
        )

    if n_topics is None:
        logger.info(f"  Searching topic range {topic_range} …")
        best_k, best_score, best_model = topic_range[0], -np.inf, None
        for k in range(topic_range[0], topic_range[1] + 1):
            m = _fit(k)
            cm = CoherenceModel(model=m, texts=token_lists,
                                dictionary=dictionary, coherence="c_v")
            score = cm.get_coherence()
            logger.info(f"    k={k}  c_v={score:.4f}")
            if score > best_score:
                best_k, best_score, best_model = k, score, m
        n_topics = best_k
        lda_model = best_model
        logger.info(f"  Best k={n_topics} (c_v={best_score:.4f})")
    else:
        lda_model = _fit(n_topics)

    # Extract topics
    topics = []
    for tid in range(n_topics):
        pairs = lda_model.show_topic(tid, topn=n_top_words)
        topics.append({
            "id": tid,
            "words": [w for w, _ in pairs],
            "weights": [float(p) for _, p in pairs],
        })

    # Document-topic matrix
    doc_topic = np.zeros((len(corpus), n_topics))
    for i, bow in enumerate(corpus):
        for tid, prob in lda_model.get_document_topics(bow, minimum_probability=0.0):
            doc_topic[i, tid] = prob

    logger.info(f"  LDA fitted: {n_topics} topics")

    return {
        "method": "lda",
        "n_topics": n_topics,
        "topics": topics,
        "doc_topic_matrix": doc_topic,
        "model": lda_model,
        "extra": {
            "dictionary": dictionary,
            "corpus": corpus,
            "token_lists": token_lists,
        },
    }


# ─────────────────────────────────────────────────────────────
# 2.  BERTopic
# ─────────────────────────────────────────────────────────────

BERTOPIC_EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
# This model supports 50+ languages including Sotho / Tswana language family.

def run_bertopic(joined_strings: list[str],
                 n_topics: int | str = "auto",
                 n_top_words: int = 10,
                 min_topic_size: int = 2,
                 random_state: int = 42) -> dict:
    """
    Fit BERTopic with multilingual sentence embeddings.

    FIX: We pre-compute embeddings with SentenceTransformer directly,
    then pass them to BERTopic as a numpy array. This bypasses BERTopic's
    internal backend selection which tries to import StaticEmbedding
    (only in sentence-transformers v3+) and crashes on v2.x.

    Parameters
    ----------
    joined_strings : list of space-joined preprocessed strings (one per doc)
    n_topics       : int for fixed topics, "auto" for automatic detection
    n_top_words    : words per topic
    min_topic_size : minimum documents per cluster (lower = more topics)
    random_state   : reproducibility seed

    Returns
    -------
    Standardised result dict
    """
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP
    from hdbscan import HDBSCAN

    logger.info("=== BERTopic ===")
    logger.info(f"  Embedding model: {BERTOPIC_EMBEDDING_MODEL}")

    if len(joined_strings) < 4:
        logger.warning("  Too few documents for BERTopic - returning empty result.")
        return _empty_result("bertopic")

    # Step 1: compute embeddings ourselves using sentence-transformers directly.
    # We do NOT pass the SentenceTransformer object to BERTopic - that triggers
    # the broken backend import. Instead we pass the raw numpy embedding array.
    logger.info("  Computing embeddings ...")
    st_model = SentenceTransformer(BERTOPIC_EMBEDDING_MODEL)
    embeddings = st_model.encode(joined_strings, show_progress_bar=False,
                                  convert_to_numpy=True)
    logger.info(f"  Embeddings shape: {embeddings.shape}")

    # Step 2: UMAP + HDBSCAN + Vectorizer (unchanged)
    umap_model = UMAP(
        n_neighbors=min(5, len(joined_strings) - 1),
        n_components=min(5, len(joined_strings) - 2),
        metric="cosine",
        random_state=random_state,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_topic_size,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    vectorizer = CountVectorizer(min_df=1, ngram_range=(1, 2))
    nr_topics = n_topics if isinstance(n_topics, int) else "auto"

    # Step 3: Build BERTopic WITHOUT an embedding_model argument.
    # Passing embedding_model=None tells BERTopic we will supply embeddings
    # ourselves, so it never tries to load a SentenceTransformer backend.
    topic_model = BERTopic(
        embedding_model=None,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        nr_topics=nr_topics,
        top_n_words=n_top_words,
        verbose=True,
    )

    # Step 4: fit_transform with pre-computed embeddings
    logger.info("  Fitting BERTopic ...")
    topics_assigned, probs = topic_model.fit_transform(
        joined_strings, embeddings=embeddings
    )

    # Extract topic info
    topic_info = topic_model.get_topic_info()
    # Remove outlier topic (-1)
    topic_info = topic_info[topic_info["Topic"] != -1]
    n_found = len(topic_info)
    logger.info(f"  BERTopic found {n_found} topics")

    topics = []
    for _, row in topic_info.iterrows():
        tid = row["Topic"]
        words_weights = topic_model.get_topic(tid)
        if words_weights:
            topics.append({
                "id": int(tid),
                "words": [w for w, _ in words_weights[:n_top_words]],
                "weights": [float(p) for _, p in words_weights[:n_top_words]],
            })

    # Document-topic probability matrix
    n_docs = len(joined_strings)
    n_actual_topics = max(n_found, 1)
    doc_topic = np.zeros((n_docs, n_actual_topics))
    valid_topics = [t["id"] for t in topics]

    if probs is not None:
        # probs may be 1D (single topic) or 2D
        if hasattr(probs, "shape") and len(probs.shape) == 2:
            doc_topic = probs[:, :n_actual_topics]
        else:
            # Use topic assignments for one-hot approximation
            for i, t in enumerate(topics_assigned):
                if t in valid_topics:
                    col = valid_topics.index(t)
                    doc_topic[i, col] = 1.0

    return {
        "method": "bertopic",
        "n_topics": n_found,
        "topics": topics,
        "doc_topic_matrix": doc_topic,
        "model": topic_model,
        "extra": {
            "topic_assignments": topics_assigned,
            "probabilities": probs,
        },
    }


# ─────────────────────────────────────────────────────────────
# 3.  NMF  (sklearn)
# ─────────────────────────────────────────────────────────────

def run_nmf(joined_strings: list[str],
            n_topics: int = 7,
            n_top_words: int = 10,
            random_state: int = 42,
            max_df: float = 0.85,
            min_df: int = 2) -> dict:
    """
    Non-negative Matrix Factorisation on TF-IDF features.

    NMF is deterministic (with fixed random state), fast, and often
    produces more interpretable topics than LDA on small corpora.

    Parameters
    ----------
    joined_strings : preprocessed text strings (one per document)
    n_topics       : number of topics (try 5-10 for this corpus)
    n_top_words    : words per topic to extract
    random_state   : reproducibility seed
    max_df         : ignore tokens in > max_df fraction of docs
    min_df         : ignore tokens in < min_df documents

    Returns
    -------
    Standardised result dict
    """
    from sklearn.decomposition import NMF
    from sklearn.feature_extraction.text import TfidfVectorizer

    logger.info("=== NMF ===")

    if len(joined_strings) < 3:
        logger.warning("  Too few documents for NMF - returning empty result.")
        return _empty_result("nmf")

    vectorizer = TfidfVectorizer(
        max_df=max_df,
        min_df=min_df,
        max_features=5000,
        ngram_range=(1, 2),        # unigrams + bigrams for richer topics
    )
    tfidf_matrix = vectorizer.fit_transform(joined_strings)
    feature_names = np.array(vectorizer.get_feature_names_out())

    logger.info(f"  TF-IDF matrix: {tfidf_matrix.shape}")

    nmf_model = NMF(
        n_components=n_topics,
        random_state=random_state,
        init="nndsvda",            # better initialisation for sparse matrices
        max_iter=500,
    )
    doc_topic = nmf_model.fit_transform(tfidf_matrix)   # (n_docs × n_topics)

    # Normalise rows so they sum to 1 (topic proportions)
    row_sums = doc_topic.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    doc_topic_norm = doc_topic / row_sums

    # Extract top words per topic from H matrix (topics × vocab)
    H = nmf_model.components_
    topics = []
    for tid in range(n_topics):
        top_idx = H[tid].argsort()[::-1][:n_top_words]
        topics.append({
            "id": tid,
            "words": feature_names[top_idx].tolist(),
            "weights": H[tid][top_idx].tolist(),
        })

    logger.info(f"  NMF fitted: {n_topics} topics")

    return {
        "method": "nmf",
        "n_topics": n_topics,
        "topics": topics,
        "doc_topic_matrix": doc_topic_norm,
        "model": nmf_model,
        "extra": {
            "tfidf_matrix": tfidf_matrix,
            "vectorizer": vectorizer,
            "H": H,
        },
    }


# ─────────────────────────────────────────────────────────────
# 4.  BATCH RUNNER
# ─────────────────────────────────────────────────────────────

def run_all_topic_models(token_lists: list[list[str]],
                         joined_strings: list[str],
                         n_topics: int = 7) -> dict:
    """
    Run all three topic models and return results keyed by method name.

    Parameters
    ----------
    token_lists    : corrected token lists (for LDA + coherence)
    joined_strings : joined preprocessed strings (for BERTopic + NMF)
    n_topics       : fixed topic count for NMF; used as starting point for LDA search

    Returns
    -------
    {"lda": {...}, "bertopic": {...}, "nmf": {...}}
    """
    return {
        "lda":      run_lda(token_lists, n_topics=n_topics),
        "bertopic": run_bertopic(joined_strings, n_topics=n_topics),
        "nmf":      run_nmf(joined_strings, n_topics=n_topics),
    }


# ─────────────────────────────────────────────────────────────
# 5.  HELPER
# ─────────────────────────────────────────────────────────────

def _empty_result(method: str) -> dict:
    return {
        "method": method,
        "n_topics": 0,
        "topics": [],
        "doc_topic_matrix": np.array([]),
        "model": None,
        "extra": {},
    }