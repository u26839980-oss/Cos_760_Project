"""
preprocessing.py
================
Text preprocessing for ASR transcripts.

Pipeline:
  1. Lowercase
  2. Remove punctuation & non-alphabetic tokens
  3. Tokenise
  4. Remove stopwords (English + custom Setswana list)
  5. Dictionary-based correction (Levenshtein) to handle ASR noise
  6. Return clean token lists per document (episode)

Why these steps?
----------------
ASR output is noisy: repeated tokens, hesitation markers, misrecognised
words.  For LDA / NMF the input is a bag-of-words, so token quality
directly drives coherence.  BERTopic uses sentence embeddings, so it is
more robust to individual token errors – but still benefits from stop-
word removal.
"""

import re
import logging
import string
from collections import Counter
from Levenshtein import distance as levenshtein_distance
import nltk

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# 0.  SETSWANA STOPWORDS
# ─────────────────────────────────────────────────────────────
# These are the most frequent Setswana function words that carry
# no thematic information.  Extend this list as you find more
# high-frequency non-content words in your transcripts.

SETSWANA_STOPWORDS = {
    # Copulatives / auxiliaries
    "ke", "ga", "go", "le", "la", "lo", "ba", "bo", "di",
    "se", "mo", "re", "o", "a", "e", "wa", "ya", "ra",
    # Common conjunctions / discourse markers
    "le", "mme", "fela", "jaanong", "gape", "kwa", "jalo",
    "gore", "fa", "ka", "kwa", "go", "ne", "ene", "ruri",
    "thata", "gone", "bjang", "jaaka", "ntlo", "yo", "yona",
    # Pronouns
    "ene", "bone", "rona", "lona", "bona",
    # Numbers as words that recur without content value
    "one", "two", "three",
    # English filler words that leak into Setswana speech
    "the", "and", "of", "to", "is", "that", "it", "in",
    "we", "you", "i", "this", "for", "on", "are", "with",
    "said", "so", "at", "be", "but", "by", "from", "or",
    "an", "they", "all", "our", "not", "as", "have", "also",
    "its", "was", "he", "she", "his", "her", "their", "been",
    "do", "did", "has", "had", "will", "would", "can", "could",
    "may", "might", "must", "shall", "should", "about", "which",
    "there", "when", "up", "out", "more", "who", "than",
}

# Also download NLTK English stopwords as a fallback
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

from nltk.corpus import stopwords as nltk_sw
ENGLISH_STOPWORDS = set(nltk_sw.words("english"))
ALL_STOPWORDS = SETSWANA_STOPWORDS | ENGLISH_STOPWORDS


# ─────────────────────────────────────────────────────────────
# 1.  BASIC CLEANING
# ─────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Lowercase, remove punctuation, collapse whitespace.

    >>> clean_text("O a bua, COVID-19?")
    'o a bua covid19'

    Note: We keep digits attached to letters (covid19) because
    they may be semantically meaningful in health discourse.
    Standalone digits are removed in tokenisation.
    """
    text = text.lower()
    # Remove everything that isn't a letter, digit, or space
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenise(text: str, min_len: int = 2) -> list[str]:
    """
    Split on whitespace, remove pure-digit tokens and short tokens.

    Parameters
    ----------
    text    : already cleaned (lowercase, no punct) string
    min_len : discard tokens shorter than this
    """
    tokens = text.split()
    tokens = [t for t in tokens if not t.isdigit() and len(t) >= min_len]
    return tokens


def remove_stopwords(tokens: list[str]) -> list[str]:
    """Remove tokens that appear in the combined stopword set."""
    return [t for t in tokens if t not in ALL_STOPWORDS]


# ─────────────────────────────────────────────────────────────
# 2.  DICTIONARY-BASED ASR CORRECTION
# ─────────────────────────────────────────────────────────────

def build_vocab(all_token_lists: list[list[str]],
                min_freq: int = 3) -> set[str]:
    """
    Build a vocabulary of 'trusted' words: those appearing at least
    min_freq times across the entire corpus.

    Words that appear only once or twice are likely ASR mis-recognitions.
    We use the frequent words as the correction dictionary.

    Parameters
    ----------
    all_token_lists : list of token lists (one per document)
    min_freq        : minimum corpus frequency to be trusted

    Returns
    -------
    set of trusted word-forms
    """
    counter = Counter(tok for doc in all_token_lists for tok in doc)
    vocab = {word for word, freq in counter.items() if freq >= min_freq}
    logger.info(f"  Vocab size (freq ≥ {min_freq}): {len(vocab):,}")
    return vocab


def correct_token(token: str, vocab: set[str],
                  max_edit_dist: int = 2) -> str:
    """
    If `token` is not in vocab, find the closest vocab word by
    Levenshtein distance.  If the closest match is within max_edit_dist,
    replace the token; otherwise keep it.

    This approximates ASR post-processing without any language model.

    Parameters
    ----------
    token        : single word to check
    vocab        : set of trusted words
    max_edit_dist: maximum allowed edit distance for a replacement

    Returns
    -------
    corrected token (or original if no close match found)
    """
    if token in vocab:
        return token                  # already trusted
    if len(token) <= 2:
        return token                  # too short to correct meaningfully

    best_word, best_dist = token, max_edit_dist + 1
    for candidate in vocab:
        # Skip candidates with very different lengths (speed optimisation)
        if abs(len(candidate) - len(token)) > max_edit_dist:
            continue
        d = levenshtein_distance(token, candidate)
        if d < best_dist:
            best_dist = d
            best_word = candidate

    if best_dist <= max_edit_dist:
        return best_word
    return token


def apply_dictionary_correction(token_lists: list[list[str]],
                                 vocab: set[str],
                                 max_edit_dist: int = 2) -> list[list[str]]:
    """
    Apply correct_token to every token in every document.

    Note: This step is intentionally lightweight.  The proposal specifies
    'no ASR fine-tuning' and this is purely post-hoc string correction.

    Parameters
    ----------
    token_lists  : list of token lists (one per document)
    vocab        : trusted vocabulary (from build_vocab)
    max_edit_dist: passed to correct_token

    Returns
    -------
    corrected list of token lists
    """
    corrected = []
    total_corrections = 0
    for doc in token_lists:
        new_doc = []
        for tok in doc:
            corrected_tok = correct_token(tok, vocab, max_edit_dist)
            if corrected_tok != tok:
                total_corrections += 1
            new_doc.append(corrected_tok)
        corrected.append(new_doc)

    n_total = sum(len(d) for d in token_lists)
    logger.info(f"  Dictionary corrections: {total_corrections:,} / {n_total:,} tokens "
                f"({100*total_corrections/max(n_total,1):.1f}%)")
    return corrected


# ─────────────────────────────────────────────────────────────
# 3.  FULL PREPROCESSING PIPELINE
# ─────────────────────────────────────────────────────────────

def preprocess_transcripts(transcripts: list[str],
                            apply_correction: bool = True,
                            min_freq: int = 3,
                            min_token_len: int = 2) -> dict:
    """
    Full preprocessing pipeline.

    Parameters
    ----------
    transcripts      : list of raw transcript strings (one per episode)
    apply_correction : whether to run dictionary-based ASR correction
    min_freq         : minimum corpus frequency for trusted vocab
    min_token_len    : minimum token length to keep

    Returns
    -------
    dict with keys:
      "raw_tokens"       : token lists before correction
      "corrected_tokens" : token lists after correction (same as raw if skipped)
      "joined_strings"   : space-joined strings (for BERTopic / NMF)
      "vocab"            : trusted vocabulary set
      "correction_rate"  : fraction of tokens changed
    """
    logger.info("=== Preprocessing pipeline ===")
    logger.info(f"  Documents: {len(transcripts)}")

    # ── Step 1-3: clean → tokenise → remove stopwords ────────
    raw_token_lists = []
    for i, text in enumerate(transcripts):
        cleaned = clean_text(text)
        tokens = tokenise(cleaned, min_len=min_token_len)
        tokens = remove_stopwords(tokens)
        raw_token_lists.append(tokens)
        logger.debug(f"  Doc {i}: {len(tokens)} tokens after cleaning")

    total_raw = sum(len(d) for d in raw_token_lists)
    logger.info(f"  Total tokens after cleaning: {total_raw:,}")

    # ── Step 4: Build vocab ───────────────────────────────────
    vocab = build_vocab(raw_token_lists, min_freq=min_freq)

    # ── Step 5: Dictionary correction (optional) ─────────────
    if apply_correction:
        logger.info("  Applying dictionary-based correction …")
        corrected_lists = apply_dictionary_correction(raw_token_lists, vocab)
    else:
        corrected_lists = raw_token_lists
        logger.info("  Dictionary correction: SKIPPED")

    # ── Step 6: Joined strings ────────────────────────────────
    joined = [" ".join(doc) for doc in corrected_lists]

    # Correction rate
    diffs = sum(
        1 for raw, cor in zip(
            (t for doc in raw_token_lists for t in doc),
            (t for doc in corrected_lists for t in doc)
        ) if raw != cor
    )
    rate = diffs / max(total_raw, 1)

    logger.info(f"  Correction rate: {100*rate:.2f}%")
    logger.info("=== Preprocessing complete ===")

    return {
        "raw_tokens": raw_token_lists,
        "corrected_tokens": corrected_lists,
        "joined_strings": joined,
        "vocab": vocab,
        "correction_rate": rate,
    }


# ─────────────────────────────────────────────────────────────
# 4.  HELPER: PREPARE GENSIM CORPUS
# ─────────────────────────────────────────────────────────────

def build_gensim_corpus(token_lists: list[list[str]]):
    """
    Build Gensim dictionary and corpus (bag-of-words) from token lists.

    Returns
    -------
    dictionary : gensim.corpora.Dictionary
    corpus     : list of (id, count) tuples per document
    """
    from gensim import corpora

    dictionary = corpora.Dictionary(token_lists)
    # Filter extremes: remove tokens in <2 docs or >50% of docs
    dictionary.filter_extremes(no_below=2, no_above=0.5)
    corpus = [dictionary.doc2bow(doc) for doc in token_lists]

    logger.info(f"  Gensim dictionary: {len(dictionary)} unique tokens")
    logger.info(f"  Corpus: {len(corpus)} documents")
    return dictionary, corpus