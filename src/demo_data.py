"""
demo_data.py
============
Generates realistic synthetic Setswana COVID-19 transcript data for
pipeline testing when real podcast audio is not yet available.

The synthetic transcripts simulate the kinds of vocabulary and topics
that would appear in Setswana COVID-19 health podcasts, including
intentional 'ASR noise' variants to test the correction pipeline.

Usage
-----
    from demo_data import generate_demo_transcripts
    transcripts_by_engine = generate_demo_transcripts(n_episodes=10, language="tsn")
"""

import random
import numpy as np

random.seed(42)
np.random.seed(42)

# ─────────────────────────────────────────────────────────────
# VOCABULARY POOLS
# ─────────────────────────────────────────────────────────────

# Health / COVID vocabulary in Setswana with English code-switching
# (common in South African media)
HEALTH_VOCAB = [
    "bolwetsi", "pholo", "kalafi", "ngaka", "sepetlele", "tlhokego",
    "ditlhare", "tlhokafalo", "bogole", "bophelo", "phekolo",
    "mente", "setlhopha", "maemo", "botsalano", "dikotla",
    "dikoloto", "mokgwa", "tshwanelo", "lefapha", "tirisano",
]

COVID_VOCAB = [
    "covid", "coronavirus", "pandemic", "vaccine", "vaccination",
    "lockdown", "quarantine", "isolation", "testing", "positive",
    "negative", "symptoms", "fever", "cough", "oxygen", "hospital",
    "icu", "ventilator", "antibody", "immunity", "booster",
    "omicron", "delta", "variant", "mask", "sanitiser",
    "social distancing", "transmission", "infection", "mortality",
]

GOVERNMENT_VOCAB = [
    "mmuso", "puso", "molao", "tsamaiso", "pholisi", "kgotla",
    "kgosi", "lekgotla", "tshwanelo", "ditshwanelo", "dikgang",
    "government", "minister", "department", "health", "protocol",
    "regulation", "restriction", "announcement", "statement",
    "regulation", "national", "provincial", "district",
]

COMMUNITY_VOCAB = [
    "setshaba", "motse", "morafe", "batho", "lelapa", "losika",
    "dikgosi", "masepala", "selegae", "tikologo", "tshepego",
    "community", "people", "families", "support", "awareness",
    "education", "information", "resources", "access",
]

ECONOMIC_VOCAB = [
    "ikonomi", "tshebetso", "madi", "lefapha", "ntshetsapele",
    "tlamorago", "kgwebo", "sekoloto", "economy", "jobs",
    "unemployment", "business", "relief", "grant", "social",
    "sassa", "payment", "income", "poverty", "inequality",
]

ALL_VOCAB = HEALTH_VOCAB + COVID_VOCAB + GOVERNMENT_VOCAB + COMMUNITY_VOCAB + ECONOMIC_VOCAB

# Realistic Setswana filler words / discourse markers that ASR keeps
FILLERS = ["ke", "le", "go", "mo", "re", "o", "ga", "a", "e", "ba"]

# Simulated ASR noise patterns (common misrecognitions)
NOISE_VARIANTS = {
    "bolwetsi": ["bolwesti", "bolwetse", "blwetsi"],
    "vaccine": ["vasine", "vaccin", "vaksin"],
    "lockdown": ["lokdown", "lockdawn", "lochdown"],
    "hospital": ["haspital", "hospitel", "ospital"],
    "community": ["commuity", "comunity", "commnity"],
    "government": ["goverment", "govenment", "govermnent"],
    "testing": ["tesing", "tsting", "testng"],
    "symptoms": ["symptons", "symtoms", "symptms"],
    "pandemic": ["pandemc", "pandamic", "pandmic"],
    "quarantine": ["quaratine", "quarentine", "quaranteen"],
}


def _add_noise(word: str, noise_level: float = 0.05) -> str:
    """Randomly corrupt a word to simulate ASR noise."""
    if word in NOISE_VARIANTS and random.random() < noise_level:
        return random.choice(NOISE_VARIANTS[word])
    return word


def _generate_episode(topic_weights: dict, n_words: int = 300,
                       noise_level: float = 0.05) -> str:
    """
    Generate a synthetic episode transcript.

    Parameters
    ----------
    topic_weights : dict mapping vocab list name to sampling probability
    n_words       : approximate word count
    noise_level   : fraction of words that get corrupted

    Returns
    -------
    transcript string
    """
    vocab_pools = {
        "health": HEALTH_VOCAB,
        "covid": COVID_VOCAB,
        "government": GOVERNMENT_VOCAB,
        "community": COMMUNITY_VOCAB,
        "economic": ECONOMIC_VOCAB,
    }

    # Sample words based on topic weights
    words = []
    for _ in range(n_words):
        # Occasionally add filler words (realistic)
        if random.random() < 0.15:
            words.append(random.choice(FILLERS))
            continue

        # Choose a vocab pool based on weights
        pool_name = random.choices(
            list(topic_weights.keys()),
            weights=list(topic_weights.values()),
        )[0]
        pool = vocab_pools[pool_name]
        word = random.choice(pool)
        word = _add_noise(word, noise_level)
        words.append(word)

    return " ".join(words)


def generate_demo_transcripts(n_episodes: int = 12,
                              language: str = "tsn") -> dict[str, list[str]]:
    """
    Generate synthetic transcripts for three ASR engine slots.

    Simulates different ASR quality levels. Engine labels depend on the
    selected language so demo mode reflects the actual experiment setup
    instead of legacy placeholder names.

    Each engine gets the same underlying episode content but with
    different noise levels.

    Parameters
    ----------
    n_episodes : number of podcast episodes to simulate

    Returns
    -------
    dict { engine_name: [transcript_str, ...] }
    """
    # Define 4 distinct episode topic archetypes
    EPISODE_ARCHETYPES = [
        # Medical / clinical focus
        {"health": 0.45, "covid": 0.35, "government": 0.10, "community": 0.05, "economic": 0.05},
        # Government policy focus
        {"government": 0.45, "covid": 0.25, "community": 0.15, "health": 0.10, "economic": 0.05},
        # Community awareness
        {"community": 0.40, "health": 0.25, "covid": 0.20, "government": 0.10, "economic": 0.05},
        # Economic impact
        {"economic": 0.40, "community": 0.25, "government": 0.20, "covid": 0.10, "health": 0.05},
        # Vaccination drive
        {"covid": 0.50, "health": 0.30, "government": 0.10, "community": 0.05, "economic": 0.05},
    ]

    # Base transcripts (clean, no noise)
    base_transcripts = []
    for i in range(n_episodes):
        archetype = EPISODE_ARCHETYPES[i % len(EPISODE_ARCHETYPES)]
        # Episodes vary in length (short podcasts ~200–400 words)
        n_words = random.randint(200, 400)
        transcript = _generate_episode(archetype, n_words=n_words, noise_level=0.0)
        base_transcripts.append(transcript)

    if language == "tsn":
        noise_levels = {
            "afrispeech_whisper": 0.03,
            "whisper_large_v3": 0.07,
            "setswana_whisper_ft": 0.05,
        }
        drop_repeat_engine = None
    else:
        noise_levels = {
            "whisper_zulu": 0.03,
            "lelapa": 0.05,
            "mms": 0.12,
        }
        drop_repeat_engine = "mms"

    transcripts_by_engine = {}
    for engine, noise in noise_levels.items():
        engine_transcripts = []
        for base in base_transcripts:
            words = base.split()
            noisy_words = [_add_noise(w, noise) for w in words]
            # The weakest demo engine also sometimes drops or repeats words.
            if engine == drop_repeat_engine:
                final_words = []
                for w in noisy_words:
                    if random.random() < 0.03:
                        continue              # dropped word
                    if random.random() < 0.02:
                        final_words.append(w)  # repeated word
                    final_words.append(w)
                noisy_words = final_words
            engine_transcripts.append(" ".join(noisy_words))
        transcripts_by_engine[engine] = engine_transcripts

    print(f"[Demo] Generated {n_episodes} synthetic episodes per engine")
    for eng, trs in transcripts_by_engine.items():
        avg_words = np.mean([len(t.split()) for t in trs])
        print(f"  {eng}: avg {avg_words:.0f} words/episode")

    return transcripts_by_engine


def get_episode_labels(n_episodes: int) -> list[str]:
    """Return human-readable episode labels."""
    return [f"Episode {i+1:02d}" for i in range(n_episodes)]
