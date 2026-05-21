# COS760 NLP Project – Step-by-Step Coaching Guide
## Group 37: ASR + Topic Modelling for Setswana COVID-19 Podcasts

> **How to use this guide:** Read each phase before you touch the code.
> The "Why" sections are not padding — they explain decisions that will
> come up in your report. The "Junior trap" callouts flag the mistakes
> almost every first-timer makes.

---

## OVERVIEW: What you're actually building

```
Audio Files (MP3/WAV)
       │
       ▼
┌─────────────────────────────────────────┐
│  STAGE 1: ASR TRANSCRIPTION             │
│  ┌──────────┐ ┌──────────┐ ┌─────────┐ │
│  │ Whisper  │ │  Lelapa  │ │wav2vec  │ │
│  └────┬─────┘ └────┬─────┘ └────┬────┘ │
└───────┼─────────────┼─────────────┼─────┘
        │             │             │
        ▼             ▼             ▼
┌─────────────────────────────────────────┐
│  STAGE 2: TEXT PREPROCESSING            │
│  lowercase → tokenise → stopwords       │
│  → dictionary correction (Levenshtein)  │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  STAGE 3: TOPIC MODELLING (3×3 grid)    │
│  ┌─────┐ ┌──────────┐ ┌─────┐          │
│  │ LDA │ │ BERTopic │ │ NMF │          │
│  └─────┘ └──────────┘ └─────┘          │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  STAGE 4: EVALUATION                    │
│  NPMI coherence + UMass + Diversity     │
│  Error propagation (Δ NPMI ±correction) │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  STAGE 5: VISUALISATION                 │
│  Bar charts, pyLDAvis, word clouds,     │
│  BERTopic maps, heatmaps                │
└─────────────────────────────────────────┘
```

**The core research question:** Does ASR quality (Whisper > Lelapa > wav2vec) 
directly affect how coherent the topics are? And does simple dictionary-based 
post-processing recover some of the lost coherence?

---

## PHASE 0: Environment Setup

### Step 0.1 – Create the project directory

```bash
mkdir -p cos760_group37/{data/audio,data/transcripts,outputs/{figures,results,coherence},src}
cd cos760_group37
```

Copy all the provided Python files into position:
- `src/asr_module.py`
- `src/preprocessing.py`
- `src/topic_modeling.py`
- `src/evaluation.py`
- `src/visualization.py`
- `src/demo_data.py`
- `src/native_speaker_eval.py`
- `pipeline.py`
- `requirements.txt`

### Step 0.2 – Set up Python environment

Always use a virtual environment. This prevents library version conflicts
between your project and other things on your machine.

```bash
python -m venv venv

# Linux/Mac:
source venv/bin/activate

# Windows:
venv\Scripts\activate
```

> **Why a venv?** Gensim, BERTopic, and Whisper all have different
> dependency trees. Without isolation, `pip install bertopic` will
> sometimes silently downgrade something Whisper needs. A venv means
> only your project's packages exist in this environment.

### Step 0.3 – Install dependencies

```bash
pip install -r requirements.txt
```

This will take 10–20 minutes on first run (downloading PyTorch, 
transformers, etc.). Get coffee.

> ⚠️ **Junior trap:** Don't `pip install` packages one at a time as you
> need them. Install everything from requirements.txt upfront. Mixing
> installation orders causes silent version conflicts that only show up
> at 2am when you're running experiments.

### Step 0.4 – Verify the installation

```python
python -c "
import whisper, gensim, bertopic, sklearn, torch
print('Whisper:', whisper.__version__ if hasattr(whisper, '__version__') else 'ok')
print('Gensim:', gensim.__version__)
print('BERTopic:', bertopic.__version__)
print('Torch:', torch.__version__)
print('GPU available:', torch.cuda.is_available())
"
```

If all imports succeed without error, you're ready.

### Step 0.5 – Create .env file for Lelapa API key

```bash
# Create a .env file (do NOT commit this to git)
echo "LELAPA_API_KEY=your_key_here" > .env
echo ".env" >> .gitignore
```

Load it at the top of your notebook/script:
```python
from dotenv import load_dotenv
load_dotenv()  # reads .env into os.environ
```

> **Why .env?** Never hardcode API keys in code. If you push to GitHub
> with a key in the code, bots scrape it within minutes. Using .env
> + .gitignore is a professional habit that protects you.

---

## PHASE 1: Getting and Preparing the Data

### Step 1.1 – Obtain the podcast corpus

Contact Dr. Marivate / the DSFSI lab for access to the Setswana COVID-19 
podcast corpus. Place audio files in `data/audio/`.

Expected format: `.mp3`, `.wav`, `.ogg`, or `.m4a`

> **While waiting for the corpus:** Run in demo mode first (see Step 1.3).
> This lets you build and test the entire pipeline before the real data
> arrives. This is senior dev thinking: never block your own progress.

### Step 1.2 – Inspect your audio files

Before running anything, know your data:

```bash
# Count files
ls data/audio/ | wc -l

# Check durations (requires ffmpeg)
for f in data/audio/*; do
    duration=$(ffprobe -i "$f" -show_entries format=duration \
               -v quiet -of csv="p=0" 2>/dev/null)
    echo "$f: ${duration}s"
done
```

Write down in your lab notebook:
- How many episodes?
- Average duration?
- Audio quality (bitrate, sample rate)?
- Are there multiple speakers?

> **Why this matters for your report:** The corpus statistics section
> needs these numbers. Also, very short episodes (<2 minutes) may not 
> produce enough tokens for meaningful topic models.

### Step 1.3 – Run the demo pipeline first

Before touching real audio, validate the entire pipeline works end-to-end:

```bash
python pipeline.py --mode demo --n-episodes 12 --n-topics 7
```

Expected output:
```
STAGE 1: ASR TRANSCRIPTION
[Demo] Generated 12 synthetic episodes per engine
...
STAGE 4: EVALUATION
engine  method  n_topics  npmi    umass  diversity
whisper lda     7         0.XXX   -X.XX  0.XX
...
PIPELINE COMPLETE
Figures: outputs/figures/
```

Open `outputs/figures/npmi_comparison.png`. If you see a grouped bar
chart with three engine groups and three methods per group, the pipeline
is working.

> **Why run demo first?** Because debugging a broken pipeline on real
> data is 10× harder. With demo data you know exactly what the input
> is. Always test with controlled, known data first.

---

## PHASE 2: ASR Transcription

### Step 2.1 – Understand what each ASR engine is doing

**Whisper (OpenAI, `large-v3`)**
- Transformer encoder-decoder trained on 680,000 hours of multilingual audio
- Has explicit Setswana (`tn`) language support
- Best general-purpose quality
- Slowest (~1× realtime on CPU, 10× on GPU)

**Lelapa AI API**
- African-language specialist API
- Trained on more African speech data
- May handle Setswana morphology and tonal patterns better than Whisper
- Requires API key and internet connection
- Results are async (you upload, then poll)

**wav2vec-2.0 / AfroXLSR**
- Self-supervised model pre-trained on multilingual speech (XLS-R)
- Fine-tuned adapter for Setswana
- Fast once loaded, runs locally
- Highest expected error rate in zero-shot scenarios
- Good baseline to show how much specialised training matters

> **This comparison is the scientific contribution of your project.**
> You're not just transcribing — you're measuring how different
> training approaches affect downstream NLP utility.

### Step 2.2 – Run Whisper on real audio

```python
# In a Python script or Jupyter cell:
import sys
sys.path.insert(0, "src")
from asr_module import transcribe_whisper

result = transcribe_whisper("data/audio/episode_01.mp3", language="tn")
print(result["transcript"][:500])   # first 500 chars
print(f"Segments: {len(result['segments'])}")
```

> ⚠️ **Junior trap – Language code:** Pass `language="tn"` explicitly.
> Without it, Whisper auto-detects and often guesses English or Zulu for
> Setswana speech, which produces garbage transcripts. Always force the
> language code when you know it.

### Step 2.3 – Set up and test Lelapa API

First, test with a short audio clip:

```python
from asr_module import transcribe_lelapa
import os

os.environ["LELAPA_API_KEY"] = "your_actual_key"

# Test with a short clip first (< 30 seconds)
result = transcribe_lelapa("data/audio/test_short.wav")
print(result["transcript"])
```

If you get a 401 error: your API key is wrong.
If you get a 422 error: the audio format is wrong — make sure it's 16kHz mono WAV.

> **Check the Lelapa docs for the current language code for Setswana.**
> It may be `"tsn"`, `"tn"`, or `"sot"` depending on their API version.
> Update `language_code` in `asr_module.py` accordingly.

### Step 2.4 – Run the full batch transcription

```bash
# With Lelapa:
python pipeline.py --mode real

# Without Lelapa (if API key not available yet):
python pipeline.py --mode real --skip-lelapa

# Without wav2vec (if running on CPU and it's too slow):
python pipeline.py --mode real --skip-wav2vec
```

Transcripts are cached as JSON in `data/transcripts/`. 
If a file already exists, it won't re-run the ASR (saves time).

> **Pro tip:** Transcription is the most expensive stage. Run it once,
> save results, and iterate on preprocessing/topic modelling without
> re-running ASR. The caching in `run_all_engines()` handles this
> automatically.

### Step 2.5 – Inspect transcripts for quality

After transcription, do a quick manual review:

```python
import json

# Compare the same episode across engines
for engine in ["whisper", "lelapa", "wav2vec"]:
    with open(f"data/transcripts/episode_01_16k_{engine}.json") as f:
        data = json.load(f)
    print(f"\n=== {engine.upper()} ===")
    print(data["transcript"][:300])
```

Note down (for your report):
- Which engine sounds most coherent to the native speakers?
- Are there recurring error patterns (e.g., Whisper confuses certain phonemes)?
- Does wav2vec produce completely garbled output or just occasional errors?

---

## PHASE 3: Text Preprocessing

### Step 3.1 – Understand why each step matters

| Step | What it does | Why it matters |
|------|-------------|----------------|
| Lowercase | `BOLWETSI` → `bolwetsi` | Same word, different case = different token in bag-of-words |
| Remove punct | `bolwetsi,` → `bolwetsi` | ASR sometimes inserts punctuation mid-word |
| Tokenise | Split on spaces | Creates the unit of analysis |
| Remove stopwords | Remove `ke`, `le`, `the` | Function words dominate term freq and pollute topics |
| Dictionary correction | `vaccin` → `vaccine` | Reduces ASR noise before building topics |

### Step 3.2 – Extend the stopword list

Open `src/preprocessing.py` and look at `SETSWANA_STOPWORDS`. 

After you have real transcripts, run this to find high-frequency words:

```python
from collections import Counter
from preprocessing import clean_text, tokenise

# Load your whisper transcripts
with open("outputs/results/preprocessed_whisper.json") as f:
    data = json.load(f)

all_tokens = []
for text in data["raw_texts"]:
    all_tokens.extend(text.split())

print("Top 50 tokens:")
for word, count in Counter(all_tokens).most_common(50):
    print(f"  {word}: {count}")
```

Look at the top 30. Any that are clearly not content words (short articles,
conjunctions, discourse markers) should be added to `SETSWANA_STOPWORDS`.

> **Ask Kagiso and Shaun** to review this list! They are the native
> speakers. They will immediately recognise which high-frequency words
> are meaningless for topic analysis.

### Step 3.3 – Understand the dictionary correction

The correction works by building a "trusted vocabulary" from words that
appear frequently across all documents. Single-occurrence words are assumed
to be ASR errors and replaced with the closest frequent word.

```python
# Test it manually:
from preprocessing import build_vocab, correct_token

vocab = {"vaccine", "lockdown", "hospital", "community", "testing"}
print(correct_token("vaccin", vocab))    # → "vaccine"
print(correct_token("lochdown", vocab))  # → "lockdown"
print(correct_token("ospital", vocab))   # → "hospital"
```

> ⚠️ **Limitation to discuss in your report:** This approach only works
> if the correct word appears frequently enough to be in the vocabulary.
> For a small corpus (12 episodes), some legitimate content words may
> appear only once and get incorrectly "corrected". Acknowledge this
> as a limitation.

---

## PHASE 4: Topic Modelling

### Step 4.1 – LDA: what it is and what to tune

LDA (Latent Dirichlet Allocation) is a probabilistic model that assumes
each document is a mixture of topics, and each topic is a mixture of words.

**Key parameters:**
- `num_topics` (k): the number of topics — most important decision
- `passes`: how many training iterations (more = better, slower)
- `alpha`: document-topic prior (auto-learns from data)
- `eta`: topic-word prior (auto-learns from data)

**Choosing k:** The code searches a range (3–15) and picks the k with
highest c_v coherence score. You can narrow this range based on your
intuition about how many distinct themes the podcasts cover.

COVID-19 podcast themes you'd expect:
- Medical/clinical (symptoms, treatment, ICU)
- Government policy (lockdown regulations, vaccine rollout)
- Community impact (job losses, food security)
- Prevention (masking, sanitising, social distance)
- Vaccination (rollout, hesitancy, side effects)

So k=5 to k=8 is probably sensible. The search will find the sweet spot.

### Step 4.2 – BERTopic: what's different and why it's robust to noise

BERTopic doesn't use word counts. Instead:
1. It converts each document to a dense vector using a sentence transformer
2. It clusters documents with HDBSCAN
3. It finds the most distinctive words per cluster using c-TF-IDF

**Why this matters for noisy ASR text:** If Whisper misrecognises "bolwetsi"
(disease) as "bolwesti", BERTopic still groups that document with other
health documents because the *surrounding context* is encoded in the embedding.
LDA would create a separate term "bolwesti" that fragments topic coherence.

**The embedding model:** `paraphrase-multilingual-MiniLM-L12-v2` supports
50+ languages. It has been trained on multilingual sentence pairs and
understands Sotho/Tswana language family patterns. This is crucial.

> **Junior trap:** Don't use an English-only embedding model like
> `all-MiniLM-L6-v2`. It will convert Setswana words to near-random
> vectors, destroying the clustering completely.

### Step 4.3 – NMF: simpler but often cleaner topics

NMF (Non-negative Matrix Factorisation) decomposes the document-term
TF-IDF matrix into two non-negative matrices: document-topic and topic-term.

Why it often gives cleaner topics:
- No probabilistic assumptions (unlike LDA)
- Deterministic (same input → same output every time)
- Tends to produce more focused, sparse topics
- Faster than LDA on small corpora

The tradeoff: NMF assumes topics are orthogonal (non-overlapping), which 
may not match how Setswana COVID discourse actually works (themes overlap
significantly in health policy discussions).

### Step 4.4 – Run topic modelling

The pipeline runs all three automatically. But to test/debug individually:

```python
import sys
sys.path.insert(0, "src")
from topic_modeling import run_lda, run_bertopic, run_nmf
from preprocessing import preprocess_transcripts

# Load your transcripts (example with whisper)
import json
with open("outputs/results/preprocessed_whisper.json") as f:
    data = json.load(f)

token_lists_raw = [text.split() for text in data["corrected_texts"]]
joined_strings  = data["corrected_texts"]

# Run LDA
lda_result = run_lda(token_lists_raw, n_topics=7)

# Print topics
from evaluation import print_topics
print_topics(lda_result)
```

### Step 4.5 – Interpreting topic output

For each topic, you get the top-N words by probability/weight.
A "good" topic has words that clearly belong together semantically.

Good topic example:
```
Topic 03: vaccine, vaccination, booster, immunity, antibody, dose, rollout, jab, hesitancy, clinic
```
→ Clearly a "vaccination campaign" topic. NPMI will be high.

Bad topic example (from noisy ASR):
```
Topic 07: vaccin, lochdown, ospital, ke, tsting, bolwesti, the, go, quarantine, mo
```
→ Mix of corrupted words, stopwords that slipped through, and unrelated terms.
NPMI will be low. This is what you're trying to prevent.

---

## PHASE 5: Evaluation

### Step 5.1 – Understanding NPMI

NPMI (Normalised Pointwise Mutual Information) measures how often topic
words appear together in the same documents compared to chance.

- Range: −1 to +1
- Higher is better (words appear together more than by chance)
- Typical good LDA NPMI: 0.05 – 0.25
- Typical bad NPMI (noisy text): −0.05 – 0.02

Formula intuition: if "vaccine" and "vaccination" always appear in the
same documents, their PMI is high (they're informationally related).

### Step 5.2 – Understanding UMass

UMass measures, for each pair of topic words (w1, w2), how often the
less frequent one (w2) appears in documents that already contain w1.

- Range: typically −∞ to 0
- Higher (closer to 0) is better
- Less susceptible to corpus size than NPMI

> **For your report:** Primary metric = NPMI. Secondary = UMass.
> Use both because they measure slightly different things and together
> give a more complete picture.

### Step 5.3 – Running evaluation

```python
from evaluation import evaluate_all_conditions, print_summary_table

# Assuming all_results and token_lists_by_engine are from the pipeline
eval_df = evaluate_all_conditions(all_results, token_lists_by_engine)
print_summary_table(eval_df)
```

Expected table structure:
```
engine   method    n_topics  npmi    umass    diversity
whisper  lda       7         0.1234  -3.456   0.847
whisper  bertopic  5         0.1567  -2.890   0.912
whisper  nmf       7         0.0987  -4.123   0.823
lelapa   lda       7         ...
...
```

### Step 5.4 – Error propagation analysis

This is your methodological innovation: instead of WER (which requires 
ground truth transcripts you don't have), you measure how much coherence 
improves after dictionary correction.

```
Δ NPMI = NPMI_after_correction − NPMI_before_correction
Gain % = (Δ NPMI / |NPMI_before|) × 100
```

If Whisper shows Δ NPMI = +0.02 (2% gain) but wav2vec shows Δ NPMI = +0.08
(8% gain), this tells you:
- wav2vec has more correctable noise
- Dictionary correction is more valuable for noisier engines
- This indirectly measures transcription quality without ground truth

> **This is a key contribution to report.** Cite this reasoning explicitly
> in Section 3 (Methodology) when explaining why you use NPMI delta 
> instead of WER.

---

## PHASE 6: Visualisation

### Step 6.1 – The comparison bar chart (most important figure)

`outputs/figures/npmi_comparison.png` is your headline figure.
It shows all 9 conditions (3 engines × 3 methods) in one chart.

When writing your results section, structure it around this figure:
1. Which engine achieves highest NPMI? (expected: Whisper)
2. Which method achieves highest NPMI across engines? (expected: BERTopic)
3. How large is the gap between best and worst? (quantify it)

### Step 6.2 – pyLDAvis (interactive topic exploration)

Open `outputs/figures/pyldavis_whisper.html` in a browser.

**How to read pyLDAvis:**
- Each bubble = one topic
- Bubble size = relative prevalence in corpus
- Distance between bubbles = semantic distance
- Right panel = most relevant words for selected topic (adjust λ slider)

Set λ=0.6 (not 1.0) to balance frequency and exclusivity — this often
gives more interpretable top words.

Take screenshots of the pyLDAvis for your report. Describe what you see:
"Topics 1 and 3 overlap substantially, suggesting the COVID-clinical and
government-policy themes frequently co-occur in the corpus."

### Step 6.3 – BERTopic visualisations

`outputs/figures/bertopic_whisper/bertopic_topic_map.html` shows topics
as a 2D scatter plot (UMAP reduction). Nearby topics are semantically related.

`bertopic_barchart.html` shows word importances per topic — easier to
interpret than pyLDAvis for non-technical readers.

### Step 6.4 – Word clouds

For each topic, the word cloud visually weights the most important words.
Useful for:
- Quick visual interpretation
- Presentation slides
- Sharing with native speaker reviewers who find raw numbers less intuitive

### Step 6.5 – Creating professional figures for the report

For publication-quality figures:

```python
import matplotlib.pyplot as plt
plt.rcParams.update({
    "figure.dpi": 300,          # High DPI for print
    "font.family": "serif",     # More academic look
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
})
```

Save as PDF for LaTeX, PNG for Word:
```python
fig.savefig("outputs/figures/npmi_comparison.pdf", bbox_inches="tight")
fig.savefig("outputs/figures/npmi_comparison.png", dpi=300, bbox_inches="tight")
```

---

## PHASE 7: Native Speaker Evaluation

### Step 7.1 – Running the interactive evaluation tool

Kagiso and Shaun run:

```bash
# Review best-performing configuration for Whisper:
python src/native_speaker_eval.py --engine whisper --method bertopic

# Review Lelapa:
python src/native_speaker_eval.py --engine lelapa --method lda
```

The tool walks through each topic interactively and saves responses to CSV.

### Step 7.2 – What to look for during review

For each topic, the native speaker should ask:
1. Do these words belong together in Setswana discourse?
2. Are there misrecognised agglutinative words?
   - Setswana is agglutinative: "go bua" (to speak), "ga ke bue" (I don't speak)
   - ASR often splits or joins these incorrectly
3. Are there tonal confusions? 
   - Setswana has lexical tone — ASR may confuse minimal pairs
4. Does the topic make sense in the context of COVID-19 health communication?

### Step 7.3 – Using native speaker feedback in the report

Structure the qualitative analysis section:
1. Present the top 10 words for the best topic per engine
2. Quote the native speaker rating and theme label
3. Describe specific error types observed
4. Connect to your quantitative NPMI finding

Example paragraph (template):
> "For the Whisper + BERTopic configuration (NPMI = 0.15), native 
> speaker K. Tsiane rated 6 of 7 topics as 'Clear', identifying 
> distinct themes including vaccination campaigns, symptom awareness,
> and government restrictions. Two error patterns were documented:
> (1) agglutinative morpheme fragmentation in low-frequency words,
> and (2) English code-switch boundaries occasionally missed,
> producing compound tokens such as 'vaccinele' (vaccine+locative)."

---

## PHASE 8: Writing the Report

### Step 8.1 – Section structure and what each needs

**Section 1: Introduction (~400 words)**
- State: low-resource African language ASR + topic modelling gap
- Motivate: Setswana has 4M+ speakers; COVID health comms are vital
- Contributions: empirical comparison of 3 ASR × 3 topic models on a
  real corpus; NPMI-delta metric as substitute for WER; public release
- Responsible NLP: underrepresentation of African languages in NLP;
  your work contributes to closing this gap

**Section 2: Background (~600 words)**
- Summarise: wav2vec 2.0, Whisper, AfroSpeech, Lelapa
- Topic models: LDA (Blei 2003), BERTopic (Grootendorst 2022), NMF
- Gap: no prior work links ASR error types to topic coherence for Setswana
- Responsible NLP: acknowledge that prior datasets (AfriSpeech) are
  small and may not represent all Setswana dialects

**Section 3: Methodology (~800 words)**
- Data: DSFSI corpus (n episodes, avg duration, content domain)
- Pipeline diagram (include the ASCII art from this guide or a Mermaid diagram)
- ASR setup: exact model versions, language codes, hardware
- Preprocessing: each step justified
- Topic models: hyperparameters and rationale
- Evaluation: NPMI + UMass + NPMI-delta (justify why not WER)
- Responsible NLP: no ground truth means qualitative validation is essential;
  native speaker evaluation design; data access and licensing

**Section 4: Experiments & Results (~600 words + figures)**
- Lead with Table 1: full 9-condition coherence scores (NPMI, UMass, Diversity)
- Figure 1: NPMI comparison bar chart
- Figure 2: Error propagation chart
- Figure 3: pyLDAvis or BERTopic topic map (best config)
- Discuss: which engine wins, which method wins, interaction effects
- Responsible NLP: did all engines struggle equally with Setswana-specific
  morphology? What do the error patterns reveal about model limitations?

**Section 5: Discussion (~400 words)**
- What worked: BERTopic (expected to be most robust)
- What didn't: LDA fragmentation on noisy wav2vec transcripts
- Lessons: correction step matters most for the noisiest engine
- Responsible AI discussion:
  - Data openness: will release transcripts + code publicly
  - Fairness: all models underperform vs English — important caveat
  - Risks: incorrect health information if deployed without validation
  - Impact: demonstrates feasibility of Setswana health topic monitoring

**Section 6: Conclusion (~200 words)**
- Key finding: X engine + Y method yields best coherence (NPMI = Z)
- Dictionary correction provides N% gain for noisiest engine
- Practical takeaway: Whisper/Lelapa + BERTopic is viable without fine-tuning
- Future work: ground truth transcripts, fine-tuning, more episodes

### Step 8.2 – Writing the Results section (avoiding common mistakes)

❌ **Don't write:** "The NPMI score for Whisper LDA was 0.1234."
✅ **Do write:** "Whisper + LDA achieved the highest coherence (NPMI = 0.12),
outperforming the wav2vec baseline by 0.04 points, suggesting that
transcription quality directly propagates to topic interpretability."

❌ **Don't just describe the table. Interpret it.**
Every number needs a sentence explaining what it means scientifically.

### Step 8.3 – Tables and figures

**Table 1 template:**

| Engine  | Method   | Topics | NPMI ↑  | UMass ↑ | Diversity ↑ |
|---------|----------|--------|---------|---------|-------------|
| Whisper | LDA      | 7      | 0.XXX   | -X.XX   | 0.XX        |
| Whisper | BERTopic | 6      | **0.XXX** | -X.XX | 0.XX        |
| Whisper | NMF      | 7      | 0.XXX   | -X.XX   | 0.XX        |
| Lelapa  | LDA      | 7      | 0.XXX   | -X.XX   | 0.XX        |
| ...     | ...      | ...    | ...     | ...     | ...         |

Bold the best value per column. Add a footnote: "↑ = higher is better."

### Step 8.4 – Responsible NLP reflections (required by rubric)

Don't treat these as an afterthought. Weave them through the paper:

- **Introduction:** Why African language NLP matters (equity argument)
- **Background:** What gaps exist specifically for Setswana (be specific, not generic)
- **Methodology:** Data sourcing ethics, consent, licensing; how you handled bias
- **Results:** Did engines fail in systematic ways related to Setswana morphology?
- **Discussion:** What are the risks if this system is deployed in healthcare communication?

---

## PHASE 9: Running the Complete Pipeline

### Complete workflow commands

```bash
# ── DEMO MODE (test everything works) ──────────────────────
python pipeline.py --mode demo --n-episodes 12 --n-topics 7

# ── REAL DATA (all engines) ─────────────────────────────────
python pipeline.py --mode real --n-topics 7

# ── REAL DATA (no Lelapa API yet) ───────────────────────────
python pipeline.py --mode real --n-topics 7 --skip-lelapa

# ── REAL DATA (CPU only, no wav2vec) ────────────────────────
python pipeline.py --mode real --n-topics 7 --skip-wav2vec

# ── Native speaker review (after pipeline runs) ─────────────
python src/native_speaker_eval.py --engine whisper --method bertopic
python src/native_speaker_eval.py --engine lelapa  --method lda
```

### Key output files to reference in report

| File | What it is |
|------|-----------|
| `outputs/coherence/coherence_scores.csv` | Table 1 data |
| `outputs/coherence/error_propagation.csv` | Error analysis data |
| `outputs/figures/npmi_comparison.png` | Figure 1 |
| `outputs/figures/error_propagation.png` | Figure 2 |
| `outputs/figures/pyldavis_whisper.html` | Interactive LDA viz |
| `outputs/figures/bertopic_whisper/` | BERTopic interactive viz |
| `outputs/results/topics_*.json` | Topic words for native speaker review |
| `outputs/results/native_speaker_review_template.txt` | Print this for Kagiso & Shaun |
| `outputs/results/review_*.csv` | Native speaker completed reviews |

---

## TROUBLESHOOTING

### "Gensim dictionary too small" warning
→ You have very few episodes or very short transcripts.
→ In demo mode: increase `--n-episodes` to 20.
→ In real mode: check that ASR actually produced output (check transcript files).

### BERTopic produces only 1-2 topics
→ Your corpus is too small or too homogeneous.
→ Decrease `min_topic_size` in `run_bertopic()` to 1 or 2.
→ Increase `--n-episodes` if in demo mode.

### wav2vec runs for hours on CPU
→ Normal — wav2vec is slow on CPU for long audio.
→ Either use GPU, or use `--skip-wav2vec` and note this limitation in report.
→ Alternatively, process only the first 60 seconds of each episode for comparison.

### Lelapa API returns 401 Unauthorized
→ Your API key is wrong or expired. Check .env file.
→ Make sure there's no trailing space or newline in the key.

### pyLDAvis throws ValueError
→ Usually happens when topics have words not in the dictionary.
→ Re-run LDA with `passes=30` to get a more stable model.
→ Or use the model saved from the pipeline run directly.

### NPMI scores are all negative or very low
→ This is normal for small corpora with noisy text.
→ The *relative* comparison across conditions is what matters, not absolute values.
→ Cite Röder et al. (2015) who note that NPMI ranges vary by corpus size.

---

## WORKPLAN CHECKLIST

| Date | Task | Owner | Done? |
|------|------|-------|-------|
| Apr 20–27 | Set up repo, install libs, run demo mode | All | ☐ |
| Apr 27 | Obtain Lelapa API key | Josh | ☐ |
| Apr 27 – May 3 | Run ASR on real corpus, inspect transcripts | Josh | ☐ |
| May 3 | Extend Setswana stopword list with Kagiso & Shaun | Kagiso | ☐ |
| May 4–14 | Run full pipeline, tune n_topics | Kagiso | ☐ |
| May 14 | Kagiso & Shaun complete native speaker review | Kagiso + Shaun | ☐ |
| May 14–21 | Write results and discussion sections | Shaun + All | ☐ |
| May 21–25 | Final report + code cleanup + submission | All | ☐ |

---

*Guide written for COS760 Group 37 – Josh Cloete, Kagiso Tsiane, Shaun Seabo*
