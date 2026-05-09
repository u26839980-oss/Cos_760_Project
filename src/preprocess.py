import string, re 

def clean(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", " ", text)
    text = re.sub(r"\d+", " ", text)

    stopwords = set(["le", "a", "o", "ba", "e", "ka", "wa", "mo", "go","ke"])
    tokens = [t for t in text.split() if t not in stopwords and len(t) > 1]
    return " ".join(tokens) 