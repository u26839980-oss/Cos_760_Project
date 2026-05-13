from asr import transcribe_whisper
from preprocess import clean
from gensim.models import LdaModel
from gensim.corpora import Dictionary

def main():
    print("Temp")

    # First load in off shelf open-ai whisper asr(and perhaps dsfsi's asr model, or Lelapa[but this requires an request needed api key...])
    
    ## Placeholder for network based audio getter, check out test.py ## 
    
    # My Network sucks currently so dataset is downloaded locally 

    for i in range(1): # range 1 simply since training size unspecified currently
        transcription_whisper = transcribe_whisper('../data/raw_audio/audio')
        ##dsfi asr model pass through here## 
        ##Lelapa##

        text = clean(transcribe_whisper)
        print(text)
        #feed into a dummy LDA model (using gensim) and print a couple of topics. This confirms the pipeline connects.

if __name__ == "__main__":
    main()
