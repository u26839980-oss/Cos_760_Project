import whisper

def transcribe_whisper(audio_path, model_size="large-v3"):
    model = whisper.load_model(model_size)
    result = model.transcribe(audio_path, language="tn")
    print("YO")
    return result["text"]

if __name__ == "__main__":
    transcribe_whisper('data/raw_audio/audio')