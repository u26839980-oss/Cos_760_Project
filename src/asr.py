def transcribe_whisper(audio_path, model_size="large-v3"):
    import whisper
    model = whisper.load_model(model_size)
    result = model.transcribe(audio_path, language="tn")
    return result["text"]