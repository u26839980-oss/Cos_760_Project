import soundfile as sf 
import librosa
import torch 
import numpy as np
import requests
import logging
import json
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

#Load in audio and normalises it to mono, 16000HZ and compatible audio_file type#
def load_audio(file_path: str, target_sr: int=16000) -> tuple[np.ndarray, int]:
    logger.info(f"Loading Audio File @{file_path}")
    audio,sr = librosa.load(file_path, sr=target_sr, mono=True)
    logger.info(f"Duration & Sample Rate: {len(audio)}, {sr}HZ")
    return audio,sr

# transcript asr pipeline for OpenAi's whisper ASR #
def transcribe_whisper(audio_path, model_size="large-v3"):
    import whisper
    model = whisper.load_model(model_size)
    result = model.transcribe(audio_path, language="tn")
    print("YO")
    return result["text"]

def transcribe_vec()

if __name__ == "__main__":
    transcribe_whisper('data/raw_audio/audio')