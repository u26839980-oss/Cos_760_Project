import soundfile as sf 
import librosa
import torch 
import numpy as np
import requests
import logging
import time
import json
import os
import whisper

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

#Load in audio and normalises it to mono, 16000HZ and compatible audio_file type#
def load_audio(file_path: str, target_sr: int=16000) -> tuple[np.ndarray, int]:
    logger.info(f"Loading Audio File @{file_path}")
    audio,sr = librosa.load(file_path, sr=target_sr, mono=True)
    logger.info(f"Duration & Sample Rate: {len(audio)}, {sr}HZ")
    return audio,sr

# takes numpy array and converts to .wav format neede for wav2vec and Lelapa transcribe api #
def save_wav(audio: np.ndarray, sr: int, outbound_path: str) -> str:
    sf.write(outbound_path, audio, sr, subtype="PCM_16")
    logger.info(f".wav saved to {outbound_path}")

# transcript asr pipeline for OpenAi's whisper ASR #
def transcribe_whisper(audio_path: str, model_size: str="large-v3", language: str = "tn") -> dict:
    logger.info(f"Launching Whisper using {model_size}")
    model = whisper.load_model(model_size)
    real_path = os.path.basename(audio_path)

    logging.info(f"Whisper transcribing for {real_path} ...")
    
    result = model.transcribe(
        audio_path,
        language=language,
        verbose=False,
        word_timestamps=False
    )

    segments = [
        {
            "start": s["start"],
            "end": s["end"],
            "text": s["text"].strip()
        }
        for s in result.get("segments", [])
    ]

    return{
        "engine": "whisper",
        "file": audio_path,
        "transcript": result["text"].strip(),
        "segment": segments,
        "language_found": result.get("language", language)
    }

    # model = whisper.load_model(model_size)
    # result = model.transcribe(audio_path, language="tn")
    # print("YO")
    # return result["text"]

LELAPA_API_BASE = "https://api.lelapa.ai/v1"

# UPDATE: Lelapa doesn't support Tswana, Lelapa option &Deprecated&, nvm language will just be auto detected
# transcript pipeline for lelapa api #
def transcribe_lelapa(audio_path: str, api_key: str | None = None, language_code = None) -> dict:
    key = api_key or os.environ.get("LELAPA_API_KEY", "")
    if not key:
        raise EnvironmentError(
            "Lelapa Api key not found or detected. Ensure use of .env file for key under alias 'LELAPA_API_KEY'..."
        )
    
    headers = {"X-CLIENT-TOKEN": key}
    logger.info(f"Uploading Audio to Lelapa...")
    with open(audio_path, "rb") as f:
        upload_response = requests.post(
            f"{LELAPA_API_BASE}/transcribe/sync",
            headers=headers,
            files={"file": (os.path.basename(audio_path), f, "audio/wav")},
            timeout=120, #could try 150 as api docs specify max process timing 
        )
    upload_response.raise_for_status()
    upload_id = upload_response.json()["upload_id"]
    logger.info(f"Upload Id = {upload_id}")
    
    logger.info("Lelapa Transcription Starting...")
    job_response = requests.post(
        f"{LELAPA_API_BASE}/transcribe/sync", 
        headers=headers,
        json={"upload_id" : upload_id, "lang_code" : language_code}, # could be language instead of lang_code,
        timeout=60,
    )
    job_response.raise_for_status()
    job_id = job_response.json()["job_id"]
    logger.info(f"job with id: {job_id} started")

    logger.info(f"Lelapa Progress Checker...")
    for att in range(60):
        time.sleep(10)
        status_response = requests.get(
            f"{LELAPA_API_BASE}/transcribe/status/{job_id}",
            headers=headers,
            timeout=30,
        )
        status_response.raise_for_status()
        data = status_response.json()
        status_update = data.get("status", "...unknown...")
        logger.info(f"{att}.) Current Status: {status_update}")
        
        if status_update == "completed":
            transcript = data.get("transcript", "")
            segments = data.get("segments", [])
            return{
                "engine" : "Lelapa",
                "file": audio_path,
                "transcript": transcript.strip(),
                "segments": segments,
            }
        elif status_update in ("failed", "error"):
            raise RuntimeError(f"Lelapa job did not successfully finish, data received: {data} ...")
        else:
            raise RuntimeError(f"Something Unexpected Occured...")
    raise TimeoutError("Lelapa transcription timed out after 10 min limit reached...")        


# asr pipeline for wav2vec # 
# def transcribe_vec():

def main():
    audio_name = "test_audio"
    audio_type = ".wav"
    audio_pre = audio_name+audio_type
    a,s = load_audio(f"data/audio/pre/{audio_pre}")
    audio_pro = audio_name+"_pro"+".wav"
    audio_pro_path = f"data/audio/post/{audio_pro}"
    save_wav(a, s, audio_pro_path)
    print(f"{audio_pro_path}")
    transcribe_whisper(str(audio_pro_path))
    transcribe_lelapa(str(audio_pro_path))

if __name__ == "__main__":
    main()