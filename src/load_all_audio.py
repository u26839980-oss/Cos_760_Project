import os
from datasets import load_dataset
import soundfile as sf

# 1. Define output directory path
OUTPUT_DIR = "data/audio"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading dataset from Hugging Face...")
# 2. Load the isiZulu (zul) dataset configuration using streaming mode
# Streaming is highly recommended as speech datasets are very large.
dataset = load_dataset(
    "dsfsi-anv/za-african-next-voices", 
    "tsn", 
    split="train", 
    streaming=True
)

# 3. Fetch and save the first 20 audio files
count = 0
max_files = 100

print(f"Starting download of {max_files} audio files...")

for sample in dataset:
    if count >= max_files:
        break
        
    audio_data = sample["audio"]
    
    # Extract audio components
    audio_array = audio_data["array"]
    sampling_rate = audio_data["sampling_rate"]
    
    # Generate a safe filename using the sample index or given IDs if available
    filename = f"tsn_sample_{count + 1}.wav"
    file_path = os.path.join(OUTPUT_DIR, filename)
    
    # Write the numpy array out as a WAV file
    sf.write(file_path, audio_array, sampling_rate)
    print(f"[{count + 1}/{max_files}] Saved: {file_path}")
    
    count += 1

print("\nProcessing complete! All 20 files are saved in 'src/data/audio'.")
