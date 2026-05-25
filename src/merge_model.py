"""
merge_kesego_model_fixed.py
Merge LoRA adapter into Whisper Turbo, handling tokenizer size mismatch.
"""

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel

BASE_MODEL = "openai/whisper-large-v3-turbo"
ADAPTER   = "kesbeast23/whisper-large-turbo-setswana-lora"
OUTPUT_DIR = "models/kesego_whisper_setswana_merged"

# --- Load processor FROM THE ADAPTER to get its exact vocab size ---
print("Loading adapter's tokenizer/processor...")
adapter_processor = WhisperProcessor.from_pretrained(ADAPTER)
adapter_vocab_size = len(adapter_processor.tokenizer)
print(f"Adapter tokenizer vocab size: {adapter_vocab_size}")

# --- Load base model ---
print("Loading base model...")
model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL)
base_vocab_size = model.config.vocab_size
print(f"Base model vocab size: {base_vocab_size}")

# --- Resize base model embeddings to match adapter ---
if adapter_vocab_size != base_vocab_size:
    print(f"Resizing base model from {base_vocab_size} to {adapter_vocab_size} tokens...")
    model.resize_token_embeddings(adapter_vocab_size)
    # The output projection layer (proj_out) is also resized automatically by Whisper's resize logic.
else:
    print("Vocab sizes already match, no resizing needed.")

# --- Load adapter ---
print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, ADAPTER)

# --- Merge and save ---
print("Merging adapter into base model...")
merged_model = model.merge_and_unload()

print("Saving merged model...")
merged_model.save_pretrained(OUTPUT_DIR)

# Save the adapter's processor (which matches the merged model's tokenizer)
adapter_processor.save_pretrained(OUTPUT_DIR)

print(f"✅ Merged model saved to '{OUTPUT_DIR}'")