import os
import warnings

from datasets import load_dataset  
from huggingface_hub import login
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=UserWarning)

load_dotenv()

token = os.environ.get("HF_ACCESS_TOKEN")
login(token=token)
data_set = load_dataset("dsfsi-anv/za-african-next-voices", "tsn", split="train", streaming=True)

for i, sample in enumerate(data_set):
    if i >= 10: break
    # Access metadata without decoding audio
    print(f"Sample {i}: {sample['audio']}")