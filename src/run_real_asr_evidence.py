import json
import logging
from pathlib import Path

from asr_module import run_all_engines


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    project_root = Path(__file__).resolve().parent.parent
    audio = [str(project_root / "src" / "data" / "audio" / "tsn_sample_1.wav")]
    out_dir = project_root / "src" / "outputs" / "evidence" / "real_asr_check"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = run_all_engines(audio_files=audio, out_dir=str(out_dir), language="tsn")

    summary = {}
    for engine, items in results.items():
        item = items[0] if items else {}
        transcript = item.get("transcript", "")
        summary[engine] = {
            "file": item.get("file"),
            "word_count": len(transcript.split()),
            "transcript_preview": " ".join(transcript.split()[:60]),
            "error": item.get("error"),
        }

    summary_path = out_dir / "tsn_sample_1_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary_path)


if __name__ == "__main__":
    main()
