import argparse
from pathlib import Path

from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def main():
    parser = argparse.ArgumentParser(description="Merge a Whisper LoRA adapter into the base model.")
    parser.add_argument("--model-name", default="openai/whisper-small")
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/whisper-small-lora-merged"))
    args = parser.parse_args()

    base = WhisperForConditionalGeneration.from_pretrained(args.model_name)
    model = PeftModel.from_pretrained(base, args.adapter)
    merged = model.merge_and_unload()
    processor = WhisperProcessor.from_pretrained(args.adapter)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(args.out_dir)
    processor.save_pretrained(args.out_dir)
    print(f"Saved merged model to {args.out_dir}")


if __name__ == "__main__":
    main()
