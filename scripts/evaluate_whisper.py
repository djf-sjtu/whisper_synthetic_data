import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from opencc import OpenCC
from peft import PeftModel
from scipy import signal
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor


TARGET_SR = 16000
KEYWORDS = ["小飒", "飒智", "WAIC", "Sage Robot One", "Sage Dog", "飒智智能科技有限公司"]
REPO_ROOT = Path(__file__).resolve().parents[1]
SIMPLIFIER = OpenCC("t2s")


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_audio(path: Path):
    if not path.is_absolute():
        path = REPO_ROOT / path
    audio, sr = sf.read(path, always_2d=False)
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if sr != TARGET_SR:
        gcd = math.gcd(sr, TARGET_SR)
        audio = signal.resample_poly(audio, TARGET_SR // gcd, sr // gcd).astype(np.float32)
    return audio


def normalize_text(text: str, simplify_chinese: bool = True):
    text = text.strip()
    if simplify_chinese:
        text = SIMPLIFIER.convert(text)
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[，。？！、；：,.?!;:]", "", text)
    return text.upper()


def edit_distance(left: str, right: str):
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def keyword_stats(rows, simplify_chinese: bool = True):
    stats = {}
    for keyword in KEYWORDS:
        refs = [row for row in rows if keyword in row["text"]]
        if not refs:
            continue
        hits = [
            row
            for row in refs
            if normalize_text(keyword, simplify_chinese) in normalize_text(row["prediction"], simplify_chinese)
        ]
        stats[keyword] = {"total": len(refs), "hit": len(hits), "recall": len(hits) / len(refs)}
    return stats


def score_rows(rows, simplify_chinese: bool):
    total_edits = 0
    total_chars = 0
    exact = 0
    for row in rows:
        ref = normalize_text(row["text"], simplify_chinese)
        hyp = normalize_text(row["prediction"], simplify_chinese)
        total_edits += edit_distance(ref, hyp)
        total_chars += max(1, len(ref))
        exact += int(ref == hyp)
    return total_edits / total_chars if total_chars else 0.0, exact / len(rows) if rows else 0.0


def summarize(rows, simplify_chinese: bool = True):
    cer, exact = score_rows(rows, simplify_chinese=simplify_chinese)
    cer_raw, exact_raw = score_rows(rows, simplify_chinese=False)
    return {
        "rows": len(rows),
        "cer": cer,
        "exact_match": exact,
        "cer_raw": cer_raw,
        "exact_match_raw": exact_raw,
        "simplified_metrics": simplify_chinese,
        "keywords": keyword_stats(rows, simplify_chinese=simplify_chinese),
    }


def load_model(args):
    processor_path = args.adapter or args.model_name
    processor = WhisperProcessor.from_pretrained(processor_path, language=args.language, task=args.task)
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name)
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.generation_config.language = args.language
    model.generation_config.task = args.task
    model.generation_config.forced_decoder_ids = None
    model.generation_config.suppress_tokens = []
    model.to(args.device)
    model.eval()
    return processor, model


def transcribe_rows(rows, processor, model, args):
    output = []
    forced_decoder_ids = processor.get_decoder_prompt_ids(language=args.language, task=args.task)
    for row in tqdm(rows, desc=Path(args.manifest).name):
        audio = load_audio(Path(row["audio"]))
        inputs = processor.feature_extractor(audio, sampling_rate=TARGET_SR, return_tensors="pt")
        input_features = inputs.input_features.to(args.device)
        with torch.no_grad():
            generated_ids = model.generate(
                input_features=input_features,
                forced_decoder_ids=forced_decoder_ids,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
            )
        prediction = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        result = dict(row)
        result["prediction"] = prediction
        result["prediction_simplified"] = SIMPLIFIER.convert(prediction)
        output.append(result)
    return output


def write_predictions(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_summary(path: Path, summary):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerow(["rows", summary["rows"]])
        writer.writerow(["cer", f"{summary['cer']:.6f}"])
        writer.writerow(["exact_match", f"{summary['exact_match']:.6f}"])
        writer.writerow(["cer_raw", f"{summary['cer_raw']:.6f}"])
        writer.writerow(["exact_match_raw", f"{summary['exact_match_raw']:.6f}"])
        writer.writerow(["simplified_metrics", summary["simplified_metrics"]])
        for keyword, stat in summary["keywords"].items():
            writer.writerow([f"keyword_recall/{keyword}", f"{stat['recall']:.6f} ({stat['hit']}/{stat['total']})"])


def main():
    parser = argparse.ArgumentParser(description="Evaluate Whisper base or LoRA adapter on a manifest.")
    parser.add_argument("--model-name", default="openai/whisper-small")
    parser.add_argument("--adapter", type=Path, help="Path to LoRA adapter directory.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("eval_outputs"))
    parser.add_argument("--name", default="eval")
    parser.add_argument("--language", default="chinese")
    parser.add_argument("--task", default="transcribe")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--no-simplify-chinese", action="store_true", help="Disable Traditional-to-Simplified normalization for metrics.")
    args = parser.parse_args()

    rows = read_jsonl(args.manifest)
    processor, model = load_model(args)
    results = transcribe_rows(rows, processor, model, args)
    summary = summarize(results, simplify_chinese=not args.no_simplify_chinese)

    prediction_path = args.output_dir / f"{args.name}_predictions.jsonl"
    summary_path = args.output_dir / f"{args.name}_summary.csv"
    write_predictions(prediction_path, results)
    write_summary(summary_path, summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote predictions to {prediction_path}")
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
