import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from peft import LoraConfig, get_peft_model
from scipy import signal
from torch.utils.data import Dataset
from transformers import (
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)


TARGET_SR = 16000
REPO_ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


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


def sample_rows(rows, limit, seed):
    if not limit or len(rows) <= limit:
        return rows
    rng = random.Random(seed)
    rows = list(rows)
    rng.shuffle(rows)
    return rows[:limit]


class ManifestSpeechDataset(Dataset):
    def __init__(self, rows, processor):
        self.rows = rows
        self.processor = processor

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        audio = load_audio(Path(row["audio"]))
        features = self.processor.feature_extractor(
            audio,
            sampling_rate=TARGET_SR,
            return_attention_mask=True,
        )
        labels = self.processor.tokenizer(row["text"]).input_ids
        return {
            "input_features": features.input_features[0],
            "attention_mask": features.attention_mask[0],
            "labels": labels,
        }


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features):
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def print_trainable_parameters(model):
    trainable = 0
    total = 0
    for _, parameter in model.named_parameters():
        total += parameter.numel()
        if parameter.requires_grad:
            trainable += parameter.numel()
    percent = 100 * trainable / total
    print(f"trainable params: {trainable:,} / {total:,} ({percent:.4f}%)")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Whisper small with LoRA on jsonl manifests.")
    parser.add_argument("--model-name", default="openai/whisper-small")
    parser.add_argument("--train-manifest", type=Path, default=Path("data/manifests/train.jsonl"))
    parser.add_argument("--valid-manifest", type=Path, default=Path("data/manifests/valid.jsonl"))
    parser.add_argument("--aishell-manifest", type=Path)
    parser.add_argument("--max-aishell-rows", type=int, default=250)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/whisper-small-lora"))
    parser.add_argument("--language", default="chinese")
    parser.add_argument("--task", default="transcribe")
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-train-epochs", type=float, default=10)
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    processor = WhisperProcessor.from_pretrained(args.model_name, language=args.language, task=args.task)
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name)
    model.config.use_cache = False
    model.generation_config.language = args.language
    model.generation_config.task = args.task
    model.generation_config.forced_decoder_ids = None
    model.generation_config.suppress_tokens = []

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    print_trainable_parameters(model)

    train_rows = read_jsonl(args.train_manifest)
    if args.aishell_manifest:
        aishell_rows = sample_rows(read_jsonl(args.aishell_manifest), args.max_aishell_rows, args.seed)
        train_rows = train_rows + aishell_rows
    random.shuffle(train_rows)
    valid_rows = read_jsonl(args.valid_manifest)

    print(f"train rows: {len(train_rows)}")
    print(f"valid rows: {len(valid_rows)}")

    train_dataset = ManifestSpeechDataset(train_rows, processor)
    valid_dataset = ManifestSpeechDataset(valid_rows, processor)
    collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.num_train_epochs,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        logging_steps=args.logging_steps,
        predict_with_generate=False,
        generation_max_length=64,
        fp16=args.fp16,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
        label_names=["labels"],
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=collator,
        tokenizer=processor.feature_extractor,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )
    trainer.train()
    trainer.save_model(str(args.output_dir))
    processor.save_pretrained(str(args.output_dir))
    print(f"Saved LoRA adapter and processor to {args.output_dir}")


if __name__ == "__main__":
    main()
