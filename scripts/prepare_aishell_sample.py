import argparse
import json
import random
from pathlib import Path


def find_transcript(root: Path):
    candidates = [
        root / "transcript" / "aishell_transcript_v0.8.txt",
        root / "data_aishell" / "transcript" / "aishell_transcript_v0.8.txt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = list(root.rglob("aishell_transcript_v0.8.txt"))
    if matches:
        return matches[0]
    raise SystemExit(f"Could not find aishell_transcript_v0.8.txt under {root}")


def load_transcripts(path: Path):
    transcripts = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split(maxsplit=1)
            if len(parts) != 2:
                continue
            utt_id, text = parts
            transcripts[utt_id] = text.replace(" ", "")
    return transcripts


def find_wavs(root: Path, split: str):
    split_dirs = [
        root / "wav" / split,
        root / "data_aishell" / "wav" / split,
    ]
    for split_dir in split_dirs:
        if split_dir.exists():
            return sorted(split_dir.rglob("*.wav"))
    wav_root = root / "wav"
    if wav_root.exists():
        return sorted(wav_root.rglob("*.wav"))
    return sorted(root.rglob("*.wav"))


def main():
    parser = argparse.ArgumentParser(description="Sample AISHELL-1 wavs into a jsonl manifest.")
    parser.add_argument("--aishell-root", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--out", type=Path, default=Path("data/manifests/aishell_train_1000.jsonl"))
    args = parser.parse_args()

    transcript_path = find_transcript(args.aishell_root)
    transcripts = load_transcripts(transcript_path)
    wavs = find_wavs(args.aishell_root, args.split)

    rows = []
    for wav in wavs:
        utt_id = wav.stem
        text = transcripts.get(utt_id)
        if not text:
            continue
        rows.append(
            {
                "audio": str(wav.resolve().as_posix()),
                "text": text,
                "category": "aishell",
                "source_id": f"aishell_{utt_id}",
                "speaker": wav.parent.name,
                "voice": "real",
                "rate": None,
                "snr_db": None,
                "augmentation": "real_clean",
            }
        )

    if len(rows) < args.limit:
        raise SystemExit(f"Only found {len(rows)} matched AISHELL rows, need {args.limit}")

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    sampled = rows[: args.limit]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for row in sampled:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(sampled)} AISHELL rows to {args.out}")


if __name__ == "__main__":
    main()
