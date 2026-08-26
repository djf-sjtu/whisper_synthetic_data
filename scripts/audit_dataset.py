import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

import soundfile as sf


KEYWORDS = ["WAIC", "Sage Robot One", "Sage Dog", "飒智智能科技有限公司", "小飒", "飒智"]


def read_jsonl(path: Path):
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def summarize_manifest(path: Path):
    rows = read_jsonl(path)
    by_source = {}
    for row in rows:
        by_source.setdefault(row.get("source_id"), row)
    print(f"\n{path.name}: rows={len(rows)}, unique_commands={len(by_source)}")
    print("  category:", dict(Counter(row.get("category") for row in by_source.values())))
    print("  augmentation:", dict(Counter(row.get("augmentation", "clean") for row in rows)))
    print("  snr:", dict(Counter(row.get("snr_db") for row in rows)))
    for keyword in KEYWORDS:
        print(f"  {keyword} unique:", sum(keyword in row.get("text", "") for row in by_source.values()))
    return rows


def check_overlap(manifests):
    source_sets = {}
    for name, rows in manifests.items():
        source_sets[name] = {row.get("source_id") for row in rows}
    print("\nsource_id overlap:")
    names = list(source_sets)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            print(f"  {left} vs {right}: {len(source_sets[left] & source_sets[right])}")


def check_audio(rows):
    durations = []
    bad = []
    for row in rows:
        path = Path(row["audio"])
        try:
            info = sf.info(str(path))
            durations.append(info.duration)
            if info.samplerate != 16000 or info.channels != 1 or info.duration <= 0:
                bad.append((str(path), info.samplerate, info.channels, info.duration))
        except Exception as exc:
            bad.append((str(path), "read_error", str(exc), None))
    print("\naudio:")
    print(f"  files checked: {len(rows)}")
    print(f"  bad files: {len(bad)}")
    if durations:
        durations_sorted = sorted(durations)
        print(
            "  duration min/avg/p50/p95/max:",
            round(min(durations), 2),
            round(statistics.mean(durations), 2),
            round(statistics.median(durations), 2),
            round(durations_sorted[int(0.95 * (len(durations_sorted) - 1))], 2),
            round(max(durations), 2),
        )
    for item in bad[:10]:
        print("  bad:", item)


def main():
    parser = argparse.ArgumentParser(description="Audit synthetic Whisper dataset manifests and audio files.")
    parser.add_argument("--manifest-dir", type=Path, default=Path("data/manifests"))
    parser.add_argument("--check-audio", action="store_true")
    args = parser.parse_args()

    manifests = {}
    for name in [
        "train",
        "valid",
        "test_domain",
        "test_domain_clean",
        "test_domain_noisy",
        "test_keywords",
        "test_short",
    ]:
        path = args.manifest_dir / f"{name}.jsonl"
        rows = summarize_manifest(path)
        if name in {"train", "valid", "test_domain"}:
            manifests[name] = rows

    check_overlap(manifests)
    if args.check_audio:
        all_rows = []
        seen = set()
        for rows in manifests.values():
            for row in rows:
                audio = row["audio"]
                if audio not in seen:
                    seen.add(audio)
                    all_rows.append(row)
        check_audio(all_rows)


if __name__ == "__main__":
    main()
