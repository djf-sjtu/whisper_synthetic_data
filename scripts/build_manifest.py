import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


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


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            minimal = {
                "audio": row["audio"],
                "text": row["text"],
                "category": row.get("category", "domain"),
                "source_id": row.get("source_id"),
                "speaker": row.get("speaker"),
                "voice": row.get("voice"),
                "rate": row.get("rate"),
                "snr_db": row.get("snr_db"),
                "augmentation": row.get("augmentation", "clean"),
            }
            handle.write(json.dumps(minimal, ensure_ascii=False) + "\n")


def split_source_ids(source_ids, valid_ratio, test_ratio, seed):
    ids = sorted(source_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    n = len(ids)
    if n < 3:
        return set(ids), set(), set()
    n_test = max(1, round(n * test_ratio)) if n >= 10 else max(1, n // 5)
    n_valid = max(1, round(n * valid_ratio)) if n >= 10 else max(1, n // 5)
    test_ids = set(ids[:n_test])
    valid_ids = set(ids[n_test : n_test + n_valid])
    train_ids = set(ids[n_test + n_valid :])
    return train_ids, valid_ids, test_ids


def split_source_ids_by_category(by_source, valid_ratio, test_ratio, seed):
    by_category = defaultdict(list)
    for source_id, group in by_source.items():
        by_category[group[0].get("category", "domain")].append(source_id)

    train_ids = set()
    valid_ids = set()
    test_ids = set()
    for offset, source_ids in enumerate(by_category.values()):
        category_train, category_valid, category_test = split_source_ids(
            source_ids, valid_ratio, test_ratio, seed + offset
        )
        train_ids.update(category_train)
        valid_ids.update(category_valid)
        test_ids.update(category_test)
    return train_ids, valid_ids, test_ids


def source_text(by_source, source_id):
    return by_source[source_id][0]["text"]


def source_category(by_source, source_id):
    return by_source[source_id][0].get("category", "domain")


def ensure_keyword_coverage(by_source, train_ids, valid_ids, test_ids, keywords, seed):
    rng = random.Random(seed)
    split_sets = {"valid": valid_ids, "test": test_ids}

    for keyword in keywords:
        keyword_ids = [
            source_id
            for source_id in sorted(by_source)
            if keyword in source_text(by_source, source_id)
        ]
        if len(keyword_ids) < 3:
            continue

        for split_name, target_ids in split_sets.items():
            if any(source_id in target_ids for source_id in keyword_ids):
                continue

            candidates = [source_id for source_id in keyword_ids if source_id in train_ids]
            if not candidates:
                continue
            chosen = rng.choice(candidates)
            category = source_category(by_source, chosen)
            swap_candidates = [
                source_id
                for source_id in target_ids
                if source_category(by_source, source_id) == category
                and all(other_keyword not in source_text(by_source, source_id) for other_keyword in keywords)
            ]
            if swap_candidates:
                swapped = rng.choice(swap_candidates)
                target_ids.remove(swapped)
                train_ids.add(swapped)
            train_ids.remove(chosen)
            target_ids.add(chosen)


def main():
    parser = argparse.ArgumentParser(description="Build train/valid/test manifests without source command leakage.")
    parser.add_argument("--tts", type=Path, default=Path("data/manifests/tts_metadata.jsonl"))
    parser.add_argument("--augmented", type=Path, default=Path("data/manifests/augmented_metadata.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/manifests"))
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--no-clean", action="store_true", help="Do not include clean TTS rows.")
    args = parser.parse_args()

    rows = []
    if not args.no_clean:
        clean_rows = read_jsonl(args.tts)
        for row in clean_rows:
            row = dict(row)
            row.setdefault("augmentation", "clean")
            rows.append(row)
    rows.extend(read_jsonl(args.augmented))

    if not rows:
        raise SystemExit("No metadata found. Run generate_tts.py and augment_noise.py first.")

    by_source = defaultdict(list)
    for row in rows:
        by_source[row["source_id"]].append(row)

    train_ids, valid_ids, test_ids = split_source_ids_by_category(
        by_source, args.valid_ratio, args.test_ratio, args.seed
    )
    ensure_keyword_coverage(
        by_source,
        train_ids,
        valid_ids,
        test_ids,
        keywords=KEYWORDS,
        seed=args.seed + 1000,
    )

    splits = {"train": [], "valid": [], "test_domain": []}
    for source_id, group in by_source.items():
        if source_id in test_ids:
            splits["test_domain"].extend(group)
        elif source_id in valid_ids:
            splits["valid"].extend(group)
        elif source_id in train_ids:
            splits["train"].extend(group)

    rng = random.Random(args.seed)
    for name, split_rows in splits.items():
        rng.shuffle(split_rows)
        write_jsonl(args.out_dir / f"{name}.jsonl", split_rows)
        print(f"{name}: {len(split_rows)} rows")

    clean_test = [row for row in splits["test_domain"] if row.get("augmentation", "clean") == "clean"]
    noisy_test = [row for row in splits["test_domain"] if row.get("augmentation") == "snr_noise"]
    keyword_test = [
        row
        for row in splits["test_domain"]
        if any(keyword in row["text"] for keyword in KEYWORDS)
    ]
    short_test = [row for row in splits["test_domain"] if row.get("category") == "short"]
    write_jsonl(args.out_dir / "test_domain_clean.jsonl", clean_test)
    write_jsonl(args.out_dir / "test_domain_noisy.jsonl", noisy_test)
    write_jsonl(args.out_dir / "test_keywords.jsonl", keyword_test)
    write_jsonl(args.out_dir / "test_short.jsonl", short_test)
    print(f"test_domain_clean: {len(clean_test)} rows")
    print(f"test_domain_noisy: {len(noisy_test)} rows")
    print(f"test_keywords: {len(keyword_test)} rows")
    print(f"test_short: {len(short_test)} rows")
    print(f"source_id split: train={len(train_ids)}, valid={len(valid_ids)}, test={len(test_ids)}")


if __name__ == "__main__":
    main()
