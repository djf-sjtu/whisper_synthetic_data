import argparse
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal


TARGET_SR = 16000


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_audio(path: Path, target_sr: int = TARGET_SR):
    audio, sr = sf.read(path, always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if sr != target_sr:
        gcd = math.gcd(sr, target_sr)
        audio = signal.resample_poly(audio, target_sr // gcd, sr // gcd).astype(np.float32)
    return audio, target_sr


def write_audio(path: Path, audio: np.ndarray, sr: int = TARGET_SR):
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.clip(audio, -0.98, 0.98)
    sf.write(path, audio, sr, subtype="PCM_16")


def rms(audio: np.ndarray):
    return float(np.sqrt(np.mean(np.square(audio)) + 1e-12))


def repeat_to_length(audio: np.ndarray, length: int, rng: np.random.Generator):
    if len(audio) == 0:
        return np.zeros(length, dtype=np.float32)
    if len(audio) >= length:
        start = int(rng.integers(0, len(audio) - length + 1))
        return audio[start : start + length].astype(np.float32)
    repeats = int(np.ceil(length / len(audio)))
    tiled = np.tile(audio, repeats)
    return tiled[:length].astype(np.float32)


def synthetic_noise(kind: str, length: int, sr: int, rng: np.random.Generator):
    white = rng.normal(0.0, 1.0, length).astype(np.float32)
    if kind == "white":
        noise = white
    elif kind == "pink":
        b, a = signal.butter(1, 0.08, btype="low")
        noise = signal.lfilter(b, a, white).astype(np.float32)
    elif kind == "hum":
        t = np.arange(length, dtype=np.float32) / sr
        hum = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 100 * t)
        noise = hum.astype(np.float32) + 0.15 * white
    else:
        b1, a1 = signal.butter(4, [250 / (sr / 2), 3400 / (sr / 2)], btype="band")
        speech_band = signal.lfilter(b1, a1, white).astype(np.float32)
        envelope = signal.lfilter([1.0], [1.0, -0.995], rng.random(length).astype(np.float32) - 0.5)
        envelope = np.abs(envelope)
        envelope = envelope / (np.max(envelope) + 1e-6)
        low = signal.lfilter(*signal.butter(2, 600 / (sr / 2), btype="low"), white).astype(np.float32)
        noise = 0.75 * speech_band * (0.35 + envelope) + 0.25 * low
    noise = noise - np.mean(noise)
    return noise / (rms(noise) + 1e-8)


def add_reverb(audio: np.ndarray, sr: int, rng: np.random.Generator, amount: float):
    if amount <= 0:
        return audio
    decay_seconds = float(rng.uniform(0.12, 0.35))
    ir_len = max(1, int(sr * decay_seconds))
    t = np.arange(ir_len, dtype=np.float32) / sr
    ir = np.exp(-t / decay_seconds).astype(np.float32)
    ir[0] = 1.0
    ir += 0.02 * rng.normal(size=ir_len).astype(np.float32)
    ir = ir / (np.sum(np.abs(ir)) + 1e-8)
    wet = signal.fftconvolve(audio, ir, mode="full")[: len(audio)].astype(np.float32)
    return ((1.0 - amount) * audio + amount * wet).astype(np.float32)


def mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float):
    clean_rms = rms(clean)
    noise_rms = rms(noise)
    target_noise_rms = clean_rms / (10 ** (snr_db / 20.0))
    scaled_noise = noise * (target_noise_rms / (noise_rms + 1e-8))
    mixed = clean + scaled_noise
    peak = float(np.max(np.abs(mixed)) + 1e-8)
    if peak > 0.98:
        mixed = mixed * (0.98 / peak)
    return mixed.astype(np.float32)


def load_noise_pool(noise_dir: Path, target_sr: int):
    if not noise_dir.exists():
        return []
    pool = []
    for path in sorted(noise_dir.rglob("*")):
        if path.suffix.lower() not in {".wav", ".flac", ".ogg"}:
            continue
        try:
            audio, _ = load_audio(path, target_sr)
            if len(audio) >= target_sr // 2:
                pool.append((path, audio))
        except Exception as exc:
            print(f"Skipping noise file {path}: {exc}")
    return pool


def main():
    parser = argparse.ArgumentParser(description="Create SNR-based noisy speech from TTS metadata.")
    parser.add_argument("--input", type=Path, default=Path("data/manifests/tts_metadata.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/augmented"))
    parser.add_argument("--metadata", type=Path, default=Path("data/manifests/augmented_metadata.jsonl"))
    parser.add_argument("--noise-dir", type=Path, default=Path("data/noise"))
    parser.add_argument("--snrs", nargs="*", type=float, default=[10, 20])
    parser.add_argument("--synthetic-noise-kinds", nargs="*", default=["crowd"])
    parser.add_argument("--reverb-prob", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260823)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    rows = list(read_jsonl(args.input))
    if not rows:
        raise SystemExit(f"No TTS metadata rows found in {args.input}")

    noise_pool = load_noise_pool(args.noise_dir, TARGET_SR)
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    written = 0

    with args.metadata.open("w", encoding="utf-8") as meta:
        for row in rows:
            clean_path = Path(row["audio"])
            clean, sr = load_audio(clean_path, TARGET_SR)
            clean = add_reverb(clean, sr, rng, 0.25 if rng.random() < args.reverb_prob else 0.0)
            stem = clean_path.stem
            for snr in args.snrs:
                if noise_pool and rng.random() < 0.7:
                    noise_path, noise_audio = noise_pool[int(rng.integers(0, len(noise_pool)))]
                    noise = repeat_to_length(noise_audio, len(clean), rng)
                    noise_name = noise_path.stem
                else:
                    kind = args.synthetic_noise_kinds[int(rng.integers(0, len(args.synthetic_noise_kinds)))]
                    noise = synthetic_noise(kind, len(clean), sr, rng)
                    noise_name = f"synthetic_{kind}"
                mixed = mix_at_snr(clean, noise, snr)
                out_path = args.out_dir / f"{stem}__snr{int(snr)}__{noise_name}.wav"
                write_audio(out_path, mixed, sr)
                out_row = dict(row)
                out_row.update(
                    {
                        "audio": str(out_path.as_posix()),
                        "augmentation": "snr_noise",
                        "snr_db": snr,
                        "noise": noise_name,
                    }
                )
                meta.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                written += 1

    print(f"Wrote {written} augmented rows to {args.metadata}")
    if not noise_pool:
        print("No external noise files found; used generated venue-like noise only.")


if __name__ == "__main__":
    main()
