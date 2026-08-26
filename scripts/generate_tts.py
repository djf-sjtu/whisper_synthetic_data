import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path


EDGE_VOICES = [
    "zh-CN-XiaoxiaoNeural",
    "zh-CN-XiaoyiNeural",
    "zh-CN-YunjianNeural",
    "zh-CN-YunxiNeural",
    "zh-CN-YunyangNeural",
    "zh-CN-XiaochenNeural",
]

DEFAULT_EDGE_PROFILES = [
    ("xiaoxiao_slow", "zh-CN-XiaoxiaoNeural", "-10%"),
    ("yunxi_normal", "zh-CN-YunxiNeural", "+0%"),
    ("yunyang_fast", "zh-CN-YunyangNeural", "+10%"),
]


def safe_name(value: str) -> str:
    value = re.sub(r"[^0-9A-Za-z_\-.]+", "_", value.strip())
    return value.strip("_") or "item"


def parse_commands(path: Path):
    rows = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "\t" in line:
            category, text = line.split("\t", 1)
        else:
            category, text = "domain", line
        category = category.strip() or "domain"
        text = text.strip()
        if not text:
            continue
        rows.append({"line_no": line_no, "category": category, "text": text})
    return rows


def parse_profiles(profile_values):
    profiles = []
    for raw in profile_values or []:
        parts = [part.strip() for part in raw.split(",")]
        if len(parts) != 3:
            raise SystemExit(
                "Each --profiles value must be profile_id,voice,rate. "
                "Example: xiaoxiao_slow,zh-CN-XiaoxiaoNeural,-10%"
            )
        profile_id, voice, rate = parts
        profiles.append((profile_id, voice, rate))
    return profiles


def sapi_rate(percent_rate: str) -> int:
    value = int(percent_rate.replace("%", ""))
    return max(-10, min(10, round(value / 8)))


def run_sapi(text: str, out_path: Path, voice: str, rate: str):
    ps_script = r"""
param(
  [string]$Text,
  [string]$OutPath,
  [string]$VoiceName,
  [int]$Rate
)
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$ErrorActionPreference = "Stop"
if ($VoiceName -and $VoiceName -ne "auto") {
  $synth.SelectVoice($VoiceName)
}
$synth.Rate = $Rate
$format = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo 16000, Sixteen, Mono
$synth.SetOutputToWaveFile($OutPath, $format)
$synth.Speak($Text)
$synth.Dispose()
"""
    with tempfile.NamedTemporaryFile("w", suffix=".ps1", delete=False, encoding="utf-8") as handle:
        handle.write(ps_script)
        script_path = handle.name
    try:
        cmd = [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            script_path,
            "-Text",
            text,
            "-OutPath",
            str(out_path),
            "-VoiceName",
            voice,
            "-Rate",
            str(sapi_rate(rate)),
        ]
        subprocess.run(cmd, check=True)
    finally:
        try:
            os.unlink(script_path)
        except OSError:
            pass


def run_sapi_batch(jobs, quiet: bool):
    if not jobs:
        return

    payload = [
        {
            "text": job["text"],
            "out_path": str(Path(job["out_path"]).resolve()),
            "voice": job["voice"],
            "rate": sapi_rate(job["rate"]),
        }
        for job in jobs
    ]
    ps_script = r"""
param(
  [string]$JobsPath,
  [switch]$Quiet
)
$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Speech
$payload = Get-Content -Raw -Encoding UTF8 $JobsPath | ConvertFrom-Json
$jobs = @($payload.jobs)
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$format = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo 16000, Sixteen, Mono
$total = $jobs.Count
$done = 0
foreach ($job in $jobs) {
  $done += 1
  if ((-not $Quiet) -or $done -eq 1 -or $done % 25 -eq 0 -or $done -eq $total) {
    Write-Output ("[{0}/{1}] {2} -> {3}" -f $done, $total, $job.text, $job.out_path)
  }
  if ($job.voice -and $job.voice -ne "auto") {
    $synth.SelectVoice([string]$job.voice)
  }
  $synth.Rate = [int]$job.rate
  $synth.SetOutputToWaveFile([string]$job.out_path, $format)
  $synth.Speak([string]$job.text) | Out-Null
  $synth.SetOutputToNull()
}
$synth.Dispose()
"""
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as jobs_file:
        json.dump({"jobs": payload}, jobs_file, ensure_ascii=False)
        jobs_path = jobs_file.name
    with tempfile.NamedTemporaryFile("w", suffix=".ps1", delete=False, encoding="utf-8") as handle:
        handle.write(ps_script)
        script_path = handle.name
    try:
        cmd = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", script_path, "-JobsPath", jobs_path]
        if quiet:
            cmd.append("-Quiet")
        subprocess.run(cmd, check=True)
    finally:
        for temp_path in [jobs_path, script_path]:
            try:
                os.unlink(temp_path)
            except OSError:
                pass


def run_edge(text: str, out_path: Path, voice: str, rate: str):
    mp3_path = out_path.with_suffix(".edge.mp3")
    cmd = [
        sys.executable,
        "-m",
        "edge_tts",
        "--voice",
        voice,
        "--rate",
        rate,
        "--text",
        text,
        "--write-media",
        str(mp3_path),
    ]
    try:
        subprocess.run(cmd, check=True)
        convert_to_wav_16k(mp3_path, out_path)
    finally:
        if mp3_path.exists():
            mp3_path.unlink()


def convert_to_wav_16k(input_path: Path, out_path: Path):
    try:
        import numpy as np
        import soundfile as sf
        from scipy import signal
    except ImportError as exc:
        raise RuntimeError(
            "Edge TTS writes compressed audio first; install requirements-synthetic.txt "
            "so it can be converted to 16 kHz wav."
        ) from exc

    audio, sr = sf.read(input_path, always_2d=False)
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if sr != 16000:
        gcd = math.gcd(sr, 16000)
        audio = signal.resample_poly(audio, 16000 // gcd, sr // gcd).astype(np.float32)
    peak = float(np.max(np.abs(audio)) + 1e-8)
    if peak > 0.98:
        audio = audio * (0.98 / peak)
    sf.write(out_path, audio, 16000, subtype="PCM_16")


def generate(args):
    commands = parse_commands(args.commands)
    if args.limit:
        commands = commands[: args.limit]
    if not commands:
        raise SystemExit(f"No commands found in {args.commands}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.metadata.parent.mkdir(parents=True, exist_ok=True)

    profiles = parse_profiles(args.profiles)
    if not profiles and args.engine == "edge" and args.default_edge_profiles:
        profiles = DEFAULT_EDGE_PROFILES
    if not profiles:
        voices = args.voices or (EDGE_VOICES if args.engine == "edge" else ["auto"])
        rates = args.rates
        profiles = [(safe_name(voice), voice, rate) for voice in voices for rate in rates]
    rows = []
    sapi_jobs = []
    total = len(commands) * len(profiles)
    done = 0

    for index, item in enumerate(commands, start=1):
        source_id = f"cmd{index:04d}"
        for profile_id, voice, rate in profiles:
            done += 1
            filename = f"{source_id}__{safe_name(profile_id)}__rate{rate.replace('%', '')}.wav"
            out_path = args.out_dir / filename
            needs_generate = args.overwrite or not out_path.exists()
            if needs_generate:
                if args.engine == "edge":
                    if not args.quiet or done == 1 or done % 25 == 0 or done == total:
                        print(f"[{done}/{total}] {item['text']} -> {out_path}")
                    run_edge(item["text"], out_path, voice, rate)
                else:
                    sapi_jobs.append(
                        {
                            "text": item["text"],
                            "out_path": out_path,
                            "voice": voice,
                            "rate": rate,
                        }
                    )
            row = {
                "audio": str(out_path.as_posix()),
                "text": item["text"],
                "category": item["category"],
                "source_id": source_id,
                "speaker": profile_id,
                "voice": voice,
                "rate": rate,
                "engine": args.engine,
                "line_no": item["line_no"],
            }
            rows.append(row)

    if args.engine == "sapi":
        run_sapi_batch(sapi_jobs, args.quiet)

    with args.metadata.open("w", encoding="utf-8") as meta:
        for row in rows:
            meta.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} TTS rows to {args.metadata}")


def list_sapi_voices():
    ps_script = r"""
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
Write-Output ("Default: " + $synth.Voice.Name)
try {
  $voices = $synth.GetInstalledVoices()
  foreach ($voice in $voices) {
    Write-Output $voice.VoiceInfo.Name
  }
} catch {
  Write-Warning "Could not enumerate installed voices, but default synthesis may still work."
}
$synth.Dispose()
"""
    with tempfile.NamedTemporaryFile("w", suffix=".ps1", delete=False, encoding="utf-8") as handle:
        handle.write(ps_script)
        script_path = handle.name
    try:
        subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", script_path],
            check=False,
        )
    finally:
        try:
            os.unlink(script_path)
        except OSError:
            pass


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic command speech with Edge TTS or Windows SAPI.")
    parser.add_argument("--commands", type=Path, default=Path("commands.txt"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/tts"))
    parser.add_argument("--metadata", type=Path, default=Path("data/manifests/tts_metadata.jsonl"))
    parser.add_argument("--engine", choices=["edge", "sapi"], default="sapi")
    parser.add_argument("--voices", nargs="*", help="Voice names. Defaults to several zh-CN Edge voices, or SAPI auto.")
    parser.add_argument("--rates", nargs="*", default=["-15%", "+0%", "+15%"])
    parser.add_argument(
        "--profiles",
        nargs="*",
        help="Profile triples: profile_id,voice,rate. This avoids voices x rates expansion.",
    )
    parser.add_argument(
        "--default-edge-profiles",
        action="store_true",
        help="Use three built-in Edge profiles: slow female, normal male, fast male.",
    )
    parser.add_argument("--limit", type=int, help="Generate only the first N commands for a smoke test.")
    parser.add_argument("--quiet", action="store_true", help="Print only every 25th item and final summary.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--list-sapi-voices", action="store_true")
    args = parser.parse_args()

    if args.list_sapi_voices:
        list_sapi_voices()
        return
    generate(args)


if __name__ == "__main__":
    main()
