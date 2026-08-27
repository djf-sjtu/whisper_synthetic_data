# Whisper small synthetic domain data

这个目录用于模拟生成 Whisper small LoRA 微调前四步的数据：

1. 写领域指令文本
2. 用 TTS 批量合成语音
3. 叠加 10/20 dB 噪声和简单混响
4. 生成 `train/valid/test_domain` manifest
5. 抽样 1000 条 AISHELL-1 通用中文语音

## 服务器手动执行版

服务器上从空仓库开始，按这个顺序执行即可。先确认当前目录是仓库根目录：

```powershell
cd D:\your\path\whisper_synthetic_data
```

创建环境并安装依赖。PyTorch 的 CUDA 版本要按服务器实际情况改，下面以 CUDA 12.1 为例：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-synthetic.txt
pip install -r requirements-train.txt
```

验证 GPU：

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

本仓库已经包含一份由 Edge TTS 生成好的领域数据：

```text
data/tts_edge：360 条干净 wav
data/augmented_edge：720 条 10/20dB 噪声增强 wav
data/manifests：训练、验证、测试 jsonl
```

本仓库也已经包含 AISHELL-1 抽样数据：`data/aishell_sample/train` 1000 条、`data/aishell_sample/test` 100 条。服务器上正常不用重新生成 TTS，也不用重新准备 AISHELL，安装依赖后可以直接从 “微调前 baseline” 开始。

当前数据规模：

```text
领域 synthetic 全量：1080 条，约 0.69 小时
AISHELL train：1000 条，约 1.24 小时
AISHELL test：100 条，约 0.12 小时
仓库总音频：2180 条，约 2.05 小时
首轮实际训练：领域 train 792 条 + AISHELL 250 条，约 0.81 小时
```

如果以后改了 `commands.txt`，才需要重新生成领域数据。`commands.txt` 是人工策划的 120 条领域文本，正常不要运行旧生成脚本。Edge TTS 不需要登录，但会把 `commands.txt` 文本发送给 Microsoft 在线 TTS：

```powershell
python scripts/generate_tts.py --engine edge --default-edge-profiles --out-dir data\tts_edge --metadata data\manifests\tts_metadata.jsonl --quiet --overwrite
python scripts/augment_noise.py --input data\manifests\tts_metadata.jsonl --out-dir data\augmented_edge --metadata data\manifests\augmented_metadata.jsonl
python scripts/build_manifest.py
python scripts/audit_dataset.py --check-audio
```

如果 Edge TTS 失败，就先用本地 SAPI 跑通流程：

```powershell
python scripts/generate_tts.py --engine sapi --profiles sapi_slow,auto,-10% sapi_normal,auto,+0% sapi_fast,auto,+10% --quiet --overwrite
python scripts/augment_noise.py
python scripts/build_manifest.py
python scripts/audit_dataset.py --check-audio
```

领域数据生成后的合理数量大致是：

```text
data/tts_edge：360 条 wav
data/augmented_edge：720 条 wav
train：约 792 行
valid：约 117 行
test_domain：约 171 行
```

首轮训练只从 1000 条 AISHELL 训练池里抽 250 条混入，避免 AISHELL 压过领域指令。

先跑微调前 baseline：

```powershell
python scripts/evaluate_whisper.py --manifest data/manifests/test_domain_clean.jsonl --name base_domain_clean
python scripts/evaluate_whisper.py --manifest data/manifests/test_domain_noisy.jsonl --name base_domain_noisy
python scripts/evaluate_whisper.py --manifest data/manifests/test_keywords.jsonl --name base_keywords
python scripts/evaluate_whisper.py --manifest data/manifests/test_short.jsonl --name base_short
python scripts/evaluate_whisper.py --manifest data/manifests/test_general_aishell_100.jsonl --name base_general_aishell
```

开始 LoRA 微调。RTX 4090 24GB 推荐先用这版：

```powershell
python scripts/train_whisper_lora.py `
  --train-manifest data/manifests/train.jsonl `
  --valid-manifest data/manifests/valid.jsonl `
  --aishell-manifest data/manifests/aishell_train_1000.jsonl `
  --max-aishell-rows 250 `
  --output-dir outputs/whisper-small-lora `
  --per-device-train-batch-size 16 `
  --per-device-eval-batch-size 16 `
  --gradient-accumulation-steps 1 `
  --num-train-epochs 12 `
  --learning-rate 1e-4 `
  --warmup-steps 30 `
  --eval-steps 50 `
  --save-steps 50 `
  --bf16
```

如果 `--bf16` 报错，改成 `--fp16`。如果显存不足，把 `--per-device-train-batch-size` 改成 8，并加上 `--gradient-accumulation-steps 2`，等效 batch 仍然是 16。

8GB 显存用这版：

```powershell
python scripts/train_whisper_lora.py `
  --train-manifest data/manifests/train.jsonl `
  --valid-manifest data/manifests/valid.jsonl `
  --aishell-manifest data/manifests/aishell_train_1000.jsonl `
  --max-aishell-rows 250 `
  --output-dir outputs/whisper-small-lora `
  --per-device-train-batch-size 2 `
  --gradient-accumulation-steps 8 `
  --num-train-epochs 10 `
  --learning-rate 1e-4 `
  --fp16 `
  --gradient-checkpointing
```

微调后复跑同一批测试：

```powershell
python scripts/evaluate_whisper.py --adapter outputs/whisper-small-lora --manifest data/manifests/test_domain_clean.jsonl --name lora_domain_clean
python scripts/evaluate_whisper.py --adapter outputs/whisper-small-lora --manifest data/manifests/test_domain_noisy.jsonl --name lora_domain_noisy
python scripts/evaluate_whisper.py --adapter outputs/whisper-small-lora --manifest data/manifests/test_keywords.jsonl --name lora_keywords
python scripts/evaluate_whisper.py --adapter outputs/whisper-small-lora --manifest data/manifests/test_short.jsonl --name lora_short
python scripts/evaluate_whisper.py --adapter outputs/whisper-small-lora --manifest data/manifests/test_general_aishell_100.jsonl --name lora_general_aishell
```

对比 `eval_outputs/*_summary.csv`。如果 `lora_keywords` 明显变好，同时 `lora_general_aishell` 没明显变差，这轮就算有效。

## 目录

```text
commands.txt                 # 领域指令文本，每行 category<TAB>text
requirements-synthetic.txt   # 只用于合成数据的依赖
requirements-train.txt       # Whisper/LoRA 训练依赖
scripts/generate_command_catalog.py  # 旧版可选脚本；只有加 --force 才会覆盖人工清单
scripts/generate_tts.py      # TTS 合成
scripts/augment_noise.py     # SNR 噪声增强
scripts/build_manifest.py    # 生成训练/验证/测试 manifest
scripts/prepare_aishell_sample.py    # 从 AISHELL-1 抽样通用中文语音
scripts/train_whisper_lora.py        # LoRA 微调 Whisper small
scripts/evaluate_whisper.py          # 微调前后评估
scripts/merge_lora.py                # 可选：合并 LoRA adapter
scripts/clean_generated_data.ps1     # 清理旧生成数据和训练输出
data/noise/                  # 可选：放公开噪声 wav/flac/ogg
data/tts_edge/               # 已入库的 Edge TTS 干净音频
data/augmented_edge/         # 已入库的 10/20dB 噪声增强音频
data/aishell_sample/         # 已入库的 AISHELL-1 通用中文抽样音频
data/manifests/              # jsonl 元数据和切分结果
```

## 安装依赖

如果本地之前生成过旧数据，先清理：

```powershell
.\scripts\clean_generated_data.ps1
.\scripts\clean_generated_data.ps1 -Force
```

在本目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-synthetic.txt
```

服务器训练建议使用 Python 3.10 或 3.11。PyTorch 请按服务器 CUDA 版本从官网选择安装命令，例如：

```powershell
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

然后安装其余训练依赖：

```powershell
pip install -r requirements-train.txt
```

如果服务器不能访问 Edge TTS，可以改用 Windows 本地 SAPI：

```powershell
python scripts/generate_tts.py --engine sapi --list-sapi-voices
```

## 1. 编辑指令

默认方案是 120 条人工策划领域输入，质量优先：

```text
command: 55
qa: 45
fallback: 10
short: 10
```

直接修改 `commands.txt`。格式是：

```text
command	小飒小飒，帮我拿瓶可乐
qa	小飒小飒，介绍Sage Robot One
fallback	你听清楚了吗
short	向左转
```

## 2. 生成 TTS

推荐规模是 `120 条文本 × 3 个声音配置 = 360 条干净语音`。后面再叠加 10/20 dB 噪声，总领域数据是 `360 × 3 = 1080 条`。

默认使用 Windows 本地 SAPI，不会把指令文本发到外部服务，但通常只有一个系统声音。可以用三个语速配置先跑通：

```powershell
python scripts/generate_tts.py --engine sapi --limit 3
python scripts/generate_tts.py --engine sapi --profiles sapi_slow,auto,-10% sapi_normal,auto,+0% sapi_fast,auto,+10% --quiet --overwrite
```

如果你确认 `commands.txt` 可以发送给 Microsoft 在线 TTS 服务，可以使用 Edge TTS 三个人声配置，声音更自然：

```powershell
python scripts/generate_tts.py --engine edge --default-edge-profiles --out-dir data\tts_edge --metadata data\manifests\tts_metadata.jsonl --limit 3
python scripts/generate_tts.py --engine edge --default-edge-profiles --out-dir data\tts_edge --metadata data\manifests\tts_metadata.jsonl --quiet --overwrite
```

查看本机可用 SAPI 声音：

```powershell
python scripts/generate_tts.py --engine sapi --list-sapi-voices
```

不要同时使用 `--voices 三个声音` 和 `--rates 三个语速`，那会变成 `120 × 3 × 3 × 3 = 3240` 条领域数据，而不是 1080 条。

生成结果：

```text
data/tts_edge/*.wav
data/manifests/tts_metadata.jsonl
```

## 3. 叠加噪声

可以先不放任何噪声文件，脚本会生成类似会场的合成噪声：

```powershell
python scripts/augment_noise.py --input data\manifests\tts_metadata.jsonl --out-dir data\augmented_edge --metadata data\manifests\augmented_metadata.jsonl
```

如果你后来下载了 DEMAND/MUSAN 等公开噪声，把 wav/flac/ogg 放进 `data/noise/`，脚本会自动混用真实噪声：

```powershell
python scripts/augment_noise.py --input data\manifests\tts_metadata.jsonl --out-dir data\augmented_edge --metadata data\manifests\augmented_metadata.jsonl --noise-dir data/noise
```

默认生成 10/20 dB 两档：

```text
data/augmented_edge/*.wav
data/manifests/augmented_metadata.jsonl
```

## 4. 生成 manifest

```powershell
python scripts/build_manifest.py
```

输出：

```text
data/manifests/train.jsonl
data/manifests/valid.jsonl
data/manifests/test_domain.jsonl
data/manifests/test_domain_clean.jsonl
data/manifests/test_domain_noisy.jsonl
data/manifests/test_keywords.jsonl
data/manifests/test_short.jsonl
```

切分按类别分层后再按 `source_id` 做，保证同一句指令的干净版和所有噪声增强版不会同时出现在训练集和测试集里。默认 `valid=10%`、`test=15%`，所以 120 条文本大约会切成：

```text
train：约 88 条唯一领域输入
valid：约 13 条唯一领域输入
test_domain：约 19 条唯一领域输入
```

`test_domain_clean` 和 `test_domain_noisy` 用来分别比较安静和噪声场景，`test_keywords` 用来重点看 “飒智”“小飒”“WAIC”“Sage Robot One”“Sage Dog” 等专名，`test_short` 用来看短指令。

## 5. 审计数据集

```powershell
python scripts/audit_dataset.py --check-audio
```

## 6. AISHELL-1 抽样数据

仓库已包含抽样好的 AISHELL-1 通用中文语音：

```text
data/aishell_sample/train/*.wav
data/aishell_sample/test/*.wav
data/manifests/aishell_train_1000.jsonl
data/manifests/test_general_aishell_100.jsonl
```

其中 `aishell_train_1000.jsonl` 来自说话人 `S0002/S0003/S0004`，`test_general_aishell_100.jsonl` 来自独立说话人 `S0005`。训练时建议混合：

```text
领域 train：约 75%-80%
AISHELL：约 20%-25%
AISHELL train 1000：约 1.24 小时
AISHELL test 100：约 0.12 小时
```

## 7. 微调前 baseline

先跑原始 `openai/whisper-small`，保存一份 baseline：

```powershell
python scripts/evaluate_whisper.py --manifest data/manifests/test_domain_clean.jsonl --name base_domain_clean
python scripts/evaluate_whisper.py --manifest data/manifests/test_domain_noisy.jsonl --name base_domain_noisy
python scripts/evaluate_whisper.py --manifest data/manifests/test_keywords.jsonl --name base_keywords
python scripts/evaluate_whisper.py --manifest data/manifests/test_short.jsonl --name base_short
python scripts/evaluate_whisper.py --manifest data/manifests/test_general_aishell_100.jsonl --name base_general_aishell
```

## 8. LoRA 微调

RTX 4090 24GB 推荐先用这个配置：

```powershell
python scripts/train_whisper_lora.py `
  --train-manifest data/manifests/train.jsonl `
  --valid-manifest data/manifests/valid.jsonl `
  --aishell-manifest data/manifests/aishell_train_1000.jsonl `
  --max-aishell-rows 250 `
  --output-dir outputs/whisper-small-lora `
  --per-device-train-batch-size 16 `
  --per-device-eval-batch-size 16 `
  --gradient-accumulation-steps 1 `
  --num-train-epochs 12 `
  --learning-rate 1e-4 `
  --warmup-steps 30 `
  --eval-steps 50 `
  --save-steps 50 `
  --bf16
```

这组参数的含义：

```text
per-device-train-batch-size 16：4090 显存够，直接吃 16 条，训练更快。
gradient-accumulation-steps 1：不再累积梯度；有效 batch = 16。
num-train-epochs 12：数据少，最多跑 12 轮；脚本会按 valid loss 早停。
learning-rate 1e-4：LoRA 常用起点，适合小规模领域适配。
warmup-steps 30：前 30 步慢慢升学习率，减少一开始震荡。
eval/save-steps 50：约每 0.75 个 epoch 验证和保存一次，方便早停选最好 checkpoint。
bf16：4090 支持，通常比 fp16 更稳。
max-aishell-rows 250：只混入约 25% AISHELL，防止通用语音压过领域指令。
```

如果 `--bf16` 报错，改成 `--fp16`。如果显存不足，把 `--per-device-train-batch-size` 改成 8，并加上 `--gradient-accumulation-steps 2`，等效 batch 仍然是 16。

8GB 显存可以用这个配置：

```powershell
python scripts/train_whisper_lora.py `
  --train-manifest data/manifests/train.jsonl `
  --valid-manifest data/manifests/valid.jsonl `
  --aishell-manifest data/manifests/aishell_train_1000.jsonl `
  --max-aishell-rows 250 `
  --output-dir outputs/whisper-small-lora `
  --per-device-train-batch-size 2 `
  --gradient-accumulation-steps 8 `
  --num-train-epochs 10 `
  --learning-rate 1e-4 `
  --fp16 `
  --gradient-checkpointing
```

12GB 或 16GB 显存可以把 `--per-device-train-batch-size` 调到 4 或 8。

## 9. 微调后评估

```powershell
python scripts/evaluate_whisper.py --adapter outputs/whisper-small-lora --manifest data/manifests/test_domain_clean.jsonl --name lora_domain_clean
python scripts/evaluate_whisper.py --adapter outputs/whisper-small-lora --manifest data/manifests/test_domain_noisy.jsonl --name lora_domain_noisy
python scripts/evaluate_whisper.py --adapter outputs/whisper-small-lora --manifest data/manifests/test_keywords.jsonl --name lora_keywords
python scripts/evaluate_whisper.py --adapter outputs/whisper-small-lora --manifest data/manifests/test_short.jsonl --name lora_short
python scripts/evaluate_whisper.py --adapter outputs/whisper-small-lora --manifest data/manifests/test_general_aishell_100.jsonl --name lora_general_aishell
```

重点看：

```text
CER 是否下降
exact_match 是否上升
keyword_recall/WAIC、keyword_recall/Sage Robot One、keyword_recall/Sage Dog 是否上升
general_aishell 的 CER 是否明显变差
```

## 10. 合并 LoRA

如果部署端不想单独加载 adapter，可以合并：

```powershell
python scripts/merge_lora.py --adapter outputs/whisper-small-lora --out-dir outputs/whisper-small-lora-merged
```

## 建议

这套数据能先验证 LoRA 微调流程和专有名词适配，但真实上线前最好仍然录 50-100 条现场语音做最终验收。模拟数据负责快速起步，真实小样本负责兜底。
