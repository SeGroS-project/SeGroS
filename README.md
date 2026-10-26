# SeGroS — Harmon Implementation

Official implementation of **Enhancing Alignment for Unified Multimodal Models via Semantically-Grounded Supervision** (ECCV 2026).

[Jiyeong Kim, Yerim So, Hyesong Choi, Uiwon Hwang, and Dongbo Min — arXiv:2603.19807](https://arxiv.org/abs/2603.19807)

SeGroS uses text–image affinity to select semantic visual hints and reconstruction targets during fine-tuning. This repository contains the **Harmon 0.5B and 1.5B** implementation, built on [Harmon](https://github.com/wusize/Harmon). Implementations for other backbones studied in the paper are outside this repository's scope.

## Method

1. Rank text tokens using the sum of normalized intra-modal and inter-modal affinity scores, with equal weights.
2. Aggregate the selected tokens' text-to-image affinities into a visual grounding map.
3. Min–max normalize the map and always add uniform noise.
4. Select high-scoring image tokens as visual hints, preserving their spatial order.
5. Mask high-scoring target tokens within the existing MAR mask budget, retaining lower-scoring tokens as visible context.

Visual hints and reconstruction targets belong to separate sequences and may overlap. At inference, use the base Harmon generation path without constructing the grounding map.

## Repository layout

```text
configs/examples/segros.py       # Harmon 1.5B training
configs/examples/segros_05.py    # Harmon 0.5B training
configs/models/                 # Base model configurations
configs/datasets/               # Data sources, tokenization, and task mixture
src/models/harmon_dev.py        # SeGroS training and affinity computation
src/models/harmon.py            # Base Harmon model
src/models/mar/                 # MAR components and masking
scripts/train.py               # Training entry point
scripts/text2image.py           # Text-to-image inference
scripts/geneval.py             # GenEval image generation
scripts/dpgbench.py             # DPG-Bench image generation
scripts/parallel_geneval.py     # Multi-GPU generation launcher
```

Other example configurations are retained as legacy experiments. Use `segros` and `segros_05` for the settings documented here.

## Installation

Use a Linux environment with an NVIDIA GPU and a compatible CUDA build environment for FlashAttention and DeepSpeed. Run all commands from the repository root.

```bash
conda create -n segros python=3.10 -y
conda activate segros

# CUDA 12.1 example; install PyTorch before the remaining dependencies.
python -m pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
export PYTHONPATH=.
```

See the [PyTorch version matrix](https://pytorch.org/get-started/previous-versions/#v230) for supported alternative builds. The dependency list retains the original PyTorch 2.3.0 pin; a fresh GPU installation and end-to-end training run have not yet been validated for this cleaned snapshot.

## Base checkpoints

Obtain the base weights from [wusize/harmon](https://huggingface.co/wusize/harmon) and place them as follows:

```text
checkpoints/
├── kl16.ckpt
├── harmon_0.5b.pth
└── harmon_1.5b.pth
```

The model configurations also load Qwen model/tokenizer assets from Hugging Face. Check `configs/models/` and `configs/datasets/qwen2_5_1_5b/processors.py` when using local assets. Fine-tuned SeGroS checkpoints are not bundled in this repository.

## Training data

### Image-to-text

Prepare LLaVA-format annotations and their referenced image folders. Set `data_root`, `data_path`, and `image_folder` in `configs/datasets/qwen2_5_1_5b/image2text.py` to your local paths. This configuration still contains machine-specific defaults and must be edited before training.

Example layout:

```text
data/LLaVA-Instruct-150K/
├── llava_v1_5_mix665k.json
└── tuning_data/
    ├── coco/
    ├── gqa/
    ├── ocr_vqa/
    ├── textvqa/
    └── vg/
```

### Text-to-image

`configs/datasets/qwen2_5_1_5b/text2image.py` uses `MidJourneyDataset`, whose default dataset identifier is `brivangl/midjourney-v6-llava`. It is loaded through Hugging Face Datasets.

Add `cache_dir=None` (the standard Hugging Face cache) or a writable local cache path to the `dataset` dictionary in **both** `text2image.py` and `reconstruction_mid.py`, overriding their dataset classes' machine-specific cache defaults. The combined dataset constructs the reconstruction dataset even when its sampling weight is zero.

### Task mixture

Both SeGroS configurations import `image2text_recon_mid.py`:

```python
group_keys = ['recon', 'image2text', 'text2image']
repeat = [0, 2, 2]
```

This samples I2T and T2I tasks with equal weights and does not sample the standalone reconstruction task. The shared data configuration currently uses `batch_size=48` and `num_workers=8`. Adjust these for available hardware and record changes when comparing results.

## SeGroS configuration

Both training configs explicitly define:

```python
visual_hint_ratio=0.3,
visual_hint_selection='highest',
target_mask_selection='highest',
text_token_filtering=True,
text_keep_ratio=0.4,
text_min_tokens=3,
noise_scale=1.0,
```

| Parameter | Meaning |
| --- | --- |
| `visual_hint_ratio` | Fraction of image tokens used as visual hints |
| `visual_hint_selection` | `highest`, `lowest`, or `random` hint selection |
| `target_mask_selection` | `highest`, `lowest`, or `random` reconstruction-target selection |
| `text_token_filtering` | Enable discriminative text-token selection |
| `text_keep_ratio` | Fraction of valid text tokens selected |
| `text_min_tokens` | Minimum number selected, capped by the available valid tokens |
| `noise_scale` | Positive finite upper bound of uniform noise added to the normalized map |

Noise is always applied: `noisy_scores = normalized_scores + U(0, noise_scale)`. Both selection paths use this perturbed map; the unperturbed scores are also returned for inspection. There is no `add_noise` switch. The affinity combination remains fixed at `norm_self + norm_it`.

**Configuration provenance:** this snapshot retains `noise_scale=1.0` from the working code. The paper's method description specifies `0.5`; reconcile this value with the original experiment logs before claiming exact reproduction of the reported results.

The masking ratio distribution remains defined by the MAR implementation and base model configuration (`mask_ratio_min=0.7`). The selection strategy changes which tokens are masked, not the sampled mask count.

## Training

Set `CUDA_VISIBLE_DEVICES`, `GPUS_PER_NODE`, and, if needed, distributed settings in `train.sh`. The launcher currently defaults to one GPU and DeepSpeed ZeRO-2.

```bash
# Harmon 1.5B
bash train.sh segros

# Harmon 0.5B
bash train.sh segros_05
```

| Setting | 1.5B (`segros`) | 0.5B (`segros_05`) |
| --- | --- | --- |
| Training iterations | 5,000 | 3,000 |
| Learning rate | 1e-5 | 1e-5 |
| Weight decay | 0.01 | 0.02 |
| Gradient accumulation | 2 | 2 |
| Checkpoint interval | 1,000 | 1,000 |
| Seed | 42 | 42 |

These values describe the shipped configs. Checkpoints and logs are written under `work_dirs/segros/` or `work_dirs/segros_05/`. Preserve the resolved configuration, environment versions, GPU count, and checkpoint alongside each reported result.

## Text-to-image inference

After training a 1.5B checkpoint:

```bash
python scripts/text2image.py \
  --config configs/models/qwen2_5_1_5b_kl16_mar_h.py \
  --checkpoint work_dirs/segros/iter_5000.pth \
  --prompt "A white dog on the left and a black cat on the right." \
  --seed 0 --image_size 512 --output output.jpg
```

For 0.5B, use `configs/models/qwen2_5_0_5b_kl16_mar_b.py` and the matching checkpoint, such as `work_dirs/segros_05/iter_3000.pth`.

## Benchmark generation and scoring

The scripts below **generate images**; they do not compute benchmark scores. Obtain the official prompt assets and install each benchmark's evaluator separately. Pass explicit prompt paths instead of relying on the scripts' sibling-directory defaults.

### GenEval

Obtain the JSONL prompt metadata from [GenEval](https://github.com/djghosh13/geneval), then run:

```bash
python scripts/parallel_geneval.py \
  --mode geneval --gpus 0,1 --batch_size 4 \
  --config configs/models/qwen2_5_1_5b_kl16_mar_h.py \
  --checkpoint work_dirs/segros/iter_5000.pth \
  --validation_prompts_file /path/to/evaluation_metadata.jsonl \
  --outdir results/geneval_segros --seed 0
```

Run the official GenEval evaluation and summary tools against the generated samples according to its README.

### DPG-Bench

Prepare a JSON object mapping prompt filenames to prompt strings, for example `{"00001.txt": "A red car beside a tree."}`, using the official [DPG-Bench prompt assets](https://github.com/TencentQQGYLab/ELLA/tree/main/dpg_bench).

```bash
python scripts/parallel_geneval.py \
  --mode dpgbench --gpus 0,1 \
  --config configs/models/qwen2_5_1_5b_kl16_mar_h.py \
  --checkpoint work_dirs/segros/iter_5000.pth \
  --prompts_file /path/to/prompts.json \
  --outdir results/dpgbench_segros --seed 0
```

The DPG generation launcher uses four samples per prompt and saves separate JPEG files. Arrange generated images in the format required by the official DPG-Bench evaluator before scoring; generation alone is not a completed evaluation. A scoring adapter and CompBench evaluation pipeline are not included in this snapshot.

## Citation

```bibtex
@article{kim2026segros,
  title={Enhancing Alignment for Unified Multimodal Models via Semantically-Grounded Supervision},
  author={Kim, Jiyeong and So, Yerim and Choi, Hyesong and Hwang, Uiwon and Min, Dongbo},
  journal={arXiv preprint arXiv:2603.19807},
  year={2026}
}
```

## Acknowledgements and license

This implementation builds on [Harmon](https://github.com/wusize/Harmon) and its MAR components, MMEngine, and XTuner. SeGroS-specific training logic is implemented in `src/models/harmon_dev.py` and the SeGroS experiment configs.

The existing [LICENSE](LICENSE) and source-level copyright/license notices are retained. Consult those notices and the respective terms for upstream code, datasets, and model weights.
