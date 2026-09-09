# TransNormal-2: Geometry-Grounded Rectified Flow with Edge-Aware Decoding for Precise Normal Estimation

[![Hugging Face](https://img.shields.io/badge/Hugging_Face-Model_Weights-FFD21E)](https://huggingface.co/Longxiang-ai/TransNormal-2)
[![Project Page](https://img.shields.io/badge/Project-Page-7057d9)](https://longxiang-ai.github.io/TransNormal-2/)
[![Interactive Comparisons](https://img.shields.io/badge/Explore-Comparisons-3988de)](https://longxiang-ai.github.io/TransNormal-2/#explore)
[![arXiv](https://img.shields.io/badge/arXiv-2609.06665-b31b1b)](https://arxiv.org/abs/2609.06665)
[![Inference](https://img.shields.io/badge/Code-Inference_Available-2d9d78)](#inference)

Official implementation of **TransNormal-2: Geometry-Grounded Rectified Flow with Edge-Aware Decoding for Precise Normal Estimation**.

[Mingwei Li<sup>1,2</sup>](https://github.com/longxiang-ai), [Yi Yang<sup>1</sup>](https://scholar.google.com/citations?user=RMSuNFwAAAAJ), [Hehe Fan<sup>1</sup>](https://hehefan.github.io/)

*<sup>1</sup>College of Artificial Intelligence, Zhejiang University · <sup>2</sup>Zhongguancun Academy*

**[GitHub](https://github.com/longxiang-ai/TransNormal-2) · [Hugging Face](https://huggingface.co/Longxiang-ai/TransNormal-2) · [arXiv](https://arxiv.org/abs/2609.06665) · [Project Page](https://longxiang-ai.github.io/TransNormal-2/) · [Visual Comparisons](https://longxiang-ai.github.io/TransNormal-2/#explore) · [Results](#quantitative-results) · [Citation](#citation)**

## TL;DR

- **One RGB image, one prediction step.** A FLUX.2-based rectified-flow model estimates surface normals for general scenes and transparent objects.
- **Geometry-grounded prediction and decoding.** Pixel-space geometry objectives, latent correction, and an RGB-guided Geometric Refinement Module (GRM) address VAE reconstruction degradation.
- **Strong performance with 122K training samples.** Matches or exceeds MoGe-2 on all eight reported general-scene metrics with 1.4% as many task-specific normal annotations.
- **Clear gains on transparent objects.** Mean angular error improves by 4.2° on ClearGrasp and 3.1° on ClearPose over the strongest prior baselines.

## News

- **[2026-09-09]** Inference code is available, including single-image prediction, folder processing, and CPU offload.

- **[2026-09-09]** [Model weights](https://huggingface.co/Longxiang-ai/TransNormal-2) are available on Hugging Face, with configuration, download instructions, and license information.

- **[2026-09-09]** The [arXiv preprint](https://arxiv.org/abs/2609.06665) is available.

- **[2026-09-06]** Project page and interactive comparisons are online. Explore RGB images, multiple baselines, our predictions, and available ground truth.

## Release Roadmap

- [x] Project overview and visual results.
- [x] Interactive comparisons with baseline methods.
- [x] [arXiv preprint](https://arxiv.org/abs/2609.06665).
- [x] [Inference code](#inference).
- [x] [Model weights](https://huggingface.co/Longxiang-ai/TransNormal-2).
- [ ] Training code.

The project documentation, visual results, [arXiv preprint](https://arxiv.org/abs/2609.06665), and [model weights](https://huggingface.co/Longxiang-ai/TransNormal-2) are available. Inference code is available below; training code will follow in a later release.

## Model Weights

Download the BF16 Safetensors weights and loading configuration from **[Hugging Face](https://huggingface.co/Longxiang-ai/TransNormal-2#download-and-use)**. The model card provides download instructions and the applicable licenses. Use these weights with the [inference code](#inference) in this repository.

## Installation

Use Python 3.10 and a CUDA GPU with BF16 support. The base model is the
**undistilled FLUX.2 [klein] base 9B** variant. Obtain access at
[Black Forest Labs on Hugging Face](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B)
and follow its license terms before downloading it.

```bash
git clone https://github.com/longxiang-ai/TransNormal-2.git
cd TransNormal-2
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If the base-model download requires authentication, run `hf auth login` with an
account that has access. TransNormal-2 task weights are public and download
without authentication. The complete base model requires substantially more
disk space and GPU memory than the task-specific weights alone.

## Inference

Pass an image or a folder to `--input`. Folders are searched recursively.
Replace the input paths below with your own images.

```bash
python inference.py --input path/to/image.jpg --output_dir outputs/normal --save_npy
python inference.py --input path/to/transparent_image.jpg --domain transparent --output_dir outputs/glass --save_npy
python inference.py --input path/to/images --output_dir outputs/batch --save_npy
```

The default `--weights` is `Longxiang-ai/TransNormal-2`. Both `--weights` and
`--base_model` also accept local download directories. `--domain opaque` uses an
RGB-guided anchor; `--domain transparent` uses the coarse normal prediction as
the anchor. Select the domain for the image; this flag does not detect
transparency automatically. GRM is enabled by default; use `--no_grm` only to
inspect the coarse prediction without refinement.

By default, inference uses BF16 at the input resolution, resized internally to
multiples of 16 and restored to the original dimensions. To limit processing
resolution, add `--process_res 768`. Add `--cpu_offload` to move the base-model
components between CPU and GPU; this requires adequate host RAM and increases
latency. BF16 is recommended; FP32 is available with `--dtype fp32`.

The selected domain applies to the whole folder. Run ordinary and transparent
scenes separately when they need different domain settings. Subdirectories are
preserved in the output, so identical filenames in different folders remain
separate.

### Python API

```python
import torch
from huggingface_hub import snapshot_download
from transnormal2 import TransNormal2Pipeline, load_image, save_normal_map

weights = snapshot_download(
    "Longxiang-ai/TransNormal-2",
    allow_patterns=["*.safetensors", "config.json"],
    token=False,
)
pipe = TransNormal2Pipeline.from_pretrained_transnormal2(
    weights_dir=weights,
    torch_dtype=torch.bfloat16,
    device="cuda",
)
normal = pipe(load_image("path/to/transparent_image.jpg"), domain_is_transparent=True)
save_normal_map(normal, "normal.png", save_npy="normal.npy")
```

Outputs are PNG visualizations and, with `--save_npy`, float32 NumPy arrays of
shape `(H, W, 3)` encoded in `[0, 1]`. Convert the encoding with `n = 2 * prediction - 1`
and normalize the vectors when consuming them as unit normals after resizing.
The PNG is a normal visualization, not a depth map.

The method uses a single deterministic prediction step without sampling noise.
Floating-point outputs can vary with hardware, dtype, and library versions.
See `requirements.txt` for the tested package versions. Model weights retain
the terms on their [Hugging Face model card](https://huggingface.co/Longxiang-ai/TransNormal-2).

## Qualitative Results

![Comparison of RGB inputs, MoGe-2, Lotus-2, TransNormal, and TransNormal-2 across transparent glass, mechanical details, and a TN-Syn scene](https://raw.githubusercontent.com/longxiang-ai/TransNormal-2/gh-pages/assets/readme-comparisons.png)

*Selected qualitative examples from the manuscript and supplementary material. Each row shows the same input across methods; quantitative results below summarize benchmark performance.*

**[Open the interactive viewer →](https://longxiang-ai.github.io/TransNormal-2/#explore)**

Drag the divider to compare **MoGe-2, Lotus-2, TransNormal, E2E-FT**, and other available baselines with TransNormal-2. An RGB preview provides context for every scene; **ClearGrasp and TN-Syn** also include ground-truth comparisons.

## Method Overview

![TransNormal-2 architecture: encoding, single-step rectified-flow prediction, decoding and geometric refinement](https://raw.githubusercontent.com/longxiang-ai/TransNormal-2/gh-pages/assets/pipeline.png)

TransNormal-2 combines single-step rectified flow with geometry-aware supervision and geometric refinement in a unified framework. Inverse-rendering self-consistency, von Mises-Fisher angular loss, and wavelet edge-aware regularization complement latent MSE. Latent correction and the lightweight **GRM** work with the predictor to improve decoded normals and recover boundary detail.

## Quantitative Results

The following tables reproduce selected methods from the manuscript. **Mean angular error (MAE) is in degrees; lower is better.** The paper contains the complete comparisons and evaluation protocols.

### Transparent objects

| Method | ClearGrasp (Synthetic) ↓ | TN-Syn ↓ | ClearPose (Real-world) ↓ |
|:--|--:|--:|--:|
| MoGe-2 | 26.6 | 6.2 | 36.2 |
| FE2E | 16.9 | 21.2 | 22.2 |
| Lotus-2 | 15.5 | 5.5 | 23.4 |
| TransNormal | 16.1 | 3.9 | 25.5 |
| **TransNormal-2 (Ours)** | **11.3** | **3.6** | **19.1** |

### General scenes

| Method | NYUv2 ↓ | ScanNet ↓ | iBims ↓ | Sintel ↓ |
|:--|--:|--:|--:|--:|
| DSINE | 16.4 | 16.2 | 17.1 | 34.9 |
| FE2E | 16.3 | 13.8 | 15.1 | 31.2 |
| Lotus-2 | 16.9 | 14.2 | 15.4 | 30.3 |
| MoGe-2 | **14.7** | 12.8 | **14.7** | 29.3 |
| TransNormal | 16.6 | 15.3 | 16.4 | 35.1 |
| **TransNormal-2 (Ours)** | **14.7** | **12.7** | **14.7** | **29.2** |

The general-scene table shows the four MAE metrics. The paper additionally reports accuracy within 11.25° on each dataset.

## Citation

If you find this work useful, please cite the [arXiv preprint](https://arxiv.org/abs/2609.06665):

```bibtex
@misc{li2026transnormal2,
  title = {TransNormal-2: Geometry-Grounded Rectified Flow with Edge-Aware Decoding for Precise Normal Estimation},
  author = {Mingwei Li and Yi Yang and Hehe Fan},
  year = {2026},
  eprint = {2609.06665},
  archivePrefix = {arXiv},
  primaryClass = {cs.CV},
  url = {https://arxiv.org/abs/2609.06665}
}
```

## Acknowledgements

This work substantially extends [TransNormal](https://github.com/longxiang-ai/TransNormal). We acknowledge [FLUX.2](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B), [Diffusers](https://github.com/huggingface/diffusers), [Lotus](https://github.com/EnVision-Research/Lotus), and [Lotus-2](https://arxiv.org/abs/2512.01030).

## Contact

For questions about this project, contact **[Mingwei Li (@longxiang-ai)](https://github.com/longxiang-ai)** or [open a GitHub issue](https://github.com/longxiang-ai/TransNormal-2/issues).
