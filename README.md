# TransNormal-2: Geometry-Grounded Rectified Flow with Edge-Aware Decoding for Precise Normal Estimation

[![Project Page](https://img.shields.io/badge/Project-Page-7057d9)](https://longxiang-ai.github.io/TransNormal-2/)
[![Interactive Comparisons](https://img.shields.io/badge/Explore-Comparisons-3988de)](https://longxiang-ai.github.io/TransNormal-2/#explore)
![arXiv forthcoming](https://img.shields.io/badge/arXiv-forthcoming-b31b1b)
![Code release planned](https://img.shields.io/badge/Code-release_planned-778192)

Official implementation of **TransNormal-2: Geometry-Grounded Rectified Flow with Edge-Aware Decoding for Precise Normal Estimation**.

[Mingwei Li<sup>1,2</sup>](https://github.com/longxiang-ai), [Yi Yang<sup>1</sup>](https://scholar.google.com/citations?user=RMSuNFwAAAAJ), [Hehe Fan<sup>1</sup>](https://hehefan.github.io/)

*<sup>1</sup>College of Artificial Intelligence, Zhejiang University · <sup>2</sup>Zhongguancun Academy*

**[Project Page](https://longxiang-ai.github.io/TransNormal-2/) · [Visual Comparisons](https://longxiang-ai.github.io/TransNormal-2/#explore) · [Results](#quantitative-results) · [Citation](#citation)**

## TL;DR

- **One RGB image, one prediction step.** A FLUX.2-based rectified-flow model estimates surface normals for general scenes and transparent objects.
- **Geometry-grounded prediction and decoding.** Pixel-space geometry objectives, latent correction, and an RGB-guided Geometric Refinement Module (GRM) address VAE reconstruction degradation.
- **Strong performance with 122K training samples.** Matches or exceeds MoGe-2 on all eight reported general-scene metrics with 1.4% as many task-specific normal annotations.
- **Clear gains on transparent objects.** Mean angular error improves by 4.2° on ClearGrasp and 3.1° on ClearPose over the strongest prior baselines.

## News

- **[2026-09-06]** Project page and interactive comparisons are online. Explore RGB images, multiple baselines, our predictions, and available ground truth.

## Release Roadmap

- [x] Project overview and visual results.
- [x] Interactive comparisons with baseline methods.
- [ ] arXiv preprint.
- [ ] Inference code.
- [ ] Model weights.
- [ ] Training code.

This initial release contains the project documentation and visual results. Code and model weights will be released progressively; installation and inference instructions will accompany the code release.

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

The arXiv identifier will be added when the preprint is available. In the meantime:

```bibtex
@misc{li2026transnormal2,
  title  = {TransNormal-2: Geometry-Grounded Rectified Flow with Edge-Aware Decoding for Precise Normal Estimation},
  author = {Li, Mingwei and Yang, Yi and Fan, Hehe},
  year   = {2026},
  url    = {https://longxiang-ai.github.io/TransNormal-2/}
}
```

## Acknowledgements

This work substantially extends [TransNormal](https://github.com/longxiang-ai/TransNormal). We acknowledge [FLUX.2](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B), [Diffusers](https://github.com/huggingface/diffusers), [Lotus](https://github.com/EnVision-Research/Lotus), and [Lotus-2](https://arxiv.org/abs/2512.01030).

## Contact

For questions about this project, contact **[Mingwei Li (@longxiang-ai)](https://github.com/longxiang-ai)** or [open a GitHub issue](https://github.com/longxiang-ai/TransNormal-2/issues).
