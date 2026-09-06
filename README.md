# TransNormal-2

### Geometry-Grounded Rectified Flow with Edge-Aware Decoding for Precise Normal Estimation

[Mingwei Li](https://longxiang-ai.github.io/)<sup>1,2</sup>, [Yi Yang](https://scholar.google.com/citations?user=RMSuNFwAAAAJ)<sup>1</sup> (Fellow, IEEE), [Hehe Fan](https://hehefan.github.io/)<sup>1,†</sup> (Senior Member, IEEE)

<sup>1</sup> College of Artificial Intelligence, Zhejiang University  
<sup>2</sup> Zhongguancun Academy  
<sup>†</sup> Corresponding author: [Hehe Fan](mailto:hehefan@zju.edu.cn)

**[Project Page & Interactive Results](https://longxiang-ai.github.io/TransNormal-2/)**

TransNormal-2 estimates surface normals from a single RGB image with single-step deterministic rectified-flow inference. Geometry-aware pixel-space supervision and an RGB-guided **Geometric Refinement Module (GRM)** address VAE reconstruction degradation, with strong results on both general scenes and transparent objects.

## Highlights

- **One-step prediction:** a FLUX.2-based rectified-flow framework for monocular surface normal estimation.
- **Geometry-aware supervision:** inverse rendering self-consistency, von Mises-Fisher angular loss, and wavelet edge-aware regularization complement the latent objective.
- **Precise decoded normals:** a lightweight GRM corrects boundary-localized decoding errors using RGB guidance.
- **Seven evaluation benchmarks:** matches or exceeds MoGe-2 on all eight reported general-scene metrics with 1.4% as many task-specific normal annotations. Mean angular error improves by 4.2° on ClearGrasp and 3.1° on ClearPose over the strongest prior baselines.

## Release status

This initial release contains the project overview and interactive visual results.

| Resource | Status |
|---|---|
| Project page and visual comparisons | Available |
| arXiv paper | Forthcoming |
| Inference code | Planned |
| Model weights | Planned |
| Training code | Planned |

Code and model weights will be released progressively here. Release dates will be announced when the corresponding resources are ready.

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

## Related work

This work substantially extends [TransNormal](https://github.com/longxiang-ai/TransNormal). We acknowledge [FLUX.2](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B), [Diffusers](https://github.com/huggingface/diffusers), [Lotus](https://github.com/EnVision-Research/Lotus), and [Lotus-2](https://arxiv.org/abs/2512.01030).

## Contact

For questions, open an issue or contact [Hehe Fan](mailto:hehefan@zju.edu.cn).
