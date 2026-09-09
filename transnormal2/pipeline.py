"""TransNormal-2 inference pipeline.

Single-step rectified-flow normal estimation on the FLUX.2 [klein] backbone:

    RGB image -> VAE encode -> core-predictor LoRA (1 transformer step)
              -> LCM (latent smoothing) -> VAE decode
              -> GRM (bounded geometric refinement) -> unit normal map

The released model predicts the normal latent in a single deterministic
transformer evaluation (no diffusion sampling loop, no random noise), so the
predictions use no sampling noise. Floating-point results can vary across
hardware and library versions.
"""

import json
import os
from typing import List, Optional, Union

import torch
import torch.nn.functional as F

from diffusers import Flux2KleinPipeline
from diffusers.models import Flux2Transformer2DModel

from .grm import GRM
from .lcm import LocalContinuityModule
from .utils import resize_image_first, tensor_to_output

# LoRA target modules of the FLUX.2 [klein] transformer (must match training).
FLUX2_LORA_TARGET_MODULES = "|".join([
    # Double-stream: image attention
    r".*\.attn\.to_k$",
    r".*\.attn\.to_q$",
    r".*\.attn\.to_v$",
    r".*\.attn\.to_out\.0$",
    # Double-stream: text attention
    r".*\.attn\.add_k_proj$",
    r".*\.attn\.add_q_proj$",
    r".*\.attn\.add_v_proj$",
    r".*\.attn\.to_add_out$",
    # Double-stream: image FF (SwiGLU)
    r".*\.ff\.linear_in$",
    r".*\.ff\.linear_out$",
    # Double-stream: text FF (SwiGLU)
    r".*\.ff_context\.linear_in$",
    r".*\.ff_context\.linear_out$",
    # Single-stream: fused QKV+FF input projection
    r".*\.attn\.to_qkv_mlp_proj$",
    # Single-stream: output projection
    r"single_transformer_blocks\.\d+\.attn\.to_out$",
    # Context embedder
    r"context_embedder$",
])

_WEIGHT_FILES = {
    "lora": ("lora_core_predictor.safetensors",),
    "lcm": ("lcm_normal.safetensors",),
    "grm": ("grm.safetensors",),
}


def _load_state_dict(path: str) -> dict:
    from safetensors.torch import load_file
    return load_file(path)


def _find_weight(weights_dir: str, kind: str) -> str:
    for name in _WEIGHT_FILES[kind]:
        p = os.path.join(weights_dir, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"Could not find {kind} weights in {weights_dir} "
        f"(looked for {', '.join(_WEIGHT_FILES[kind])})"
    )


class TransNormal2Pipeline(Flux2KleinPipeline):
    """FLUX.2 [klein] pipeline specialized for one-step normal estimation."""

    # Set by from_pretrained_transnormal2().
    local_continuity_module: Optional[LocalContinuityModule] = None
    grm: Optional[GRM] = None

    # ── latent helpers (FLUX.2 BatchNorm-normalized patchified latents) ──

    def _encode_vae_to_raw(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode pixel values to raw VAE latents (before patchify/BN)."""
        return self.vae.encode(pixel_values.to(dtype=self.vae.dtype)).latent_dist.mode()

    def _raw_to_transformer_input(self, raw_latents: torch.Tensor) -> torch.Tensor:
        """Raw VAE latents -> patchified + BN-normalized transformer input."""
        patchified = self._patchify_latents(raw_latents)
        bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(patchified.device, patchified.dtype)
        bn_std = torch.sqrt(
            self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps
        ).to(patchified.device, patchified.dtype)
        return (patchified - bn_mean) / bn_std

    def _transformer_output_to_raw(self, latents_norm: torch.Tensor) -> torch.Tensor:
        """Transformer output (patchified + BN-normalized) -> raw VAE latents."""
        bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(latents_norm.device, latents_norm.dtype)
        bn_std = torch.sqrt(
            self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps
        ).to(latents_norm.device, latents_norm.dtype)
        denorm = latents_norm * bn_std + bn_mean
        return self._unpatchify_latents(denorm)

    def _decode_raw_latents(self, raw_latents: torch.Tensor) -> torch.Tensor:
        """Decode raw VAE latents to pixel space ([-1, 1])."""
        return self.vae.decode(raw_latents.to(dtype=self.vae.dtype), return_dict=False)[0]

    # ── loading ──

    @classmethod
    def from_pretrained_transnormal2(
        cls,
        base_model: str = "black-forest-labs/FLUX.2-klein-base-9B",
        weights_dir: str = "weights/transnormal2",
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Optional[Union[str, torch.device]] = None,
    ) -> "TransNormal2Pipeline":
        """One-call loader: base model + core-predictor LoRA + LCM + GRM.

        Args:
            base_model: HF id or local path of ``FLUX.2-klein-base-9B``.
            weights_dir: Local directory (or HF snapshot) containing
                ``lora_core_predictor.safetensors``, ``lcm_normal.safetensors``,
                ``grm.safetensors`` and ``config.json``.
            torch_dtype: bf16 strongly recommended (fp16 can produce NaNs).
            device: Optional device to move the pipeline to (e.g. "cuda").
        """
        from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict

        with open(os.path.join(weights_dir, "config.json")) as f:
            config = json.load(f)

        transformer = Flux2Transformer2DModel.from_pretrained(
            base_model, subfolder="transformer", torch_dtype=torch_dtype
        )
        transformer.requires_grad_(False)
        if device is not None:
            transformer.to(device=device, dtype=torch_dtype)
        else:
            transformer.to(dtype=torch_dtype)

        lora_cfg = config["lora"]
        adapter_name = lora_cfg.get("adapter_name", "core_predictor")
        transformer.add_adapter(
            LoraConfig(
                r=int(lora_cfg["rank"]),
                lora_alpha=float(lora_cfg["alpha"]),
                init_lora_weights="gaussian",
                target_modules=FLUX2_LORA_TARGET_MODULES,
            ),
            adapter_name=adapter_name,
        )
        lora_state = _load_state_dict(_find_weight(weights_dir, "lora"))
        expected_state = get_peft_model_state_dict(transformer, adapter_name=adapter_name)
        if set(lora_state) != set(expected_state):
            raise ValueError("LoRA checkpoint keys do not match the configured adapter.")
        for key, value in lora_state.items():
            if value.shape != expected_state[key].shape:
                raise ValueError(f"LoRA checkpoint shape mismatch for {key}.")
        set_peft_model_state_dict(transformer, lora_state, adapter_name=adapter_name)
        del lora_state, expected_state
        transformer.eval()
        transformer.set_adapter(adapter_name)

        pipe = cls.from_pretrained(
            base_model,
            transformer=transformer,
            torch_dtype=torch_dtype,
        )

        lcm = LocalContinuityModule(transformer.config.in_channels // 4)
        lcm.load_state_dict(_load_state_dict(_find_weight(weights_dir, "lcm")))
        lcm.requires_grad_(False)
        lcm.eval()

        grm_config = config.get("grm")
        if grm_config is None:
            raise KeyError("config.json must contain a 'grm' configuration")
        grm = GRM(**grm_config)
        grm.load_state_dict(_load_state_dict(_find_weight(weights_dir, "grm")), strict=True)
        grm.requires_grad_(False)
        grm.eval()

        pipe.local_continuity_module = lcm
        pipe.grm = grm
        if device is not None:
            pipe.to(device)
            lcm.to(device=device, dtype=torch_dtype)
            grm.to(device=device, dtype=torch_dtype)
        else:
            lcm.to(dtype=torch_dtype)
            grm.to(dtype=torch_dtype)
        return pipe

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        # Keep the auxiliary heads on the same device/dtype as the pipeline.
        for module in (self.local_continuity_module, self.grm):
            if module is not None:
                module.to(device=self.device, dtype=self.dtype)
        return result

    # ── inference ──

    @torch.no_grad()
    def __call__(  # type: ignore[override]
        self,
        image: torch.Tensor,
        domain_is_transparent: Union[bool, torch.Tensor] = False,
        process_res: Optional[int] = None,
        output_type: str = "pt",
        apply_grm: bool = True,
    ) -> Union[torch.Tensor, "np.ndarray", List]:
        """Predict a surface normal map from an RGB image.

        Args:
            image: RGB tensor (B, 3, H, W) normalized to [-1, 1].
            domain_is_transparent: True for transparent-dominant scenes
                (glass/liquids): the GRM then anchors on the raw prediction
                instead of the RGB guided filter. Default False (opaque).
            process_res: Optional max processing edge; None keeps the input
                resolution (snapped to a multiple of 16 internally).
            output_type: "pt" (B, 3, H, W) in [0, 1], "np" (B, H, W, 3) in
                [0, 1], or "pil".
            apply_grm: Disable to inspect the raw (pre-GRM) prediction.

        Returns:
            Normal map(s) encoded as ``(n + 1) / 2`` in [0, 1], resized back
            to the input resolution.
        """
        if image.dim() != 4 or image.shape[1] != 3:
            raise ValueError(f"Expected image of shape (B, 3, H, W), got {tuple(image.shape)}")

        input_size = image.shape[2:]
        rgb_in = resize_image_first(image, process_res)

        device = self._execution_device
        rgb_in = rgb_in.to(device=device, dtype=self.dtype)
        batch_size = rgb_in.shape[0]

        # 1. Empty-prompt conditioning (the released model is prompt-free).
        prompt_embeds, text_ids = self.encode_prompt(prompt="", device=device)
        # The Qwen3 text encoder may produce NaNs for the empty prompt in bf16.
        if torch.isnan(prompt_embeds).any():
            prompt_embeds = torch.nan_to_num(prompt_embeds, nan=0.0)
        if prompt_embeds.shape[0] != batch_size:
            prompt_embeds = prompt_embeds.expand(batch_size, -1, -1)
            text_ids = text_ids.expand(batch_size, -1, -1)

        # 2. Encode RGB to transformer-ready latents.
        raw_rgb_latents = self._encode_vae_to_raw(rgb_in)
        rgb_latents_norm = self._raw_to_transformer_input(raw_rgb_latents)
        latent_ids = self._prepare_latent_ids(rgb_latents_norm).to(device)
        packed_rgb_latents = self._pack_latents(rgb_latents_norm)

        # 3. Klein base has guidance_embeds=False; keep the guard for safety.
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], 1.0, device=device, dtype=torch.float32)
            guidance = guidance.expand(batch_size)
        else:
            guidance = None

        # 4. Single-step core predictor (timestep fixed at 1/1000).
        timestep = torch.tensor(1, device=device, dtype=self.dtype).expand(batch_size)
        latents_out = self.transformer(
            hidden_states=packed_rgb_latents,
            timestep=timestep / 1000,
            guidance=guidance,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_ids,
            joint_attention_kwargs={},
            return_dict=False,
        )[0]
        latents_out = latents_out[:, :packed_rgb_latents.size(1)]

        # 5. Unpack -> denormalize -> LCM -> decode.
        latents_patched = self._unpack_latents_with_ids(latents_out, latent_ids)
        latents_raw = self._transformer_output_to_raw(latents_patched)
        latents_raw = self.local_continuity_module(latents_raw)
        latents_raw = latents_raw.to(dtype=self.dtype)
        normal = self._decode_raw_latents(latents_raw)

        # 6. GRM bounded geometric refinement in pixel space.
        if apply_grm and self.grm is not None:
            rgb_resized = F.interpolate(
                rgb_in, size=normal.shape[2:], mode="bilinear", align_corners=False
            )
            normal = self.grm(normal, rgb_resized, domain_is_transparent=domain_is_transparent)

        # 7. [-1, 1] -> [0, 1], resize back to the input resolution.
        normal = self.image_processor.postprocess(normal, output_type="pt")
        normal = F.interpolate(normal, size=input_size, mode="bilinear", align_corners=False)

        self.maybe_free_model_hooks()
        return tensor_to_output(normal, output_type)
