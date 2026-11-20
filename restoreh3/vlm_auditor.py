"""Qwen3-VL-32B Blind Weather Sensing & Closed-Loop Epistemic OCR Auditor for RestoreH3.

Provides token-level confidence (`min_prob`, `mean_prob`) and Shannon entropy (`mean_ent`)
for Stage I blind weather routing and Stage III bidirectional scene-text auditing.
Supports single-GPU, multi-GPU (CUDA), and dual-NPU (Huawei Ascend 910B3) deployments.
"""
from __future__ import annotations

import gc
from pathlib import Path
from typing import Sequence, Union

import numpy as np
from PIL import Image, ImageOps
import torch


class Qwen3VLAuditor:
    """Frozen Qwen3-VL-32B visual-language auditor bundled with MiniMax-H3-Image."""

    def __init__(self, model_dir: Union[str, Path], device: str = "cuda:0"):
        from diffusers import ComponentsManager, ModularPipeline

        self.model_dir = Path(model_dir).expanduser().resolve()
        self.dev = torch.device(device)
        self.manager = ComponentsManager()
        self.pipe = ModularPipeline.from_pretrained(
            str(self.model_dir),
            components_manager=self.manager,
            local_files_only=True,
        )
        self.pipe.load_components(
            names=["text_encoder", "processor", "tokenizer"],
            pretrained_model_name_or_path=str(self.model_dir),
            local_files_only=True,
            dtype={"text_encoder": torch.bfloat16},
            attn_implementation={"text_encoder": "sdpa"},
            low_cpu_mem_usage={"text_encoder": True},
        )
        vlm = self.pipe.text_encoder.eval().requires_grad_(False)
        for mod in vlm.modules():
            if hasattr(mod, "config") and getattr(mod, "config", None) is not None:
                mod.config._attn_implementation = "eager"

        proc = self.pipe.processor
        if hasattr(proc, "tokenizer") and proc.tokenizer is not None:
            proc.tokenizer.padding_side = "left"
        self.pipe.tokenizer.padding_side = "left"

        self.multi_dispatched = False
        accel_count = 0
        if self.dev.type == "npu" and hasattr(torch, "npu"):
            accel_count = int(torch.npu.device_count())
        elif self.dev.type == "cuda" and torch.cuda.is_available():
            accel_count = int(torch.cuda.device_count())

        if self.dev.type in ("cuda", "npu") and accel_count >= 2:
            from accelerate import dispatch_model

            n_layers = len(vlm.model.language_model.layers)
            split_idx = n_layers // 2
            dev_map = {
                "model.visual": 0,
                "model.language_model.embed_tokens": 0,
                "model.language_model.rotary_emb": 0,
                "model.language_model.norm": 1,
                "lm_head": 1,
            }
            for idx_l in range(n_layers):
                dev_map[f"model.language_model.layers.{idx_l}"] = 0 if idx_l < split_idx else 1
            vlm = dispatch_model(vlm, device_map=dev_map)
            self.dev = torch.device(f"{self.dev.type}:0")
            self.multi_dispatched = True
        elif self.dev.type in ("cuda", "npu"):
            from diffusers.hooks.group_offloading import apply_group_offloading

            apply_group_offloading(
                vlm,
                onload_device=self.dev,
                offload_device=torch.device("cpu"),
                offload_type="block_level",
                num_blocks_per_group=16,
                use_stream=False,
                non_blocking=False,
                block_modules=["model", "language_model", "visual", "embed_tokens"],
            )
            vlm.lm_head.to(device=self.dev, dtype=torch.bfloat16)

        self.vlm = vlm
        self.proc = proc
        self.eos_ids = set()
        for eid in (
            getattr(vlm.config, "eos_token_id", None),
            getattr(getattr(vlm.config, "text_config", None), "eos_token_id", None),
            self.pipe.tokenizer.eos_token_id,
        ):
            if isinstance(eid, int):
                self.eos_ids.add(eid)
            elif isinstance(eid, (list, tuple)):
                self.eos_ids.update(int(x) for x in eid)

    @staticmethod
    def _prep_image(item: Union[Image.Image, np.ndarray], upscale_2x: bool = False) -> Image.Image:
        im = item.convert("RGB") if isinstance(item, Image.Image) else Image.fromarray(item, mode="RGB")
        w, h = im.size
        if upscale_2x and max(w, h) <= 480:
            im = im.resize((w * 2, h * 2), Image.Resampling.BICUBIC)
            w, h = im.size
        if max(w, h) > 1024:
            sc = 1024.0 / float(max(w, h))
            im = im.resize((max(32, int(round(w * sc))), max(32, int(round(h * sc)))), Image.Resampling.BICUBIC)
        if min(im.size) < 28:
            im = ImageOps.pad(im, (max(28, im.size[0]), max(28, im.size[1])), color=(0, 0, 0))
        return im

    def _run_chunk(
        self,
        chunk: Sequence[Union[Image.Image, np.ndarray]],
        prompt_text: str,
        upscale_2x: bool = False,
        max_tokens: int = 24,
    ) -> list[dict]:
        pils = [self._prep_image(x, upscale_2x=upscale_2x) for x in chunk]
        texts = [
            self.proc.apply_chat_template(
                [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for _ in pils
        ]
        m_in = self.proc(text=texts, images=pils, padding=True, return_tensors="pt")
        m_in = {k: (v.to(self.dev) if torch.is_tensor(v) else v) for k, v in m_in.items()}
        bsz = int(m_in["input_ids"].shape[0])
        step_ids = m_in["input_ids"]
        step_mask = m_in["attention_mask"]
        pix = m_in.get("pixel_values")
        thw = m_in.get("image_grid_thw")
        mm_ids = m_in.get("mm_token_type_ids")

        self.vlm.model.rope_deltas = None
        past_kv = None
        finished = [False] * bsz
        gen_rows = [[] for _ in range(bsz)]
        prob_rows = [[] for _ in range(bsz)]
        ent_rows = [[] for _ in range(bsz)]
        lm_dev = (
            torch.device(f"{self.dev.type}:1")
            if self.multi_dispatched
            else next(self.vlm.lm_head.parameters()).device
        )

        for step_i in range(max_tokens):
            call_kw = {
                "input_ids": step_ids,
                "attention_mask": step_mask,
                "past_key_values": past_kv,
                "use_cache": True,
                "return_dict": True,
            }
            if step_i == 0:
                call_kw["cache_position"] = torch.arange(step_ids.shape[1], device=self.dev, dtype=torch.long)
                if pix is not None:
                    call_kw["pixel_values"] = pix
                if thw is not None:
                    call_kw["image_grid_thw"] = thw
                if mm_ids is not None:
                    call_kw["mm_token_type_ids"] = mm_ids
            else:
                call_kw["cache_position"] = torch.tensor([step_mask.shape[1] - 1], device=self.dev, dtype=torch.long)
                text_pos = step_mask.long().sum(dim=-1, keepdim=True) - 1
                if getattr(self.vlm.model, "rope_deltas", None) is not None:
                    r_delta = self.vlm.model.rope_deltas.to(device=self.dev, dtype=torch.long)
                    mrope_pos = (text_pos + r_delta).unsqueeze(0).expand(3, bsz, 1)
                else:
                    mrope_pos = text_pos.unsqueeze(0).expand(3, bsz, 1)
                call_kw["position_ids"] = torch.cat([text_pos.unsqueeze(0), mrope_pos], dim=0)

            mout = self.vlm.model(**call_kw)
            past_kv = mout.past_key_values
            last_h = mout.last_hidden_state[:, -1:, :]
            logits = self.vlm.lm_head(last_h.to(device=lm_dev, dtype=torch.bfloat16))[:, -1, :].float()
            p_dist = torch.softmax(logits, dim=-1)
            ent = -(p_dist * torch.log(p_dist.clamp_min(1e-12))).sum(dim=-1)
            max_p, next_ids = p_dist.max(dim=-1)

            for bi in range(bsz):
                if not finished[bi]:
                    nid = int(next_ids[bi].item())
                    if nid in self.eos_ids:
                        finished[bi] = True
                    else:
                        gen_rows[bi].append(nid)
                        prob_rows[bi].append(float(max_p[bi].item()))
                        ent_rows[bi].append(float(ent[bi].item()))
            if all(finished):
                break
            step_ids = next_ids.unsqueeze(1).to(device=self.dev, dtype=m_in["input_ids"].dtype)
            step_mask = torch.cat([step_mask, torch.ones_like(step_ids, dtype=step_mask.dtype)], dim=1)

        out = []
        for bi in range(bsz):
            txt = self.pipe.tokenizer.decode(gen_rows[bi], skip_special_tokens=True).strip()
            mp = sum(prob_rows[bi]) / len(prob_rows[bi]) if prob_rows[bi] else 0.0
            min_p = min(prob_rows[bi]) if prob_rows[bi] else 0.0
            me = sum(ent_rows[bi]) / len(ent_rows[bi]) if ent_rows[bi] else 9.9
            out.append({
                "text": txt,
                "mean_prob": round(mp, 4),
                "min_prob": round(min_p, 4),
                "mean_ent": round(me, 4),
            })
        return out

    @torch.inference_mode()
    def infer_batch(
        self,
        items: Sequence[Union[Image.Image, np.ndarray]],
        prompt_text: str,
        upscale_2x: bool = False,
        max_tokens: int = 24,
        batch_sz: int = 8,
    ) -> list[dict]:
        if not items:
            return []
        out_all: list[dict] = []
        for st in range(0, len(items), batch_sz):
            chunk = items[st : st + batch_sz]
            try:
                out_all.extend(self._run_chunk(chunk, prompt_text, upscale_2x=upscale_2x, max_tokens=max_tokens))
            except Exception:
                gc.collect()
                if self.dev.type == "npu" and hasattr(torch, "npu"):
                    torch.npu.empty_cache()
                elif self.dev.type == "cuda":
                    torch.cuda.empty_cache()
                for c in chunk:
                    out_all.extend(self._run_chunk([c], prompt_text, upscale_2x=upscale_2x, max_tokens=max_tokens))
        return out_all

    def close(self) -> None:
        del self.vlm, self.proc, self.pipe, self.manager
        gc.collect()
        if self.dev.type == "npu" and hasattr(torch, "npu"):
            for di in range(torch.npu.device_count()):
                with torch.npu.device(di):
                    torch.npu.empty_cache()
        elif self.dev.type == "cuda":
            torch.cuda.empty_cache()
