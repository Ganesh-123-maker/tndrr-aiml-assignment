# GPU memory to fine-tune a 7B model — worked calculation

Model size: **7B parameters = 7 × 10⁹ params**.
All sizes use decimal GB (1 GB = 10⁹ bytes) and are rounded; they are
ballparks, not exact profiler readings — the point is the order of
magnitude and *where* the memory goes.

## Assumptions (stated explicitly)

| Item | Bytes / param | For 7B |
|---|---|---|
| Weights in bf16 | 2 | 14.0 GB |
| Weights in fp32 | 4 | 28.0 GB |
| Gradients, bf16 | 2 | 14.0 GB |
| Gradients, fp32 | 4 | 28.0 GB |
| AdamW state m (fp32) | 4 | 28.0 GB |
| AdamW state v (fp32) | 4 | 28.0 GB |
| AdamW m + v together | 8 | 56.0 GB |
| NF4-quantised weights (QLoRA) | 0.5 | 3.5 GB |

Further assumptions:
* Mixed-precision training (the standard setup): the model computes in
  bf16, but the optimiser keeps a **fp32 master copy** of the weights
  (4 B/param) and fp32 AdamW states m and v (4 + 4 B/param).
* Gradients are counted in bf16 (14 GB); counting them in fp32 would
  add another 14 GB to the full-FT total.
* Activations depend on batch size × sequence length × hidden size ×
  layers, and on whether gradient checkpointing is on. I use
  ~2–4 GB for a small batch (e.g. effective batch 4–8, seq 512) *without*
  checkpointing, and ~1 GB *with* checkpointing (QLoRA's usual setup).
  Activations are the least precise term here; they do not change the
  ranking of the three methods.
* LoRA trainable count is computed, not guessed: for a 7B-class model
  (≈32 layers, hidden size 4096), LoRA on q_proj + v_proj with r = 16
  adds per matrix 2 × r × d = 2 × 16 × 4096 ≈ 131k params, × 2 matrices
  × 32 layers ≈ **8.4M trainable params ≈ 0.12 % of 7B**.

## (a) Full fine-tuning — ≈ 112 GB + activations ≈ 115 GB

Every parameter is trained, so every parameter pays for weights,
master weights, gradients and optimiser state:

| Component | Calculation | GB |
|---|---|---|
| bf16 weights | 7 × 10⁹ × 2 B | 14 |
| fp32 master weights | 7 × 10⁹ × 4 B | 28 |
| bf16 gradients | 7 × 10⁹ × 2 B | 14 |
| AdamW m + v (fp32) | 7 × 10⁹ × 8 B | 56 |
| Activations (small batch, no checkpointing) | — | ~2–4 |
| **Total** | | **≈ 114–116 GB** |

The optimiser states alone (56 GB) are 4× the weights. **Verdict:
needs an 80 GB-class GPU only with aggressive tricks, realistically
multi-GPU / 2× A100-80GB or an H100 setup — out of reach of a single
consumer GPU.**

## (b) LoRA — ≈ 16 GB

The base model is frozen: no gradients and no optimiser state for it.
Only the ~8.4M LoRA parameters are trained.

| Component | Calculation | GB |
|---|---|---|
| Frozen base weights, bf16 | 7 × 10⁹ × 2 B | 14.0 |
| LoRA weights (fp32) | 8.4 × 10⁶ × 4 B | 0.03 |
| LoRA gradients (fp32) | 8.4 × 10⁶ × 4 B | 0.03 |
| LoRA AdamW m + v (fp32) | 8.4 × 10⁶ × 8 B | 0.07 |
| Activations (small batch, no checkpointing) | — | ~2 |
| **Total** | | **≈ 16 GB** |

The frozen weights dominate; training cost collapses to <0.2 GB.
**Verdict: fits on a single 24 GB GPU (e.g. RTX 3090/4090, L4);
tight but possible on a 16 GB card with a small batch.**

## (c) QLoRA — ≈ 5–6 GB

Same as LoRA, except the frozen base weights are stored in 4-bit
NormalFloat (NF4, 0.5 B/param) and dequantised on the fly, and
gradient checkpointing is used to shrink activations.

| Component | Calculation | GB |
|---|---|---|
| Frozen base weights, NF4 | 7 × 10⁹ × 0.5 B | 3.5 |
| LoRA weights + grads + AdamW (fp32) | as in (b) | ~0.13 |
| Quantisation constants / small overhead | — | ~0.2 |
| Activations (with gradient checkpointing) | — | ~1 |
| **Total** | | **≈ 5 GB** |

**Verdict: fits on a ~8 GB consumer GPU (e.g. RTX 3060/4060) and on
a free Colab T4 (16 GB) with room to spare.**

## Summary

| Method | Weights | Optimiser + grads | Activations | Total |
|---|---|---|---|---|
| Full fine-tuning | 14 + 28 master | 14 + 56 | ~2–4 | **≈ 115 GB** |
| LoRA (bf16 base) | 14 | ~0.13 | ~2 | **≈ 16 GB** |
| QLoRA (NF4 base) | 3.5 | ~0.13 | ~1 | **≈ 5 GB** |

Takeaway: full fine-tuning a 7B model is dominated by the fp32
master weights and AdamW states (84 GB of the ~115 GB), not by the
model weights themselves. Freezing the base model (LoRA) removes
those terms for 99.9 % of the parameters, and quantising the frozen
weights (QLoRA) removes most of what is left — which is why a 7B
model that needs >100 GB to fully fine-tune can be LoRA-tuned on a
24 GB card and QLoRA-tuned on a free Colab T4.
