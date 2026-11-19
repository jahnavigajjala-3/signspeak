# Video-conditioning diagnosis and video-dependence gate

## Summary

The multi-video overfit experiment (v2) showed teacher-forced loss with a
real video (2.2255) and a zero video (2.2251) identical to four decimals.
This write-up checks whether that is a wiring bug and then runs a tiny
gate that makes the video strictly necessary.

**Result: the wiring is correct, but the encoder LSTM saturates and becomes
input-independent, so the attention context is identical for every video.**
The custom CNN features contain real cross-video signal; the LSTM destroys
it. See the numbers below.

---

## Task 1 — Conditioning diagnostics

Script: `ml/debug/check_conditioning.py`

Run:

```bash
python -m ml.debug.check_conditioning
python -m ml.debug.check_conditioning --checkpoint checkpoints/signspeak_attention_multi_overfit.pt
```

### 1. Gradient flow (random initialization)

One backward pass on 4 real training videos:

| Component | Grad norm |
|---|---|
| encoder output (hook) | 1.07e-03 |
| CNN last conv (`cnn.6`) | 5.36e-04 / 2.18e-04 |
| CNN projection | 2.91e-04 / 5.54e-04 |
| temporal LSTM | 9.45e-04 … 2.60e-03 |
| decoder attention | 1.98e-06 / 1.44e-09 / 1.89e-06 |
| decoder (all) | up to 1.48e-01 |

Flag: `decoder.attention` is ~1.35e-5 of the decoder max (>1000x smaller).
No gradient is zero or `None`; the visual path is connected.

With the trained multi-overfit checkpoint the attention grads are larger
(1.16e-03 … 1.80e-02) and no component is flagged, but the encoder output
gradient is still ~100x smaller than the decoder gradient.

### 2. Attention weights and softmax dimension

- `encoder_outputs` shape is `(batch, 16, 256)`; the softmax is taken over
  `dim=-1`, which is the 16-frame time dimension. The resulting weight
  shape is `(batch, 16)` — confirmed at runtime.
- Random init: weights are exactly `1/16` for every frame and every step;
  entropy = 2.7726 = `ln(16)` (uniform).
- Trained multi-overfit checkpoint: entropy drops slightly below uniform at
  later steps (2.65–2.67), i.e. attention is barely non-uniform.

### 3. Context usage (exact source lines)

From `ml/models/translation/decoder.py`:

```
10:         context = encoder_outputs.mean(dim=1)
51:             attended_context = torch.bmm(
56:             current_embedding = self.embedding(
60:             lstm_input = torch.cat(
62:                     current_embedding,
63:                     attended_context
68:             output, (hidden, cell) = self.lstm(
69:                 lstm_input,
73:             logits = self.fc(output)
```

The attended context is concatenated with the token embedding and fed into
the decoder LSTM. It is not dropped or overwritten.

### 4. Feature statistics

Random initialization (4 videos):

| Metric | Value |
|---|---|
| per-video feature mean abs | 5.06e-02 |
| within-video std | 2.15e-04 |
| cross-video variance (of per-video means) | 2.01e-09 |
| cross-video / within-video variance | 0.0301 |
| real vs zero-video feature diff | 1.46e-02 |
| dead ReLU fraction (relu1/2/3) | 0.55 / 0.46 / 0.48 |

Trained multi-overfit checkpoint:

| Metric | Value |
|---|---|
| per-video feature mean abs | 1.83 |
| within-video std | 1.24e-02 |
| cross-video variance | 2.04e-06 |
| cross-video / within-video variance | 0.0093 |
| dead ReLU fraction | 0.45 / 0.21 / 0.50 |

The custom CNN + global average pooling gives weak cross-video signal
relative to within-video signal.

### 5. Gradient-blocking constructs

Static search of `ml/` found:

- `.detach()` only in `decoder.py` lines 146/152, inside the scheduled
  sampling branch (`teacher_forcing_ratio < 1`). With the default ratio 1.0
  this branch is never taken.
- `no_grad`/`eval()` only in validation and inference code.
- No `requires_grad=False` anywhere.

No construct blocks gradients through the encoder during training.

### Diagnosis

There is no wiring bug. The chain is:

1. Custom CNN features have real cross-video variance
   (cross-video std 0.10 vs within-video std 0.028 for the gate model).
2. The 2-layer LSTM encoder collapses that signal: after the LSTM the
   cross-video std of the encoder output is **1.7e-09**, while the
   within-time std is **0.078**. The encoder output is identical across
   videos to 9 decimal places.
3. Attention therefore produces identical scores for every video
   (score diff between videos = 0.0) and the context vector is constant.
4. The decoder converges to the video-independent optimum: a uniform
   distribution over the 5 target words (loss `ln(5)=1.609` per video).

The LSTM collapse is caused by cell-state saturation: in the trained gate
checkpoint the cell state norm is **240** and **95.3%** of cell-state units
are saturated (`|c| > 0.99`). Once `tanh(c)` saturates, the LSTM output no
longer depends on its input.

---

## Task 2 — Video-dependence gate

Script: `ml/training/video_dependence_gate.py`

Task: 5 distinct training videos, each mapped to one distinct target word:

```
video_i -> <BOS> word_i <EOS>
words = ["water", "apple", "music", "happy", "friend"]
```

The decoder's first input is always `<BOS>`, so the first predicted token
can only be correct if the model conditions on the video.

### Baseline result (custom CNN + LSTM + attention)

- TF loss real video: **0.8054**
- TF loss zero video: **0.8054**
- Loss gap: **0.0000** (FAIL, needs >= 1.0)
- Greedy correct: **1/5** (FAIL, needs 5/5); all videos predicted "music"
- Shuffled accuracy: **0.20** (PASS, <= chance 0.20)
- **OVERALL: FAIL**

Per-video analysis of the baseline checkpoint:

| Video | Target | Step-0 loss | Step-1 loss | Zero-video step-0 loss |
|---|---|---|---|---|
| 0 | water | 1.610 | 0.000 | 1.610 |
| 1 | apple | 1.611 | 0.000 | 1.611 |
| 2 | music | 1.610 | 0.000 | 1.610 |
| 3 | happy | 1.610 | 0.000 | 1.610 |
| 4 | friend | 1.610 | 0.000 | 1.610 |

Step-0 loss = `ln(5)` for every video: the model learned a uniform
distribution over the five targets and no video dependence. Real-vs-zero
logit difference is **7.3e-09**.

### Bisection

(Results pending — variants `layernorm`, `high_lr`, `resnet`.)

---

## Recommended next step

(pending)
