# Multi-Video Overfitting Experiment

## Objective

Determine whether the current SignSpeak pipeline
(video → CNN → LSTM temporal encoder → attention decoder)
can learn multiple training examples simultaneously, and whether
greedy autoregressive decoding produces video-dependent output.

This is an intentional-overfitting experiment. **No generalization is claimed.**

## Dataset

- Dataset: How2Sign RGB front videos (local copy).
- Split used: `val`.
- 10 deterministic (`seed=42`) samples, distinct video IDs,
  distinct/mostly distinct translations, non-empty English.

## Architecture

Videos are sampled at 16 evenly spaced frames, resized to 224×224, normalized to [0,1]:

```
video [16,3,224,224]
→ VisualEncoder (custom CNN) → [16,512]
→ TemporalEncoder (2-layer LSTM, hidden 256) → [16,256]
→ TranslationDecoder (2-layer LSTM + Bahdanau attention) → vocab logits
```

## Training configuration

- Device: CPU (MPS disabled for training due to prior Metal errors)
- Teacher forcing with `CrossEntropyLoss(ignore_index=<PAD>)`
- Optimizer: Adam, lr=1e-3
- Gradient clipping: max_norm=1.0
- Steps: 350 (overfitting one batch of 10 examples)
- Fresh `SignSpeakModel` from random initialization
  (NOT from `signspeak_attention_best.pt` / one-video checkpoint)
- Checkpoint: `checkpoints/signspeak_attention_multi_overfit.pt`

## Results

- Initial loss: **5.0370**
- Final loss: **0.0874**
- Average teacher-forced token accuracy: **96.84%**
- Number of examples: 10
- Unique greedy predictions: **2 / 10**

### Per-video predictions

| Video | Ground truth (truncated) | Greedy prediction | Teacher-forced acc |
|---|---|---|---|
| dWOtJxrH_Ww_8-8-rgb_front.mp4 | temperature controls… | avocado/guacamole sentence | 96.43% |
| 2ri5HYm48MA_8-2-rgb_front.mp4 | develop a really great character… | avocado/guacamole sentence | 98.00% |
| -f1_kdl050s_9-1-rgb_front.mp4 | bird flies out… | avocado/guacamole sentence | 94.59% |
| BZqXT5UYUD8_4-5-rgb_front.mp4 | forward a phone call… | avocado/guacamole sentence | 98.11% |
| 4XgOVMG-zIo_7-1-rgb_front.mp4 | Just grab that. | **just grab that.** | 100% |
| 4GG2H4QR9W8_5-2-rgb_front.mp4 | The avocado for the guacamole… | **avocado/guacamole sentence** | 100% |
| 33jxeIIbBnM_31-1-rgb_front.mp4 | Excel macro from Access… | "just grab that." | 95.00% |
| 279MO2nwC_E_9-2-rgb_front.mp4 | more movement/oxygen… | avocado/guacamole sentence | 97.50% |
| eY32ru3Nstc_10-8-rgb_front.mp4 | grenadine… | avocado/guacamole sentence | 93.33% |
| bjvIIdYWc-k_7-8-rgb_front.mp4 | ceiling paint finish… | avocado/guacamole sentence | 95.45% |

## Diagnosis

### Teacher-forced accuracy is misleading — decoder ignores video

- Replacing the real video with **zeros** during teacher forcing produces
  almost identical loss (e.g. video 0: 6.0806 vs 6.0792; video 1: 9.3630 vs 9.3659).
  The video contribution to the teacher-forced objective is negligible.
- Greedy decoding on a zero/noise video still returns the same two memorized
  sentence attractors — confirming predictions are **not video-dependent**.
- Gradient norms on the trained model confirm the cause:
  - decoder.lstm gradients: ~5–14
  - decoder.embedding: ~0.6
  - decoder.attention_* : ~0.002–0.004
  - visual_encoder / video-side projection: ~0.001–0.006
  The decoder LSTM + embedding (language-model pathway) dominate training;
  the video pathway receives 2–3 orders of magnitude smaller gradients.

### Why teacher forcing can still hit 97% on the training batch

With teacher forcing the decoder sees the ground-truth token history.
A sentence's token history deterministically determines its next token,
so the decoder can drive loss to ~0.09 by memorizing each training
sequence through token history alone, without using the video at all.

### Greedy decoding collapses

Greedy decoding only reaches 2 unique predictions out of 10 videos,
reproducing training-set "attractor" sentences. This is the classic
exposure-bias / teacher-forcing mismatch, compounded by the decoder
not conditioning on video.

## Success-criteria check

| Criterion | Result |
|---|---|
| Loss decreases substantially | ✅ 5.04 → 0.087 |
| Teacher-forced accuracy high | ✅ ~97% |
| Different videos → different predictions | ❌ 2/10 unique |
| Greedy decoding more recognizable | ⚠️ only for 2 videos |
| No universal EOS collapse | ✅ |
| No universal repeated sentence | ❌ |

## Conclusion

The training pipeline (data → CNN → LSTM → attention → decoder) **runs and
reduces loss**, but under the current configuration it does **not** learn a
video→text mapping — the model memorizes the token sequences of the training
examples via the decoder language-model pathway while the CNN/LSTM/attention
pathway remains effectively unused (near-zero video contribution, near-zero
attention/lower-pathway gradients).

The repeated-translation behavior seen in production
(`signspeak_attention_overfit.pt`) is therefore **not fundamentally different
in nature** from what this fresh multi-video run produced: the model
behaves as a language model over the training sentences rather than a
video-conditioned translator.

Next steps (evidence-based, not yet done):

1. Force the conditioning signal to be used before scaling: short distinct
   targets per distinct video, and check video-dependence of greedy output
   as a gate for any real training run.
2. Investigate why attention weights / encoder outputs contribute ~zero
   signal (near-uniform encoder temporal variance ~0.086; small encoder
   feature differences; gradient norms ~1e-3 on visual path).
3. Consider that at How2Sign scale the LM pathway cannot memorize every
   sentence, so video signal may become necessary — but this must be
   verified, not assumed, possibly with stronger visual features.

No generalization is claimed. This checkpoint must NOT replace the
production checkpoint.
