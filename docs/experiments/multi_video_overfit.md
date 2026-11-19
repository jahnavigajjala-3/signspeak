# Multi-Video Overfitting Experiment

## Objective

Determine whether the current SignSpeak pipeline
(video → CNN → LSTM temporal encoder → attention decoder)
can learn multiple training examples simultaneously, and whether
greedy autoregressive decoding produces video-dependent output.

This is an intentional-overfitting experiment. **No generalization is claimed.**

## Dataset

- Dataset: How2Sign RGB front videos (local copy).
- Split used: `train`.
- 10 deterministic (`seed=42`) samples with:
  - distinct underlying How2Sign video IDs,
  - distinct translations,
  - at least 4 tokens per translation,
  - non-empty English.

## Architecture

Videos are sampled at 16 evenly spaced frames, resized to 224×224, normalized to [0,1]:

```
video [16,3,224,224]
→ VisualEncoder (custom CNN) → [16,512]
→ TemporalEncoder (2-layer LSTM, hidden 256) → [16,256]
→ TranslationDecoder (2-layer LSTM + Bahdanau attention) → vocab logits
```

The complete encoder sequence is passed to attention.
Mean pooling is used only for decoder hidden-state initialisation.

## Training configuration

- Device: CPU (MPS disabled for training due to prior Metal errors)
- Fresh `SignSpeakModel` from random initialization
  (NOT from `signspeak_attention_best.pt` / one-video checkpoint)
- Vocabulary: existing production vocabulary loaded from
  `signspeak_attention_best.pt` (9,866 tokens)
- CrossEntropyLoss(ignore_index=`<PAD>`)
- Optimizer: Adam, lr=5e-4
- Batch size: 4 (gradient accumulation per update)
- Gradient clipping: max_norm=1.0
- Steps: 400
- **Scheduled sampling**: teacher-forcing ratio linearly decays
  1.0 → 0.4 (exposes the decoder to its own predictions to reduce
  exposure bias and force it to use the attention context)
- Checkpoint: `checkpoints/signspeak_attention_multi_overfit.pt`
  (previous run preserved as
  `checkpoints/signspeak_attention_multi_overfit_v1.pt`)

## Results (run v2, existing vocabulary + scheduled sampling)

- Initial loss: **9.1940**
- Final loss: **2.6358** (loss still decreasing at step 400)
- Average teacher-forced token accuracy: **39.34%**
- Average teacher-forced loss, real video: **2.2255**
- Average teacher-forced loss, zero video: **2.2251**
- Unique greedy predictions: **2 / 10**

### Per-video predictions

| Video | Ground truth (truncated) | Greedy prediction | TF acc |
|---|---|---|---|
| CUWThEnO-OQ_0-5 | sports bra / camisole… | `this we'' going to to to a a a…` | 22.03% |
| fHT1NhNUono_1-8 | learning to play Portal… | `this we'' going to to to a a a…` | 36.84% |
| f1R7MZSlOOg_1-5 | black bean… | `this we'' going to to to a a a…` | 46.15% |
| _BOJ9cYPRe4_12-8 | took it out of the ground… | `this we'' going to to to a a a…` | 25.00% |
| 11ysPjNRN4Y_1-1 | zones / handwriting… | `this we'' going to to to a a a…` | 33.33% |
| 0-bR2rq-qps_0-5 | astringent… | `this we'' going to to to a a a…` | 42.11% |
| aygp09D7Z_w_0-5 | hit a ball inside/outside/up… | `this we'' going to to to a a a…` | 51.43% |
| 00m1mQFF5zo_0-3 | taking messages… | `this we'' going to to to a a a…` | 41.67% |
| CylhtiqMiTc_0-8 | Large Arm Circles… | `this we'' going to to to a a a…` | 62.50% |
| ahki1p2lqPI_0-8 | Guillermo Gomez / soccer… | `this we'' going to to to a a a…` | 32.35% |

Exact greedy text is one of two repeated fragments:

```
this we'' going to to to a a a,,,,,'''''t t a a a,,,'''' t t think a a,,,'''' t t to a a
```

## Diagnosis

### The decoder does not use the video

- Teacher-forced loss with the **real video** and with an **all-zero video**
  is identical to 4 decimal places (difference < 0.001) for every example.
- Greedy decoding produces the same degenerate fragment for 9/10 videos.
- Therefore the model memorised/partially fit token sequences through the
  decoder language-model pathway; the CNN/LSTM/attention path contributes
  almost nothing to the prediction.

### Gradient evidence (run v1 checkpoint)

Gradient norms measured on the 10-example overfit model:

| Component | Gradient norm |
|---|---|
| `decoder.lstm` | ~5–14 |
| `decoder.embedding` | ~0.6 |
| `decoder.fc` | ~3.8 |
| `decoder.attention_*` | ~0.002–0.004 |
| `visual_encoder` (all conv/projection) | ~0.001–0.006 |
| `temporal_encoder` layer 1 | ~0.005–0.016 |

The video pathway receives 2–3 orders of magnitude less gradient than the
decoder language-model pathway.

### Scheduled sampling did not fix it

Run v2 added scheduled sampling and the production vocabulary.
Loss decreased but greedy output collapsed into a repetitive fragment,
and real-vs-zero-video loss remained identical. The model is still not
conditioned on the input video.

### Why teacher-forced accuracy can be misleading

With teacher forcing the decoder sees the ground-truth history.
A training sentence's token history determines its next token, so the
decoder can fit the batch through token history alone, without using the
video. That is exactly what happened in run v1 (97% TF accuracy,
0.087 loss, but only 2/10 unique greedy predictions and zero video
dependence).

## Success-criteria check

| Criterion | Result |
|---|---|
| Loss decreases substantially | ✅ (v1 5.04 → 0.087; v2 9.19 → 2.64) |
| Teacher-forced accuracy high | ⚠️ v1 yes (97%); v2 no (39%, underfit) |
| Different videos → different predictions | ❌ 2/10 unique |
| Greedy decoding more recognizable | ❌ degenerate repetition |
| No universal EOS collapse | ✅ |
| No universal repeated sentence | ❌ |
| Video actually influences predictions | ❌ real-vs-zero loss identical |

## Conclusion

The data → CNN → LSTM → attention → decoder pipeline runs and reduces
loss, but under the current configuration it does **not** learn a
video→text mapping. The decoder fits the training targets mainly through
token history, while the video pathway remains effectively unused.

The repeated translation observed in the product
(`signspeak_attention_overfit.pt`) is the expected output of a checkpoint
trained on one video. It is **not** caused by the backend reusing the same
upload; different uploads were verified to produce different tensors and
CNN features (see the takeover report).

No generalization is claimed. This checkpoint must NOT replace the
production checkpoint.

## Product symptom: the same sentence for every upload

The UI symptom — every upload returns
`"we're going to work on a arm drill that will help you you <UNK> hand hand
movements in front you you you you"` — is not a backend upload bug:

- The frontend creates a fresh `FormData` and appends the currently selected
  `File` on every request (`frontend/src/lib/api.ts`).
- Different How2Sign videos produce different tensors and different CNN
  features (verified), so different content reaches the model.
- The default checkpoint is `signspeak_attention_overfit.pt`, trained on one
  validation video. It reproduces that video's memorised sentence.
- The other available checkpoints are not usable substitutes:
  `signspeak_attention_best.pt` greedily emits EOS (empty text) and
  `signspeak_best.pt` has an incompatible pre-attention state dict.
- The new multi-video checkpoint also collapses to repeated fragments and is
  not production-quality.

Therefore the repeated output cannot be fixed by frontend/backend logic or by
switching to an existing checkpoint; it requires a model that actually learns
a video→text mapping.

## Next steps (evidence-based)


1. Gate any training run on a **video-dependence check**: real-video vs
   zero/swapped-video loss must differ meaningfully. Current checkpoints fail.
2. Investigate why the visual/attention gradients are ~1e-3. Candidate
   causes: weak per-frame CNN features, 16-frame sampling from long clips,
   attention uniformly distributed, and the decoder's ability to fit
   token history without context.
3. Only after a config passes the video-dependence check should full-scale
   How2Sign training be attempted. Scaling alone will not fix a decoder
   that ignores its input.
