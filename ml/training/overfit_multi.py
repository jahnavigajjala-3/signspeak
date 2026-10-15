import os
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

from ml.data.datasets.how2sign import How2SignDataset
from ml.data.datasets.vocabulary import Vocabulary
from ml.data.datasets.collate import make_collate_fn
from ml.models.signspeak import SignSpeakModel
from ml.inference.predict import greedy_decode
from ml.training.checkpoint import save_checkpoint


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate_teacher_forced_accuracy(logits, targets, pad_id):
    preds = logits.argmax(dim=-1)
    mask = (targets != pad_id)
    correct = (preds == targets) & mask
    total_valid = mask.sum().item()
    if total_valid == 0:
        return 0.0
    return (correct.sum().item() / total_valid) * 100.0


def main():
    set_seed(42)
    device = torch.device("cpu")
    print(f"Device: {device}")
    print("Initializing FRESH SignSpeakModel (random initialization)...")

    # Load How2Sign dataset
    raw_dataset = How2SignDataset(
        root="data/raw/how2sign",
        split="val",
        num_samples=16,
        image_size=224,
    )

    # Filter 10 deterministic, distinct samples
    unique_samples = []
    seen_translations = set()
    for sample in raw_dataset.samples:
        txt = sample["translation"].strip()
        if txt and txt not in seen_translations and len(txt.split()) >= 3:
            seen_translations.add(txt)
            unique_samples.append(sample)

    selected_samples = random.sample(unique_samples, 10)

    print("\n==================================================")
    print("SELECTED TRAINING EXAMPLES (Deterministic Seed = 42)")
    print("==================================================")
    for idx, s in enumerate(selected_samples):
        v_name = Path(s["video_path"]).name
        print(f"[{idx + 1:02d}] Video: {v_name}")
        print(f"     Path:  {s['video_path']}")
        print(f"     GT:    \"{s['translation']}\"\n")

    # Build vocabulary from selected 10 examples with min_frequency=1
    vocabulary = Vocabulary(min_frequency=1)
    vocabulary.build(s["translation"] for s in selected_samples)

    pad_id = vocabulary.word_to_id["<PAD>"]

    print("==================================================")
    print("VOCABULARY DIAGNOSTICS")
    print("==================================================")
    print(f"Vocab Size: {len(vocabulary)}")
    print(f"Special Tokens: <PAD>={pad_id}, <BOS>={vocabulary.word_to_id['<BOS>']}, "
          f"<EOS>={vocabulary.word_to_id['<EOS>']}, <UNK>={vocabulary.word_to_id['<UNK>']}\n")

    # Instantiate fresh model
    model = SignSpeakModel(vocab_size=len(vocabulary)).to(device)

    # Collate data for selected samples
    collate_fn = make_collate_fn(vocabulary)
    
    # Pre-extract frame tensors and tokens for fast in-memory training
    processed_data = []
    print("Pre-loading frame tensors for training samples...")
    for idx, sample in enumerate(selected_samples):
        v_tensor, txt = raw_dataset[raw_dataset.samples.index(sample)]
        videos, dec_in, targets = collate_fn([(v_tensor, txt)])
        processed_data.append({
            "idx": idx,
            "sample": sample,
            "video_tensor": v_tensor,
            "video_batch": videos.to(device),
            "dec_in": dec_in.to(device),
            "targets": targets.to(device),
            "gt_text": txt,
        })

    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("\n==================================================")
    print("TRAINING LOOP (CPU)")
    print("==================================================")
    
    model.train()
    num_steps = 350
    initial_loss = None
    final_loss = None

    for step in range(1, num_steps + 1):
        total_step_loss = 0.0
        optimizer.zero_grad()

        for item in processed_data:
            videos = item["video_batch"]
            dec_in = item["dec_in"]
            targets = item["targets"]

            logits = model(videos, dec_in)

            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1)
            )

            loss.backward()
            total_step_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        avg_loss = total_step_loss / len(processed_data)

        if step == 1:
            initial_loss = avg_loss
            print(f"Step {step:03d} | Initial Loss: {avg_loss:.4f}")
        elif step % 25 == 0 or step == num_steps:
            print(f"Step {step:03d} | Loss: {avg_loss:.4f}")
            final_loss = avg_loss

    # Save fresh checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        vocabulary=vocabulary,
        epoch=num_steps,
        loss=final_loss,
        path="checkpoints/signspeak_attention_multi_overfit.pt"
    )
    print("\nSaved multi-video overfit checkpoint to 'checkpoints/signspeak_attention_multi_overfit.pt'")

    # ==================================================
    # EVALUATION & DIAGNOSTICS
    # ==================================================
    print("\n==================================================")
    print("EVALUATION & DIAGNOSTICS")
    print("==================================================")

    model.eval()
    accuracies = []
    greedy_predictions = []

    for item in processed_data:
        v_tensor = item["video_tensor"]
        v_batch = item["video_batch"]
        dec_in = item["dec_in"]
        targets = item["targets"]
        gt = item["gt_text"]
        v_name = Path(item["sample"]["video_path"]).name

        with torch.no_grad():
            logits = model(v_batch, dec_in)
            acc = evaluate_teacher_forced_accuracy(logits, targets, pad_id)
            accuracies.append(acc)

            pred_text = greedy_decode(model, vocabulary, v_tensor, device)
            greedy_predictions.append(pred_text)

        print(f"VIDEO: {v_name}")
        print(f"GROUND TRUTH: \"{gt}\"")
        print(f"GREEDY PREDICTION: \"{pred_text}\"")
        print(f"TEACHER-FORCED TOKEN ACCURACY: {acc:.2f}%\n")

    avg_acc = sum(accuracies) / len(accuracies)

    print("==================================================")
    print("SUMMARY METRICS")
    print("==================================================")
    print(f"Number of Examples: {len(processed_data)}")
    print(f"Initial Loss: {initial_loss:.4f}")
    print(f"Final Loss:   {final_loss:.4f}")
    print(f"Average Teacher-Forced Accuracy: {avg_acc:.2f}%")
    print(f"Unique Greedy Predictions: {len(set(greedy_predictions))} / {len(processed_data)}")
    print("==================================================")


if __name__ == "__main__":
    main()
