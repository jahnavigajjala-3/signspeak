import csv
from pathlib import Path

import torch
from torch.utils.data import Dataset

from ml.data.preprocessing.frame_sampler import sample_video_frames


class How2SignDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        num_samples: int = 16,
        image_size: int = 224,
    ):
        self.root = Path(root)
        self.split = split
        self.num_samples = num_samples
        self.image_size = image_size

        annotation_path = (
            self.root / "annotations" / f"how2sign_realigned_{split}.csv"
        )
        self.video_dir = self.root / "videos" / split / "raw_videos"

        self.samples = []
        missing = 0

        with open(annotation_path, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")

            for row in reader:
                video_path = (
                    self.video_dir / f"{row['SENTENCE_NAME']}.mp4"
                )

                if not video_path.exists():
                    missing += 1
                    continue

                self.samples.append({
                    "video_path": video_path,
                    "translation": row["SENTENCE"].strip(),
                })

        print(f"Loaded {len(self.samples)} samples")
        print(f"Skipped {missing} missing videos")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        video = sample_video_frames(
            video_path=str(sample["video_path"]),
            num_samples=self.num_samples,
            image_size=self.image_size,
        )

        translation = sample["translation"]

        return video, translation