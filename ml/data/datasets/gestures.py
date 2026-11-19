from pathlib import Path

import torch
from torch.utils.data import Dataset

from ml.data.preprocessing.frame_sampler import sample_video_frames


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


class GestureDataset(Dataset):
    """Loads isolated gesture videos from one folder per class."""

    def __init__(
        self,
        root: str,
        num_samples: int = 16,
        image_size: int = 224,
        class_names=None,
    ):
        self.root = Path(root)
        self.num_samples = num_samples
        self.image_size = image_size

        if not self.root.is_dir():
            raise ValueError(f"Gesture dataset root does not exist: {root}")

        if class_names is None:
            class_names = sorted(
                path.name for path in self.root.iterdir() if path.is_dir()
            )

        self.class_names = list(class_names)
        self.class_to_id = {
            class_name: index for index, class_name in enumerate(self.class_names)
        }

        self.samples = []
        for class_name in self.class_names:
            class_dir = self.root / class_name
            for video_path in sorted(class_dir.rglob("*")):
                if video_path.is_file() and video_path.suffix.lower() in VIDEO_EXTENSIONS:
                    self.samples.append((video_path, self.class_to_id[class_name]))

        if not self.samples:
            raise ValueError(
                f"No supported gesture videos found under: {root}"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        video_path, class_id = self.samples[index]
        video = sample_video_frames(
            str(video_path),
            num_samples=self.num_samples,
            image_size=self.image_size,
        )
        return video, torch.tensor(class_id, dtype=torch.long)
