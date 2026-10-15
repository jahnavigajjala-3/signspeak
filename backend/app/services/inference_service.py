import os
import time
import cv2
import torch
import logging
from typing import Dict, Any, Tuple
from ml.inference.predict import predict_video, get_default_checkpoint
from ml.data.preprocessing.frame_sampler import sample_video_frames, sample_frame_indices

logger = logging.getLogger("signspeak.inference")
logging.basicConfig(level=logging.INFO)


class InferenceService:
    @staticmethod
    def extract_video_info(video_path: str) -> Dict[str, Any]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {
                "duration_sec": 0.0,
                "fps": 0,
                "width": 0,
                "height": 0,
                "frame_count": 0
            }
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
        
        # Fallback for browser WebM recordings where CAP_PROP_FRAME_COUNT is 0
        if frame_count <= 0:
            count = 0
            w, h = 0, 0
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                if count == 0:
                    h, w = frame.shape[:2]
                count += 1
            frame_count = count
            if width == 0:
                width = w
            if height == 0:
                height = h

        cap.release()
        
        duration_sec = round(frame_count / fps, 2) if fps > 0 else 0.0
        
        return {
            "duration_sec": duration_sec,
            "fps": round(fps, 1),
            "width": width,
            "height": height,
            "frame_count": frame_count
        }

    @classmethod
    def run_inference(cls, video_path: str) -> Tuple[str, float, Dict[str, Any], str]:
        start_time = time.time()
        
        filename = os.path.basename(video_path)
        file_size = os.path.getsize(video_path) if os.path.exists(video_path) else 0

        # Diagnostics: Get video info and sample frame indices
        video_info = cls.extract_video_info(video_path)
        frame_count = video_info["frame_count"]
        
        if frame_count <= 0:
            raise ValueError("Video file contains no readable frames")

        sampled_indices = sample_frame_indices(frame_count, 16).tolist()
        
        checkpoint_path = get_default_checkpoint()
        
        # Execute PyTorch pipeline
        translation = predict_video(video_path, checkpoint_path=checkpoint_path)
        
        elapsed_ms = round((time.time() - start_time) * 1000, 2)

        # Lightweight logging for diagnostic tracking
        logger.info(
            f"[Inference Diagnostic] file='{filename}' size={file_size}B frames={frame_count} "
            f"sampled_indices={sampled_indices} duration={elapsed_ms}ms"
        )
        
        return translation, elapsed_ms, video_info, checkpoint_path

    @staticmethod
    def get_system_device() -> str:
        if torch.backends.mps.is_available():
            return "mps (Apple Silicon Metal)"
        elif torch.cuda.is_available():
            return "cuda (NVIDIA GPU)"
        return "cpu"
