import os
import cv2
import tempfile
import unittest
import numpy as np
from ml.data.preprocessing.frame_sampler import sample_video_frames
from backend.app.services.inference_service import InferenceService


class TestVideoIngestion(unittest.TestCase):
    def test_mp4_video_ingestion(self):
        mp4_path = "data/raw/how2sign/videos/test/raw_videos/-fZc293MpJk_0-1-rgb_front.mp4"
        if not os.path.exists(mp4_path):
            self.skipTest("Test video MP4 not found")

        # Verify tensor shape
        tensor = sample_video_frames(mp4_path)
        self.assertEqual(tensor.shape, (16, 3, 224, 224))

        # Verify video info extraction
        info = InferenceService.extract_video_info(mp4_path)
        self.assertGreater(info["frame_count"], 0)
        self.assertGreater(info["duration_sec"], 0)
        self.assertGreater(info["width"], 0)
        self.assertGreater(info["height"], 0)

    def test_webm_video_ingestion(self):
        # Create a synthetic WebM file simulating browser MediaRecorder output
        tmp_dir = tempfile.mkdtemp()
        webm_path = os.path.join(tmp_dir, "test_camera_recording.webm")

        try:
            fourcc = cv2.VideoWriter_fourcc(*"VP80")
            out = cv2.VideoWriter(webm_path, fourcc, 30.0, (640, 480))
            for i in range(45):
                img = np.full((480, 640, 3), i * 5, dtype=np.uint8)
                out.write(img)
            out.release()

            self.assertTrue(os.path.exists(webm_path))

            # Test frame sampling on WebM file
            tensor = sample_video_frames(webm_path)
            self.assertEqual(tensor.shape, (16, 3, 224, 224))

            # Test video info extraction on WebM file
            info = InferenceService.extract_video_info(webm_path)
            self.assertEqual(info["frame_count"], 45)
            self.assertEqual(info["width"], 640)
            self.assertEqual(info["height"], 480)

        finally:
            if os.path.exists(tmp_dir):
                import shutil
                shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_empty_video_ingestion_error(self):
        tmp_dir = tempfile.mkdtemp()
        empty_video_path = os.path.join(tmp_dir, "empty_file.mp4")
        with open(empty_video_path, "wb") as f:
            f.write(b"")

        try:
            with self.assertRaises(ValueError) as ctx:
                sample_video_frames(empty_video_path)
            err_msg = str(ctx.exception)
            self.assertTrue("no readable frames" in err_msg or "cannot be opened" in err_msg)
        finally:
            if os.path.exists(tmp_dir):
                import shutil
                shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
