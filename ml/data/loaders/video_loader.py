import cv2
import numpy as np

def read_video_frame(Video_path:str,frame_index:int) -> np.ndarray:

   cap = cv2.VideoCapture(Video_path)
   if  not cap.isOpened():
      raise ValueError("The Video Cant be opened")
   try:
     cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
     success, frame = cap.read()
     if not success:
            raise ValueError(f"Could not read frame {frame_index}")

     return frame

   finally:
     cap.release()
   


      