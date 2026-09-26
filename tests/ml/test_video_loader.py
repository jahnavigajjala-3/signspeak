from ml.data.loaders.video_loader import read_video_frame

video_path = "data/raw/how2sign/videos/train/raw_videos/DvKS20WKiX4_22-8-rgb_front.mp4"

frame = read_video_frame(video_path, 0)

print("Shape:", frame.shape)
print("Data type:", frame.dtype)
print("Pixel:", frame[100, 100])
