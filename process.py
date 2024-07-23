import os
import cv2
import subprocess

def is_blurry(image, threshold=100.0):
    """Return True if image is blurry."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < threshold

def extract_frames(video_path, frames_dir):
    """Extract frames from a video using FFmpeg."""
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)
    
    command = [
        'ffmpeg', '-i', video_path,
        os.path.join(frames_dir, 'frame_%05d.jpg')
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def remove_blurry_frames(frames_dir, threshold=100.0):
    """Remove blurry frames from the directory."""
    for frame_file in os.listdir(frames_dir):
        frame_path = os.path.join(frames_dir, frame_file)
        image = cv2.imread(frame_path)
        
        if is_blurry(image, threshold):
            os.remove(frame_path)

def process_videos(input_dir, output_dir, threshold=100.0):
    """Process all videos in the input directory."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for video_file in os.listdir(input_dir):
        if video_file.endswith(('.mp4', '.avi', '.mov', '.mkv')):  # Add other video formats if needed
            video_path = os.path.join(input_dir, video_file)
            video_name = os.path.splitext(video_file)[0]
            frames_dir = os.path.join(output_dir, video_name)
            
            # Extract frames from the video
            extract_frames(video_path, frames_dir)
            
            # Remove blurry frames
            remove_blurry_frames(frames_dir, threshold)
            
            print(f"Processed video: {video_file}")

# 使用示例
input_dir = 'input'
output_dir = 'output'
threshold = 100.0  # Adjust the threshold as needed

process_videos(input_dir, output_dir, threshold)
