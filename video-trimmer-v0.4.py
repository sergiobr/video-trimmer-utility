import os
import cv2
import click
import numpy as np
from moviepy.editor import VideoFileClip
from scenedetect.detectors import ContentDetector

from moviepy.editor import VideoFileClip

class SceneSplitter:
    def __init__(self, scene_detector):
        self.scene_detector = scene_detector

    def split_scenes(self, video_manager):
        video_clip = VideoFileClip(video_manager.filename)

        for frame_number in range(0, int(video_clip.fps * video_clip.duration), int(video_clip.fps)):
            frame_img = video_clip.get_frame(frame_number / video_clip.fps)
            self.scene_detector.process_frame(frame_number, frame_img)

        return self.scene_detector.get_scene_list()

@click.command()
@click.argument('input_path', type=click.Path(exists=True, dir_okay=False))
@click.option('--max-scenes', default=2, help='Maximum number of scenes to extract.')
@click.option('--crop-size', default=256, help='Crop size for the video. 0 for no crop.')
@click.option('--min-frames-per-scene', default=100, help='Minimum number of frames per scene.')
@click.option('--max-frames-per-scene', default=200, help='Maximum number of frames per scene.')
@click.option('--output-dir', default='output', help='Directory where the extracted scenes will be saved.')
def main(input_path, max_scenes, crop_size, min_frames_per_scene, max_frames_per_scene, output_dir):
    output_files = process_video(
        input_path, max_scenes, crop_size, min_frames_per_scene, max_frames_per_scene, output_dir)
    print(f"Extracted scenes saved to: {output_dir}")

def process_video(input_path, max_scenes, crop_size, min_frames_per_scene,
                  max_frames_per_scene, output_dir):
    video_info = get_video_info(input_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    scenes = detect_scenes(input_path, video_info, max_scenes, crop_size,
                           min_frames_per_scene, max_frames_per_scene, output_dir)
    return scenes


def get_video_info(input_path):
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return {"width": width, "height": height, "fps": fps, "total_frames": total_frames}


def detect_scenes(input_path, video_info, max_scenes, crop_size, min_frames_per_scene, max_frames_per_scene, output_dir):
    scene_detector = ContentDetector(threshold=30.0, min_scene_len=min_frames_per_scene)
    scene_splitter = SceneSplitter(scene_detector)

    video_clip = VideoFileClip(input_path)

    # Initialize the scene detector with the video frame size
    frame_size = video_clip.size
    scene_detector.frame_width, scene_detector.frame_height = frame_size

    # Read and process each frame
    for frame_time in video_clip.iter_frames(with_times=True, dtype=float):
        frame_pos_sec, frame_img = frame_time
        video_manager = (frame_pos_sec, video_clip.get_frame(frame_pos_sec))

        # Convert the frame dtype to uint8
        frame_img_uint8 = (frame_img * 255).astype(np.uint8)

        scene_detector.process_frame(frame_pos_sec, frame_img_uint8)

    scene_splitter.split_scenes(scene_detector)
    output_files = scene_splitter.split_video(
        output_dir, os.path.splitext(os.path.basename(input_path))[0])
    return output_files


if __name__ == '__main__':
    main()
