import os
import click
import numpy as np
import cv2
from scenedetect import VideoManager
from scenedetect import SceneManager
from scenedetect.detectors import ContentDetector
from scenedetect.scene_manager import SceneDetectorChain, SceneManager
from scenedetect.scene_manager.scene_manager import save_images

class SceneSplitter:
    def __init__(self, scene_detector):
        self.scene_detector = scene_detector

    def split_scenes(self, video_clip):
        for frame_time in video_clip.iter_frames(with_times=True, dtype=float):
            frame_pos_sec, frame_img = frame_time
            frame_img_uint8 = (frame_img * 255).astype(np.uint8)
            self.scene_detector.process_frame(frame_pos_sec, frame_img_uint8)
        return self.scene_detector.get_scene_list() if self.scene_detector else []


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
    video_manager = VideoManager([input_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=30.0))
    base_timecode = video_manager.get_base_timecode()

    try:
        video_manager.set_downscale_factor()
        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager, show_progress=True)
    finally:
        video_manager.release()

    # save images of the detected scenes to disk
    save_images(scene_manager.get_scene_list(base_timecode), video_manager, output_dir)

    return output_files


if __name__ == '__main__':
    main()
