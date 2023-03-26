import datetime
import json
import typing
from typing import List, Dict, Any, Tuple, Optional, Union
import unicodedata
import cv2
import numpy as np
from tqdm import tqdm
from scenedetect import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.detectors import ContentDetector
from scenedetect.frame_timecode import FrameTimecode
from tqdm import tqdm
import numpy as np
import gradio as gr
import subprocess, os
from werkzeug.utils import secure_filename

def crop_center(frame: np.ndarray, crop_size: Tuple[int, int]) -> np.ndarray:
    """
    Crops the center square of a frame with the given crop size.

    Args:
        frame (np.ndarray): The input frame to crop.
        crop_size (Tuple[int, int]): The target size of the cropped square.

    Returns:
        np.ndarray: The cropped frame.
    """
    h, w = frame.shape[:2]
    if h > w:
        start = (h - w) // 2
        return frame[start:start+w, :, :]
    else:
        start = (w - h) // 2
        return frame[:, start:start+h, :]


def calculate_sharpness(frame: np.ndarray) -> float:
    """
    Calculates the sharpness of a frame.

    Args:
        frame (np.ndarray): The input frame.

    Returns:
        float: The sharpness score.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def detect_scenes(video_info: List[dict], max_scenes: int, crop_size: Tuple[int, int], min_frames_per_scene: int, max_frames_per_scene: int) -> List[FrameTimecode]:
    # Initialize scene detection
    video_path = video_info[0]["filename"]
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=30, min_scene_len=min_frames_per_scene))
    base_timecode = video_manager.get_base_timecode()

    # Start video processing
    video_manager.set_crop(crop_left=0, crop_right=crop_size[0], crop_top=0, crop_bottom=crop_size[1])
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)

    # Skip scenes with too few or too many frames
    new_scenes = []
    for i, scene in enumerate(scene_manager.get_scene_list(base_timecode)):
        if scene[1]-scene[0] > max_frames_per_scene:
            continue
        if scene[1]-scene[0] < min_frames_per_scene:
            continue
        new_scenes.append(scene)
        if len(new_scenes) >= max_scenes:
            break

    # Convert FrameTimecodes
    scenes = []
    for scene in new_scenes:
        start_time = base_timecode + scene[0]
        end_time = base_timecode + scene[1] - 1
        scenes.append({"start_time": start_time, "end_time": end_time})
    return scenes

def trim_video(start_time: float, end_time: float, input_path: str, output_path: str):
    """
    Trim a video from start_time to end_time and save the result at output_path.

    Args:
        start_time (float): Start time in seconds.
        end_time (float): End time in seconds.
        input_path (str): Path to the input video file.
        output_path (str): Path to the output video file.
    """

    # Set start and end times
    start = str(datetime.timedelta(seconds=start_time))
    end = str(datetime.timedelta(seconds=end_time))

    # Run FFmpeg to trim video
    cmd = ["ffmpeg", "-i", input_path, "-ss", start, "-to", end, "-c", "copy", output_path]
    subprocess.run(cmd, capture_output=True, check=True)

import re

import json
import subprocess

def get_video_info(video_path: str) -> dict:
    """
    Returns a dictionary with information about the video file.
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,width,height,duration,r_frame_rate,bit_rate",
        "-of", "json",
        video_path
    ]
    output = subprocess.check_output(cmd).decode("utf-8")
    info = json.loads(output)
    stream = info['streams'][0]
    return {
        'codec_name': stream.get('codec_name', ''),
        'width': stream.get('width', 0),
        'height': stream.get('height', 0),
        'duration': stream.get('duration', 0),
        'frame_rate': eval(stream.get('r_frame_rate', 0)),
        'bit_rate': stream.get('bit_rate', 0),
    }


######
from typing import List, Tuple, Optional
import tqdm

def process_video(input_path: str, max_scenes: int, crop_size: Tuple[int, int], 
                  min_frames_per_scene: int, max_frames_per_scene: int, output_dir: str) -> List[str]:
    # Normalize input filename to standardized form
    input_path = unicodedata.normalize('NFKD', input_path).encode('ascii', 'ignore').decode()
    # Get video info
    #cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", video_path]
    cmd = f'ffprobe -v error -select_streams v:0 -show_entries stream=codec_name,width,height,duration,r_frame_rate,bit_rate -of json "{input_path}"'

    result = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = result.communicate()[0]
    video_info = [stream for stream in sorted(
        eval(output.decode())["streams"], key=lambda x: x["codec_type"] == "video", reverse=True
    ) if "width" in stream]

    # Detect scenes
    scenes = detect_scenes(video_info, max_scenes, crop_size, min_frames_per_scene, max_frames_per_scene)
    scenes.sort(key=lambda scene: scene.start_time)

    # Trim scenes
    output_files = []
    for i, scene in enumerate(scenes):
        start_time = scene.start_time.get_seconds()
        end_time = scene.end_time.get_seconds()
        output_path = os.path.join(output_dir, f"scene_{i}.mp4")
        trim_video(start_time, end_time, video_path, output_path)
        output_files.append(output_path)
    return output_files
#######

class Scene:
    def __init__(self, scene_num, start_time, end_time):
        self.scene_num = scene_num
        self.start_time = start_time
        self.end_time = end_time
        
def extract_scenes(video_path: str, num_scenes: int, threshold: float) -> List[Scene]:
    # Initialize scene detector and video manager
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    video_manager = VideoManager([video_path])
    video_manager.set_downscale_factor()
    video_manager.start()

    # Perform scene detection
    scene_list = []
    i = 0
    for frame_num, _ in video_manager:
        # Check if the current frame contains a scene change
        scene_manager.detect(frame_num, _)
        if scene_manager.is_new_scene():
            start = video_manager.get_duration(previous=True)
            end = video_manager.get_duration()
            # Add the scene to the list
            scene_list.append(Scene(i, start, end))
            i += 1
            if len(scene_list) >= num_scenes:
                break
    video_manager.release()
    return scene_list


def get_duration(video_path: str) -> float:
    """
    Get the duration of a video in seconds.

    Args:
        video_path (str): The path to the video file.

    Returns:
        float: The duration of the video in seconds.
    """
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
           "default=noprint_wrappers=1:nokey=1", video_path]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output = result.stdout.decode().strip()
    return float(output)

def trim_video(start_time: float, end_time: float, input_path: str, output_path: str) -> None:
    """
    Trims a video file from start_time to end_time and saves it to output_path.
    Args:
        start_time (float): Start time in seconds.
        end_time (float): End time in seconds.
        input_path (str): Path to input video file.
        output_path (str): Path to output video file.
    Returns:
        None.
    """
    duration = end_time - start_time
    cmd = ["ffmpeg", "-i", input_path, "-ss", str(start_time), "-t", str(duration), "-c:v", "libx264", "-c:a", "aac",
           "-strict", "-2", "-loglevel", "error", "-y", output_path]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

def get_frame_metrics(frame_path: str, crop_size: Tuple[int, int]) -> typing.Dict[str, typing.Union[float, str]]:
    """
    Calculates the metrics for a given frame, including its average color, sharpness, and size.

    Args:
    - frame_path: str, the path to the frame file.
    - crop_size: Tuple[int, int], the size to crop the frame.

    Returns:
    - Dict[str, Union[float, str]], a dictionary containing the calculated metrics.
    """
    frame = cv2.imread(frame_path)
    cropped_frame = crop_center(frame, crop_size)
    avg_color = np.mean(cropped_frame, axis=(0, 1)).tolist()
    sharpness = calculate_sharpness(cropped_frame)
    size = os.path.getsize(frame_path)

    return {
        "avg_color": avg_color,
        "sharpness": sharpness,
        "size": size,
        "filename": os.path.basename(frame_path),
    }

def trim_video(start_time: float, end_time: float, input_path: str, output_path: str) -> None:
    """
    Trims the video from the given start time to end time and saves it to the output path.

    Args:
        start_time (float): Start time of the trimmed video in seconds.
        end_time (float): End time of the trimmed video in seconds.
        input_path (str): Path of the input video file.
        output_path (str): Path of the output video file.
    """
    duration = end_time - start_time
    cmd = ["ffmpeg", "-y", "-ss", str(start_time), "-i", input_path, "-t", str(duration), "-c", "copy", output_path]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
import tempfile    
    
class VideoTrimmer:
    def __call__(self, input_file: str, max_scenes: int, crop_size: Tuple[int, int], min_frames_per_scene: int, max_frames_per_scene: int, output_dir: str) -> str:
        try:
            # Get the path to the uploaded video file
            # input_path = os.path.join(tempfile.gettempdir(), input_file.name)
            input_path = os.path.join(tempfile.gettempdir(), secure_filename(input_file.name))

            with open(input_path, "wb") as f:
                f.write(input_file.read())
            os.chmod(input_path, 0o777)

            os.makedirs(output_dir, exist_ok=True)
            output_files = process_video(input_path, max_scenes, crop_size, min_frames_per_scene, max_frames_per_scene, output_dir)
            return f"Processed {len(output_files)} scenes: {', '.join(output_files)}"
        except Exception as e:
            return f"Error processing video: {e}"
    
iface = gr.Interface(
    fn=VideoTrimmer(),
    inputs=[
        gr.inputs.File(label="Video file"),
        gr.inputs.Number(label="Maximum number of scenes to extract", default=1),
        gr.inputs.Dropdown(label="Crop size", choices=[(240, 240), (256, 256), (480, 480), (512, 512), (720, 720), (1080, 1080)], default=(256, 256)),
        gr.inputs.Number(label="Minimum frames per scene", default=16),
        gr.inputs.Number(label="Maximum frames per scene", default=16),
        gr.inputs.Textbox(label="Output directory", default="./output")
    ],
    outputs=gr.outputs.Textbox(),
    title="Video Trimmer",
    description="Extract scenes from a video by detecting content changes and trimming them down to the specified number of frames.",
    theme="compact"
)

if __name__ == "__main__":
    iface.launch()