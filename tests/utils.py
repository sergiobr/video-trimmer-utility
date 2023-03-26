import argparse
import os
import subprocess

def create_dummy_mp4(filename='dummy', dir='.'):
    os.makedirs(dir, exist_ok=True)
    filepath = os.path.join(dir, f"{filename}.mp4")
    
    cmd = [
        "ffmpeg",
        "-y",  # overwrite output file if it already exists
        "-f", "lavfi",
        "-i", "color=white:s=1920x1080:r=30",
        "-t", "10",
        "-c:v", "libx264",
        "-b:v", "1M",
        "-pix_fmt", "yuv420p",
        "-preset", "veryslow",
        filepath
    ]
    subprocess.run(cmd, check=True)

    return filepath

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, default='dummy',
                        help='Name of the output file')
    parser.add_argument('--dir', type=str, default='.',
                        help='Directory to save the output file')
    args = parser.parse_args()

    create_dummy_mp4(filename=args.filename, dir=args.dir)
