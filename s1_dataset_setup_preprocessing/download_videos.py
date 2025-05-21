import json
import os
from yt_dlp import YoutubeDL

def download_videos(video_ids, output_dir="downloads", max_resolution=1080):
    os.makedirs(output_dir, exist_ok=True)

    # Format string: best progressive video up to max_resolution, mp4 format
    ydl_opts = {
        'format': f'bestvideo[ext=mp4][height<={max_resolution}]+bestaudio[ext=m4a]/best[ext=mp4][height<={max_resolution}]',
        'outtmpl': os.path.join(output_dir, '%(id)s.%(ext)s'),
        'merge_output_format': 'mp4',
        'quiet': False,
        'noplaylist': True,
    }

    with YoutubeDL(ydl_opts) as ydl:
        for video_id in video_ids:
            url = f"https://www.youtube.com/watch?v={video_id}"
            print(f"\nDownloading: {url}")
            try:
                ydl.download([url])
            except Exception as e:
                print(f"Failed to download {video_id}: {e}")


def load_video_ids_from_json(json_path):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        if isinstance(data, dict):
            return list(data.keys())
        else:
            raise ValueError("JSON format is not a dictionary.")
    except Exception as e:
        print(f"Error reading {json_path}: {e}")
        return []


# === MAIN USAGE ===
json_path = "captions_sample.json"  # or use extensions_sample.json
video_ids = load_video_ids_from_json(json_path)
download_videos(video_ids, output_dir="video_downloads", max_resolution=1080)
