# [Youtube Music](https://music.youtube.com/) Playlist Downloader with [`yt-dlp`](https://pypi.org/project/yt-dlp/)

###### Install `FFmpeg` and `yt-dlp` on Ubuntu

```sh
sudo add-apt-repository ppa:ubuntuhandbook1/ffmpeg8 -y
sudo apt update -y
sudo apt install ffmpeg -y
pip install -qU "yt-dlp[default]"
```

###### Download the playlist

```python
import yt_dlp
import shutil
from pathlib import Path

yt_music_playlist_url: str = "https://music.youtube.com/playlist?list={playlist_id}" # Enter the playlist URL

dl_path: Path = Path("path/directory/to/download/playlist") # Enter the path to download the playlist

# output_template = "%(playlist_title)s/%(uploader)s - %(title)s.%(ext)s"
output_template = "%(playlist_title)s/%(title)s.%(ext)s"


ydl_opts: dict = {
    "format": "bestaudio",
    "format_sort": ["hasaud", "quality", "asr", "abr"],
    "outtmpl": output_template,
    "postprocessors": [
        {
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "0",
        }
    ],
    "ignoreerrors": True,
    "yes-playlist": True,
    # 'verbose': True,
    "paths": {"home": dl_path.__str__()},
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    pl_info: dict = ydl.extract_info(yt_music_playlist_url)
    pl_title: str = pl_info["title"]

source_dir = dl_path / pl_title
output_filename = source_dir.parent / source_dir.name
archive_path: Path = Path(
    shutil.make_archive(base_name=output_filename, format="zip", root_dir=source_dir)
)

```
