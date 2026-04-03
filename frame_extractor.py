"""
Frame extraction using ffmpeg.

extract_last_frame(video, output) → PNG of the very last frame.
This PNG is used as the seed image for the next clip in the chain,
ensuring visual continuity across the 5 clips of each scene.
"""

import subprocess
from pathlib import Path


def extract_last_frame(video_path: Path, output_path: Path) -> Path:
    """
    Extracts the last frame of a video as a PNG file.

    Strategy 1 (fast): seek to 0.1s before end, grab 1 frame.
    Strategy 2 (fallback): decode whole file, use thumbnail filter.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Strategy 1 — fast seek from end
    cmd = [
        "ffmpeg", "-y",
        "-sseof", "-0.1",
        "-i", str(video_path),
        "-vframes", "1",
        "-q:v", "2",
        str(output_path),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)

    if r.returncode != 0 or not output_path.exists() or output_path.stat().st_size == 0:
        # Strategy 2 — full decode fallback
        cmd2 = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vf", "thumbnail=300",   # analyse last 300 frames, pick best
            "-frames:v", "1",
            str(output_path),
        ]
        r2 = subprocess.run(cmd2, capture_output=True, text=True)
        if r2.returncode != 0 or not output_path.exists():
            raise RuntimeError(
                f"extract_last_frame failed for {video_path}:\n{r2.stderr}"
            )

    return output_path
