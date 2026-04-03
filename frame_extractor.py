import subprocess
from pathlib import Path


def extract_last_frame(video_path: Path, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
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
        cmd2 = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vf", "thumbnail=80",
            "-frames:v", "1",
            str(output_path),
        ]
        r2 = subprocess.run(cmd2, capture_output=True, text=True)
        if r2.returncode != 0 or not output_path.exists():
            raise RuntimeError(
                f"extract_last_frame failed for {video_path}:\n{r2.stderr}"
            )

    return output_path
