"""
Video processing — two stages:

  1. assemble_scene(clips, output)
     Concatenates the 5 × 4s clips for one scene into a 20s scene.mp4.
     Uses stream-copy first (fast, lossless); re-encodes as fallback.

  2. concatenate_scenes(scenes, output)
     Stitches all scene videos into the final film with a lightning-flash
     white-burst transition between each scene, implemented as an ffmpeg
     xfade + curves filter chain.
"""

import subprocess
import tempfile
from pathlib import Path


# ── 1. Assemble 5 clips → 20s scene ──────────────────────────────────────────

def assemble_scene(clip_paths: list[Path], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        list_file = Path(f.name)
        for cp in clip_paths:
            f.write(f"file '{cp.resolve()}'\n")

    # Fast path — stream copy (no quality loss, no re-encode)
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(list_file),
        "-c", "copy",
        str(output_path),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)

    if r.returncode != 0:
        # Fallback — re-encode to H.264 to fix codec/resolution mismatches
        cmd2 = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(list_file),
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-an",
            str(output_path),
        ]
        r2 = subprocess.run(cmd2, capture_output=True, text=True)
        if r2.returncode != 0:
            raise RuntimeError(f"assemble_scene failed:\n{r2.stderr}")

    list_file.unlink(missing_ok=True)
    return output_path


# ── 2. Concatenate scenes with lightning-flash transition ─────────────────────

def _lightning_filter(n: int, fps: int = 24) -> tuple[str, str]:
    """
    Builds an ffmpeg filter_complex string that:
      - Normalises every scene to 1920×1080 @ fps
      - Applies an xfade between consecutive scenes, then overlays a
        white-burst via the curves filter to simulate a lightning flash

    Returns (filter_complex_string, final_stream_label).
    """
    flash_dur = 0.4   # seconds of white flash at each scene boundary
    parts     = []
    labels    = []

    # Step 1 — normalise each input
    for i in range(n):
        lbl = f"n{i}"
        parts.append(
            f"[{i}:v]"
            f"scale=1920:1080:force_original_aspect_ratio=decrease,"
            f"pad=1920:1080:(ow-iw)/2:(oh-ih)/2,"
            f"fps={fps},"
            f"format=yuv420p"
            f"[{lbl}]"
        )
        labels.append(lbl)

    # Step 2 — chain xfade + lightning curves between consecutive scenes
    current = labels[0]
    for i in range(1, n):
        out = f"t{i}"
        # xfade transition (fade to white)
        xfade = (
            f"[{current}][{labels[i]}]"
            f"xfade=transition=fadewhite:duration={flash_dur}:offset=0"
            f"[xf{i}]"
        )
        # curves: spike to white briefly then return — lightning feel
        # control points: time 0→normal, 0.15→white, 0.35→normal, 1→normal
        curve = (
            f"[xf{i}]"
            f"curves=all='0/0 0.15/1 0.35/0 1/1'"
            f"[{out}]"
        )
        parts.append(xfade)
        parts.append(curve)
        current = out

    return ";\n".join(parts), f"[{current}]"


def concatenate_scenes(scene_paths: list[Path], output_path: Path, fps: int = 24) -> Path:
    """
    Concatenate all scene videos with lightning flash transitions between them.
    Single scene: just copies the file. Multiple scenes: full filter chain.
    Falls back to plain concat if the filter chain fails.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if len(scene_paths) == 1:
        import shutil
        shutil.copy2(scene_paths[0], output_path)
        return output_path

    input_args = []
    for sp in scene_paths:
        input_args += ["-i", str(sp)]

    filter_complex, final_label = _lightning_filter(len(scene_paths), fps)

    cmd = [
        "ffmpeg", "-y",
        *input_args,
        "-filter_complex", filter_complex,
        "-map", final_label,
        "-c:v", "libx264", "-crf", "17", "-preset", "fast",
        "-an",
        str(output_path),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    if r.returncode != 0:
        # Fallback: plain concat, no transition
        _plain_concat(scene_paths, output_path)

    return output_path


def _plain_concat(scene_paths: list[Path], output_path: Path):
    """Emergency fallback — plain concat with no transitions."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        lf = Path(f.name)
        for sp in scene_paths:
            f.write(f"file '{sp.resolve()}'\n")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(lf),
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-an",
        str(output_path),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    lf.unlink(missing_ok=True)
    if r.returncode != 0:
        raise RuntimeError(f"Plain concat fallback also failed:\n{r.stderr}")
