"""
Video processing — two stages:

  1. assemble_scene(clips, output)
     Concatenates the 5 × 4s clips for one scene into a 20s scene.mp4.
     Uses stream-copy first (fast, lossless); re-encodes as fallback.

  2. concatenate_scenes(scenes, output)
     Stitches all scene videos into the final film with a lightning-flash
     white-burst transition between each scene, implemented as an ffmpeg
     xfade filter chain.

ROOT CAUSE FIX:
  The original code used `offset=0` in every xfade call, which made ALL
  transitions fire at t=0, causing scenes to collapse and overlap at the
  very beginning — producing only ~24s from 3×20s inputs instead of ~60s.

  The xfade `offset` parameter means: "start the transition this many
  seconds into the COMBINED output timeline so far."  It must be
  accumulated correctly across every transition:

      offset_1 = dur(scene_1) - flash_dur
      offset_2 = dur(scene_1) + dur(scene_2) - 2 × flash_dur
      offset_k = Σ dur(scene_1..k) - k × flash_dur

  Each scene's true duration is measured with ffprobe before building
  the filter, so the maths always uses real values instead of assumptions.
"""

import json
import subprocess
import shutil
import tempfile
from pathlib import Path


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_duration(video_path: Path) -> float:
    """
    Returns the video stream duration (seconds) of video_path using ffprobe.
    Raises RuntimeError if ffprobe fails or no video stream is found.
    """
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        str(video_path),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {video_path}:\n{r.stderr}")

    info = json.loads(r.stdout)
    for stream in info.get("streams", []):
        if stream.get("codec_type") == "video":
            raw = stream.get("duration") or stream.get("nb_frames")
            if raw:
                return float(raw)

    # Fallback: read container duration
    cmd2 = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        str(video_path),
    ]
    r2 = subprocess.run(cmd2, capture_output=True, text=True)
    if r2.returncode == 0:
        fmt = json.loads(r2.stdout).get("format", {})
        if "duration" in fmt:
            return float(fmt["duration"])

    raise ValueError(f"Could not determine duration of {video_path}")


# ── 1. Assemble 5 clips → 20s scene ──────────────────────────────────────────

def assemble_scene(clip_paths: list[Path], output_path: Path) -> Path:
    """
    Concatenates clip_paths in order into output_path.
    Tries stream-copy first (fast); falls back to H.264 re-encode.
    """
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

def _lightning_filter(
    n: int,
    durations: list[float],
    fps: int = 20,
    flash_dur: float = 0.4,
) -> tuple[str, str]:
    """
    Builds an ffmpeg filter_complex string that:
      - Normalises every scene to 1920×1080 @ fps
      - Chains xfade=fadewhite transitions between consecutive scenes

    CRITICAL: offset for each xfade is accumulated correctly so total
    output duration equals the sum of all scene durations minus the
    overlap introduced by each transition.

    offset formula:
      offset_i = Σ(dur[0..i-1]) - i × flash_dur
                 (sum of all PREVIOUS scene durations minus transitions so far)

    Example — 3 scenes × 20s, flash_dur=0.4s:
      offset_1 = 20 - 0.4            = 19.6 s
      offset_2 = 20 + 20 - 2×0.4    = 39.2 s
      total output ≈ 39.2 + 20       = 59.2 s  ✓

    Returns (filter_complex_string, final_output_label).
    """
    parts  = []
    labels = []

    # Step 1 — normalise every input to the same resolution / fps / pix_fmt
    for i in range(n):
        lbl = f"norm{i}"
        parts.append(
            f"[{i}:v]"
            f"scale=1920:1080:force_original_aspect_ratio=decrease,"
            f"pad=1920:1080:(ow-iw)/2:(oh-ih)/2,"
            f"fps={fps},"
            f"format=yuv420p"
            f"[{lbl}]"
        )
        labels.append(lbl)

    # Step 2 — chain xfade transitions with CORRECT cumulative offsets
    current           = labels[0]
    accumulated_offset = 0.0          # tracks where we are in the output timeline

    for i in range(1, n):
        # Move offset forward by the PREVIOUS scene's duration minus one flash
        accumulated_offset += durations[i - 1] - flash_dur

        out_label = f"xf{i}"
        xfade = (
            f"[{current}][{labels[i]}]"
            f"xfade=transition=fadewhite"
            f":duration={flash_dur}"
            f":offset={accumulated_offset:.4f}"          # ← THE FIX
            f"[{out_label}]"
        )
        parts.append(xfade)
        current = out_label

    return ";\n".join(parts), f"[{current}]"


def concatenate_scenes(
    scene_paths: list[Path],
    output_path: Path,
    fps: int = 20,
) -> Path:
    """
    Concatenate all scene videos with lightning flash transitions between them.

    • Single scene  → plain file copy (no re-encode).
    • Multi-scene   → xfade filter chain with correct offsets.
    • On filter failure → falls back to plain concat (no transitions).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if len(scene_paths) == 1:
        shutil.copy2(scene_paths[0], output_path)
        return output_path

    # Probe every scene's real duration before building the filter
    durations = []
    for sp in scene_paths:
        dur = _get_duration(sp)
        durations.append(dur)
        print(f"  Scene duration: {sp.name} = {dur:.2f}s")

    expected_total = sum(durations) - (len(scene_paths) - 1) * 0.4
    print(f"  Expected output duration: {expected_total:.2f}s "
          f"({len(scene_paths)} scenes × ~{sum(durations)/len(scene_paths):.1f}s "
          f"with {len(scene_paths)-1} × 0.4s flash)")

    input_args = []
    for sp in scene_paths:
        input_args += ["-i", str(sp)]

    filter_complex, final_label = _lightning_filter(
        len(scene_paths), durations, fps
    )

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
        print(f"Lightning filter failed, falling back to plain concat:\n{r.stderr}")
        _plain_concat(scene_paths, output_path)
    else:
        # Verify actual output duration
        try:
            actual = _get_duration(output_path)
            print(f"  ✅ Final video duration: {actual:.2f}s")
        except Exception:
            pass

    return output_path


def _plain_concat(scene_paths: list[Path], output_path: Path) -> None:
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
