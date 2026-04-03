"""
Video processing — two stages:

  1. assemble_scene(clips, output)
     Concatenates the 5 × 4s clips for one scene into a 20s scene.mp4.
     Uses stream-copy first (fast, lossless); re-encodes as fallback.
     Audio is preserved in BOTH paths.

  2. concatenate_scenes(scenes, output)
     Stitches all scene videos into the final film with a lightning-flash
     white-burst transition between each scene, implemented as an ffmpeg
     xfade + acrossfade filter chain.

  3. edit_video() - NEW
     Trim, cut, split videos for timeline editing.

  4. add_transition() - NEW
     Add transitions between clips.

AUDIO HANDLING:
  - assemble_scene() fast path (-c copy): audio is preserved as-is.
  - assemble_scene() fallback (re-encode): audio is re-encoded to AAC.
  - concatenate_scenes(): each scene's audio is normalised to 44100 Hz
    stereo, then joined with acrossfade matching the video xfade duration.
  - If a clip has NO audio stream, a silent track is synthesised with
    aevalsrc=0 so the filter chain never breaks.
  - _plain_concat() fallback also preserves audio.

XFADE OFFSET FIX:
  offset_i = Σ dur[0..i-1] - i × flash_dur
  All offsets are accumulated from real ffprobe durations, not guesses.
"""

import json
import subprocess
import shutil
import tempfile
from pathlib import Path
from typing import Optional, List


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_duration(video_path: Path) -> float:
    """
    Returns the video stream duration (seconds) using ffprobe.
    Falls back to container duration if stream duration is absent.
    Raises RuntimeError / ValueError if nothing works.
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

    # Fallback: container-level duration
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


def _has_audio(video_path: Path) -> bool:
    """
    Returns True if video_path contains at least one audio stream.
    """
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        str(video_path),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return False
    info = json.loads(r.stdout)
    return any(
        s.get("codec_type") == "audio"
        for s in info.get("streams", [])
    )


# ── NEW: Video Editing Functions ─────────────────────────────────────────────

def edit_video(
    input_path: Path,
    output_path: Path,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    action: str = "trim",
    split_position: Optional[float] = None
) -> Path:
    """
    Edit a video clip:
    - trim: cut from start_time to end_time
    - cut: remove segment between start_time and end_time
    - split: split at split_position (returns first part, second part saved as split_2)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if action == "trim":
        if start_time is None:
            start_time = 0
        if end_time is None:
            end_time = _get_duration(input_path)
        
        duration = end_time - start_time
        
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_time),
            "-i", str(input_path),
            "-t", str(duration),
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-c:a", "aac", "-b:a", "192k",
            str(output_path)
        ]
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        
    elif action == "cut":
        if start_time is None or end_time is None:
            raise ValueError("cut requires both start_time and end_time")
        
        # Get total duration
        total_dur = _get_duration(input_path)
        
        # First part: 0 to start_time
        first_part = output_path.with_stem(f"{output_path.stem}_part1")
        cmd1 = [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-t", str(start_time),
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-c:a", "aac", "-b:a", "192k",
            str(first_part)
        ]
        subprocess.run(cmd1, capture_output=True, text=True, check=True)
        
        # Second part: end_time to end
        second_part = output_path.with_stem(f"{output_path.stem}_part2")
        cmd2 = [
            "ffmpeg", "-y",
            "-ss", str(end_time),
            "-i", str(input_path),
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-c:a", "aac", "-b:a", "192k",
            str(second_part)
        ]
        subprocess.run(cmd2, capture_output=True, text=True, check=True)
        
        # Concatenate the two parts
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            list_file = Path(f.name)
            f.write(f"file '{first_part.resolve()}'\n")
            f.write(f"file '{second_part.resolve()}'\n")
        
        cmd_concat = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(list_file),
            "-c", "copy",
            str(output_path)
        ]
        subprocess.run(cmd_concat, capture_output=True, text=True, check=True)
        
        # Clean up
        first_part.unlink(missing_ok=True)
        second_part.unlink(missing_ok=True)
        list_file.unlink(missing_ok=True)
        
    elif action == "split":
        if split_position is None:
            raise ValueError("split requires split_position")
        
        total_dur = _get_duration(input_path)
        
        # First part
        first_part = output_path
        cmd1 = [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-t", str(split_position),
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-c:a", "aac", "-b:a", "192k",
            str(first_part)
        ]
        subprocess.run(cmd1, capture_output=True, text=True, check=True)
        
        # Second part
        second_part = output_path.parent / f"split_{output_path.name}"
        cmd2 = [
            "ffmpeg", "-y",
            "-ss", str(split_position),
            "-i", str(input_path),
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-c:a", "aac", "-b:a", "192k",
            str(second_part)
        ]
        subprocess.run(cmd2, capture_output=True, text=True, check=True)
        
        return second_part
    
    return output_path


def add_transition(
    clip_a: Path,
    clip_b: Path,
    output_path: Path,
    transition_type: str = "fade",
    duration: float = 0.5
) -> Path:
    """
    Add transition between two clips.
    Supported transitions: fade, crossfade, wipe, slide
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Map transition types to ffmpeg xfade types
    transition_map = {
        "fade": "fade",
        "crossfade": "fade",  # xfade with fade
        "wipe": "wipeleft",
        "slide": "slideleft",
    }
    
    xfade_type = transition_map.get(transition_type, "fade")
    
    cmd = [
        "ffmpeg", "-y",
        "-i", str(clip_a),
        "-i", str(clip_b),
        "-filter_complex", 
        f"[0:v][1:v]xfade=transition={xfade_type}:duration={duration}:offset={_get_duration(clip_a) - duration}[v];"
        f"[0:a][1:a]acrossfade=d={duration}[a]",
        "-map", "[v]",
        "-map", "[a]",
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-c:a", "aac", "-b:a", "192k",
        str(output_path)
    ]
    
    subprocess.run(cmd, capture_output=True, text=True, check=True)
    return output_path


# ── 1. Assemble 5 clips → 20s scene ──────────────────────────────────────────

def assemble_scene(clip_paths: list[Path], output_path: Path) -> Path:
    """
    Concatenates clip_paths in order into output_path.

    Fast path  : stream-copy (-c copy) — zero quality loss, audio kept.
    Fallback   : H.264 + AAC re-encode — fixes codec/resolution mismatches,
                 audio is re-encoded to AAC 192 k instead of being dropped.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        list_file = Path(f.name)
        for cp in clip_paths:
            f.write(f"file '{cp.resolve()}'\n")

    # ── Fast path: stream copy ─────────────────────────────────────────────
    cmd_copy = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(list_file),
        "-c", "copy",           # copy video AND audio streams as-is
        str(output_path),
    ]
    r = subprocess.run(cmd_copy, capture_output=True, text=True)

    if r.returncode == 0:
        list_file.unlink(missing_ok=True)
        return output_path

    # ── Fallback: re-encode video + audio ──────────────────────────────────
    print(f"  Stream-copy failed, re-encoding: {r.stderr[-300:]}")
    cmd_reencode = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(list_file),
        # Video
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        # Audio — re-encode to AAC (was -an before, which silenced the clip)
        "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2",
        str(output_path),
    ]
    r2 = subprocess.run(cmd_reencode, capture_output=True, text=True)
    if r2.returncode != 0:
        list_file.unlink(missing_ok=True)
        raise RuntimeError(f"assemble_scene failed:\n{r2.stderr}")

    list_file.unlink(missing_ok=True)
    return output_path


# ── 2. Concatenate scenes with lightning-flash transition ─────────────────────

def _lightning_filter(
    n: int,
    durations: list[float],
    audio_flags: list[bool],
    fps: int = 20,
    flash_dur: float = 0.4,
    sample_rate: int = 44100,
) -> tuple[str, str, str]:
    """
    Builds an ffmpeg filter_complex string that:

    VIDEO:
      - Normalises every scene to 1920×1080 @ fps / yuv420p
      - Chains xfade=fadewhite transitions with correct cumulative offsets

    AUDIO:
      - Normalises every scene audio to `sample_rate` Hz, stereo
      - If a scene has NO audio stream, synthesises silence via aevalsrc
      - Chains acrossfade transitions matching the video flash duration

    XFADE OFFSET FORMULA (avoids the offset=0 collapse bug):
      offset_i = Σ dur[0..i-1]  -  i × flash_dur

    Example — 3 scenes × 20 s, flash_dur = 0.4 s:
      offset_1 = 20              - 0.4          = 19.6 s
      offset_2 = 20 + 20         - 2 × 0.4      = 39.2 s
      expected total ≈ 59.2 s    ✓

    Returns:
      (filter_complex_string, final_video_label, final_audio_label)
    """
    parts      = []
    v_labels   = []   # normalised video labels
    a_labels   = []   # normalised audio labels

    # ── Step 1: normalise every input ─────────────────────────────────────
    for i in range(n):
        dur = durations[i]

        # Video normalisation
        v_lbl = f"norm_v{i}"
        parts.append(
            f"[{i}:v]"
            f"scale=1920:1080:force_original_aspect_ratio=decrease,"
            f"pad=1920:1080:(ow-iw)/2:(oh-ih)/2,"
            f"fps={fps},"
            f"format=yuv420p"
            f"[{v_lbl}]"
        )
        v_labels.append(v_lbl)

        # Audio normalisation
        # If the scene has no audio, synthesise silence of the same duration.
        a_lbl = f"norm_a{i}"
        if audio_flags[i]:
            parts.append(
                f"[{i}:a]"
                f"aresample={sample_rate},"
                f"pan=stereo|c0=c0|c1=c1,"   # ensure stereo
                f"aformat=sample_fmts=fltp:channel_layouts=stereo"
                f"[{a_lbl}]"
            )
        else:
            # aevalsrc generates a silent signal; atrim limits it to scene dur
            parts.append(
                f"aevalsrc=0:channel_layout=stereo:sample_rate={sample_rate}"
                f":duration={dur:.4f}"
                f"[{a_lbl}]"
            )
        a_labels.append(a_lbl)

    # ── Step 2: chain xfade (video) + acrossfade (audio) ──────────────────
    current_v          = v_labels[0]
    current_a          = a_labels[0]
    accumulated_offset = 0.0   # tracks cumulative output timeline position

    for i in range(1, n):
        # Advance by the PREVIOUS scene's duration minus one flash overlap
        accumulated_offset += durations[i - 1] - flash_dur

        v_out = f"xf_v{i}"
        a_out = f"xf_a{i}"

        # Video transition (fade to white)
        parts.append(
            f"[{current_v}][{v_labels[i]}]"
            f"xfade=transition=fadewhite"
            f":duration={flash_dur}"
            f":offset={accumulated_offset:.4f}"
            f"[{v_out}]"
        )

        # Audio transition (cross-fade matching video duration exactly)
        parts.append(
            f"[{current_a}][{a_labels[i]}]"
            f"acrossfade=d={flash_dur}:c1=tri:c2=tri"
            f"[{a_out}]"
        )

        current_v = v_out
        current_a = a_out

    return ";\n".join(parts), f"[{current_v}]", f"[{current_a}]"


def concatenate_scenes(
    scene_paths: list[Path],
    output_path: Path,
    fps: int = 20,
    flash_dur: float = 0.4,
) -> Path:
    """
    Concatenate all scene videos with lightning flash transitions.

    • Single scene  → plain file copy (no re-encode, audio intact).
    • Multi-scene   → xfade + acrossfade filter chain with correct offsets.
    • On failure    → falls back to plain concat (audio preserved there too).

    Scenes with no audio stream get a synthesised silent track so the
    filter graph never errors with "no audio stream" mismatches.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if len(scene_paths) == 1:
        shutil.copy2(scene_paths[0], output_path)
        return output_path

    # ── Probe every scene before building the filter ───────────────────────
    durations   : list[float] = []
    audio_flags : list[bool]  = []

    for sp in scene_paths:
        dur  = _get_duration(sp)
        has_a = _has_audio(sp)
        durations.append(dur)
        audio_flags.append(has_a)
        print(f"  {sp.name}: {dur:.2f}s  audio={'yes' if has_a else 'NO — silence will be synthesised'}")

    expected_total = sum(durations) - (len(scene_paths) - 1) * flash_dur
    print(
        f"  Expected output: {expected_total:.2f}s "
        f"({len(scene_paths)} scenes, {len(scene_paths)-1} × {flash_dur}s flash)"
    )

    input_args = []
    for sp in scene_paths:
        input_args += ["-i", str(sp)]

    filter_complex, final_v, final_a = _lightning_filter(
        len(scene_paths),
        durations,
        audio_flags,
        fps=fps,
        flash_dur=flash_dur,
    )

    cmd = [
        "ffmpeg", "-y",
        *input_args,
        "-filter_complex", filter_complex,
        "-map", final_v,          # mapped video stream
        "-map", final_a,          # mapped audio stream  ← was missing before
        "-c:v", "libx264", "-crf", "17", "-preset", "fast",
        "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2",
        str(output_path),
    ]

    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    if r.returncode != 0:
        print(f"  Lightning filter failed — falling back to plain concat.\n  {r.stderr[-400:]}")
        _plain_concat(scene_paths, output_path)
    else:
        try:
            actual = _get_duration(output_path)
            print(f"  ✅ Final video duration: {actual:.2f}s  (expected ~{expected_total:.2f}s)")
        except Exception:
            pass

    return output_path


# ── Emergency fallback ────────────────────────────────────────────────────────

def _plain_concat(scene_paths: list[Path], output_path: Path) -> None:
    """
    Plain concat fallback — no transitions, but audio is preserved.
    Re-encodes both video and audio so mixed-format inputs are accepted.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        lf = Path(f.name)
        for sp in scene_paths:
            f.write(f"file '{sp.resolve()}'\n")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(lf),
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2",
        # NOTE: -an removed — audio is preserved
        str(output_path),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    lf.unlink(missing_ok=True)
    if r.returncode != 0:
        raise RuntimeError(f"Plain concat fallback also failed:\n{r.stderr}")
