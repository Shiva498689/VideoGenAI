import json
import subprocess
import shutil
import tempfile
from pathlib import Path


def _get_duration(video_path: Path) -> float:
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

def assemble_scene(clip_paths: list[Path], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        list_file = Path(f.name)
        for cp in clip_paths:
            f.write(f"file '{cp.resolve()}'\n")

    cmd_copy = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(list_file),
        "-c", "copy",          
        str(output_path),
    ]
    r = subprocess.run(cmd_copy, capture_output=True, text=True)

    if r.returncode == 0:
        list_file.unlink(missing_ok=True)
        return output_path

    print(f"  Stream-copy failed, re-encoding: {r.stderr[-300:]}")
    cmd_reencode = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(list_file),
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2",
        str(output_path),
    ]
    r2 = subprocess.run(cmd_reencode, capture_output=True, text=True)
    if r2.returncode != 0:
        list_file.unlink(missing_ok=True)
        raise RuntimeError(f"assemble_scene failed:\n{r2.stderr}")

    list_file.unlink(missing_ok=True)
    return output_path

def _lightning_filter(
    n: int,
    durations: list[float],
    audio_flags: list[bool],
    fps: int = 20,
    flash_dur: float = 0.4,
    sample_rate: int = 44100,
) -> tuple[str, str, str]:
    parts      = []
    v_labels   = []   
    a_labels   = []   

    for i in range(n):
        dur = durations[i]
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
        a_lbl = f"norm_a{i}"
        if audio_flags[i]:
            parts.append(
                f"[{i}:a]"
                f"aresample={sample_rate},"
                f"pan=stereo|c0=c0|c1=c1,"  
                f"aformat=sample_fmts=fltp:channel_layouts=stereo"
                f"[{a_lbl}]"
            )
        else:
            parts.append(
                f"aevalsrc=0:channel_layout=stereo:sample_rate={sample_rate}"
                f":duration={dur:.4f}"
                f"[{a_lbl}]"
            )
        a_labels.append(a_lbl)
      
    current_v          = v_labels[0]
    current_a          = a_labels[0]
    accumulated_offset = 0.0

    for i in range(1, n):
        accumulated_offset += durations[i - 1] - flash_dur

        v_out = f"xf_v{i}"
        a_out = f"xf_a{i}"
        parts.append(
            f"[{current_v}][{v_labels[i]}]"
            f"xfade=transition=fadewhite"
            f":duration={flash_dur}"
            f":offset={accumulated_offset:.4f}"
            f"[{v_out}]"
        )
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
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if len(scene_paths) == 1:
        shutil.copy2(scene_paths[0], output_path)
        return output_path

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
        "-map", final_v,         
        "-map", final_a,         
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
            print(f"Final video duration: {actual:.2f}s  (expected ~{expected_total:.2f}s)")
        except Exception:
            pass

    return output_path

def _plain_concat(scene_paths: list[Path], output_path: Path) -> None:
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
