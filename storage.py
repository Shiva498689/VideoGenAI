"""
StorageManager — filesystem layout + JSON index

outputs/
├── index.json                          ← master metadata store
└── projects/
    └── {project_id}/
        ├── scenes/
        │   └── {scene_idx}/
        │       ├── source_image.png    ← txt2img keyframe (confirmed by user)
        │       ├── last_frame.png      ← last frame of previous clip (seed for clip 0)
        │       ├── clips/
        │       │   ├── clip_0.mp4
        │       │   ├── clip_0_last_frame.png   ← last frame of clip_0 (seed for clip_1)
        │       │   ├── clip_1.mp4
        │       │   ├── clip_1_last_frame.png
        │       │   ├── clip_2.mp4
        │       │   ├── clip_2_last_frame.png
        │       │   ├── clip_3.mp4
        │       │   ├── clip_3_last_frame.png
        │       │   └── clip_4.mp4
        │       └── scene.mp4           ← 5 clips assembled → 20s
        └── final.mp4                   ← all scenes with lightning transitions
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class StorageManager:
    def __init__(self, output_dir: Path):
        self.root       = Path(output_dir)
        self.index_path = self.root / "index.json"
        self.root.mkdir(parents=True, exist_ok=True)

        if self.index_path.exists():
            with open(self.index_path) as f:
                self._idx: dict = json.load(f)
        else:
            self._idx = {"projects": {}}
            self._flush()

    # ── Projects ──────────────────────────────────────────────────────────────

    def create_project(self, title: str = "Untitled") -> dict:
        pid     = str(uuid4())
        project = {
            "project_id":   pid,
            "title":        title,
            "created_at":   _now(),
            "scenes":       [],
            "final_video":  None,
            "final_status": "idle",     # idle | processing | done | error
        }
        self._idx["projects"][pid] = project
        self.project_dir(pid).mkdir(parents=True, exist_ok=True)
        self._flush()
        return project

    def get_project(self, pid: str) -> dict | None:
        return self._idx["projects"].get(pid)

    def list_projects(self) -> list:
        return sorted(
            self._idx["projects"].values(),
            key=lambda p: p["created_at"],
            reverse=True,
        )

    def update_project(self, pid: str, **kwargs):
        self._idx["projects"][pid].update(kwargs)
        self._flush()

    # ── Scenes ────────────────────────────────────────────────────────────────

    def add_scene(
        self,
        pid:               str,
        raw_prompt:        str,
        enhanced_prompts:  list[str],
    ) -> dict:
        project   = self._idx["projects"][pid]
        scene_idx = len(project["scenes"])
        scene = {
            "scene_idx":             scene_idx,
            "raw_prompt":            raw_prompt,
            "enhanced_prompts":      enhanced_prompts,  # 5 clip prompts
            "enhanced_image_prompt": None,               # set separately
            "source_image":          None,
            "clips":                 [],
            # Per-clip metadata: index → {status, prompt_override, error}
            # status: pending | generating | done | error
            "clip_meta":             {},
            "last_completed_clip":   -1,
            "scene_video":           None,
            "status": "pending",
            # pending → generating_image → confirming → generating_clips → done | error
            # NOTE: status stays "done" even when a clip is being regenerated;
            # check clip_meta[clip_idx]["status"] for per-clip progress.
            "error":      None,
            "created_at": _now(),
        }
        project["scenes"].append(scene)
        self.scene_dir(pid, scene_idx).mkdir(parents=True, exist_ok=True)
        (self.scene_dir(pid, scene_idx) / "clips").mkdir(exist_ok=True)
        self._flush()
        return scene

    def get_scene(self, pid: str, scene_idx: int) -> dict | None:
        p = self.get_project(pid)
        if not p or scene_idx >= len(p["scenes"]):
            return None
        return p["scenes"][scene_idx]

    def update_scene(self, pid: str, scene_idx: int, **kwargs):
        self._idx["projects"][pid]["scenes"][scene_idx].update(kwargs)
        self._flush()

    def add_clip(self, pid: str, scene_idx: int, clip_path: str):
        clips = self._idx["projects"][pid]["scenes"][scene_idx]["clips"]
        clip_idx = len(clips)
        # Append or replace at index
        if clip_idx < len(clips):
            clips[clip_idx] = clip_path
        else:
            clips.append(clip_path)
        self._flush()

    def set_clip(self, pid: str, scene_idx: int, clip_idx: int, clip_path: str):
        """Set (or replace) a specific clip by index."""
        clips = self._idx["projects"][pid]["scenes"][scene_idx]["clips"]
        # Extend list if needed
        while len(clips) <= clip_idx:
            clips.append(None)
        clips[clip_idx] = clip_path
        self._flush()

    def update_clip_meta(self, pid: str, scene_idx: int, clip_idx: int, **kwargs):
        """Update per-clip metadata dictionary."""
        scene = self._idx["projects"][pid]["scenes"][scene_idx]
        if "clip_meta" not in scene:
            scene["clip_meta"] = {}
        key = str(clip_idx)
        if key not in scene["clip_meta"]:
            scene["clip_meta"][key] = {}
        scene["clip_meta"][key].update(kwargs)
        self._flush()

    def get_clip_meta(self, pid: str, scene_idx: int, clip_idx: int) -> dict:
        scene = self.get_scene(pid, scene_idx)
        if not scene:
            return {}
        return scene.get("clip_meta", {}).get(str(clip_idx), {})

    # ── Paths ─────────────────────────────────────────────────────────────────

    def project_dir(self, pid: str) -> Path:
        return self.root / "projects" / pid

    def scene_dir(self, pid: str, scene_idx: int) -> Path:
        return self.project_dir(pid) / "scenes" / str(scene_idx)

    def source_image_path(self, pid: str, scene_idx: int) -> Path:
        return self.scene_dir(pid, scene_idx) / "source_image.png"

    def last_frame_path(self, pid: str, scene_idx: int) -> Path:
        """Legacy: last frame of source image used as seed for clip_0."""
        return self.scene_dir(pid, scene_idx) / "last_frame.png"

    def clip_last_frame_path(self, pid: str, scene_idx: int, clip_idx: int) -> Path:
        """
        Stores the last frame of clip_{clip_idx}.png.
        This is the seed image for clip_{clip_idx + 1}.
        Kept separate per clip so that when clip_N is regenerated,
        we can re-extract its last frame and re-chain only the downstream clips.
        """
        return (
            self.scene_dir(pid, scene_idx)
            / "clips"
            / f"clip_{clip_idx}_last_frame.png"
        )

    def clip_path(self, pid: str, scene_idx: int, clip_idx: int) -> Path:
        return self.scene_dir(pid, scene_idx) / "clips" / f"clip_{clip_idx}.mp4"

    def scene_video_path(self, pid: str, scene_idx: int) -> Path:
        return self.scene_dir(pid, scene_idx) / "scene.mp4"

    def final_video_path(self, pid: str) -> Path:
        return self.project_dir(pid) / "final.mp4"

    # ── URL helpers (relative to OUTPUT_DIR, for /files/ mount) ──────────────

    def rel_url(self, abs_path: str | Path) -> str:
        try:
            rel = Path(abs_path).relative_to(self.root)
            return f"/files/{rel}"
        except ValueError:
            return ""

    # ── Internal ──────────────────────────────────────────────────────────────

    def _flush(self):
        with open(self.index_path, "w") as f:
            json.dump(self._idx, f, indent=2)
