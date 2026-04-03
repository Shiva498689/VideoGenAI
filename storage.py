"""
StorageManager — filesystem layout + JSON index

outputs/
├── index.json                          ← master metadata store
└── projects/
    └── {project_id}/
        ├── scenes/
        │   └── {scene_idx}/
        │       ├── source_image.png    ← txt2img keyframe (confirmed by user)
        │       ├── last_frame.png      ← last frame of previous clip (seed)
        │       ├── clips/
        │       │   ├── clip_0.mp4
        │       │   ├── clip_1.mp4
        │       │   ├── clip_2.mp4
        │       │   ├── clip_3.mp4
        │       │   └── clip_4.mp4
        │       ├── transitions/        ← NEW: stores transitioned clips
        │       │   ├── transition_0_1.mp4
        │       │   └── ...
        │       └── scene.mp4           ← 5 clips assembled → 20s
        └── final.mp4                   ← all scenes with lightning transitions
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from typing import Optional, List


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
            "settings": {
                "fps": 20,
                "transition_duration": 0.4,
                "default_quality": "high"
            }
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
            "transitions":           [],                 # NEW: store transition info
            "last_completed_clip":   -1,
            "scene_video":           None,
            "status": "pending",
            # pending → generating_image → confirming → generating_clips → editing → done | error
            "error":      None,
            "created_at": _now(),
            "timeline": {                               # NEW: timeline state
                "order": [],                            # clip order indices
                "trims": {},                            # trim start/end per clip
                "transitions": []                       # transitions between clips
            }
        }
        project["scenes"].append(scene)
        self.scene_dir(pid, scene_idx).mkdir(parents=True, exist_ok=True)
        (self.scene_dir(pid, scene_idx) / "clips").mkdir(exist_ok=True)
        (self.scene_dir(pid, scene_idx) / "transitions").mkdir(exist_ok=True)  # NEW
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
        scene = self._idx["projects"][pid]["scenes"][scene_idx]
        scene["clips"].append(clip_path)
        # Update timeline order
        if "timeline" not in scene:
            scene["timeline"] = {"order": [], "trims": {}, "transitions": []}
        scene["timeline"]["order"] = list(range(len(scene["clips"])))
        self._flush()

    def update_timeline_order(self, pid: str, scene_idx: int, new_order: List[int]):
        scene = self._idx["projects"][pid]["scenes"][scene_idx]
        if "timeline" not in scene:
            scene["timeline"] = {"order": [], "trims": {}, "transitions": []}
        scene["timeline"]["order"] = new_order
        self._flush()

    def update_clip_trim(self, pid: str, scene_idx: int, clip_idx: int, start: float, end: float):
        scene = self._idx["projects"][pid]["scenes"][scene_idx]
        if "timeline" not in scene:
            scene["timeline"] = {"order": [], "trims": {}, "transitions": []}
        scene["timeline"]["trims"][str(clip_idx)] = {"start": start, "end": end}
        self._flush()

    def add_transition_info(self, pid: str, scene_idx: int, clip_a: int, clip_b: int, trans_type: str, duration: float):
        scene = self._idx["projects"][pid]["scenes"][scene_idx]
        if "timeline" not in scene:
            scene["timeline"] = {"order": [], "trims": {}, "transitions": []}
        scene["timeline"]["transitions"].append({
            "clip_a": clip_a,
            "clip_b": clip_b,
            "type": trans_type,
            "duration": duration
        })
        self._flush()

    # ── Paths ─────────────────────────────────────────────────────────────────

    def project_dir(self, pid: str) -> Path:
        return self.root / "projects" / pid

    def scene_dir(self, pid: str, scene_idx: int) -> Path:
        return self.project_dir(pid) / "scenes" / str(scene_idx)

    def source_image_path(self, pid: str, scene_idx: int) -> Path:
        return self.scene_dir(pid, scene_idx) / "source_image.png"

    def last_frame_path(self, pid: str, scene_idx: int) -> Path:
        return self.scene_dir(pid, scene_idx) / "last_frame.png"

    def clip_path(self, pid: str, scene_idx: int, clip_idx: int) -> Path:
        return self.scene_dir(pid, scene_idx) / "clips" / f"clip_{clip_idx}.mp4"

    def transition_path(self, pid: str, scene_idx: int, clip_a: int, clip_b: int) -> Path:
        return self.scene_dir(pid, scene_idx) / "transitions" / f"transition_{clip_a}_{clip_b}.mp4"

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
