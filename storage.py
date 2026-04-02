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

    def create_project(self, title: str = "Untitled") -> dict:
        pid     = str(uuid4())
        project = {
            "project_id":   pid,
            "title":        title,
            "created_at":   _now(),
            "scenes":       [],
            "final_video":  None,
            "final_status": "idle",    
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
            "enhanced_prompts":      enhanced_prompts,
            "enhanced_image_prompt": None,
            "source_image":          None,
            "clips":                 [],
            "last_completed_clip":   -1,
            "scene_video":           None,
            "status": "pending",
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
        self._idx["projects"][pid]["scenes"][scene_idx]["clips"].append(clip_path)
        self._flush()


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

    def scene_video_path(self, pid: str, scene_idx: int) -> Path:
        return self.scene_dir(pid, scene_idx) / "scene.mp4"

    def final_video_path(self, pid: str) -> Path:
        return self.project_dir(pid) / "final.mp4"


    def rel_url(self, abs_path: str | Path) -> str:
        try:
            rel = Path(abs_path).relative_to(self.root)
            return f"/files/{rel}"
        except ValueError:
            return ""

    def _flush(self):
        with open(self.index_path, "w") as f:
            json.dump(self._idx, f, indent=2)
