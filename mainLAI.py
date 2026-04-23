

import asyncio
import shutil
from pathlib import Path
from typing import Optional, List
import tempfile

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from comfy_client import comfy
from config import settings
from frame_extractor import extract_last_frame
from prompt_enhancer import enhance_image_prompt, enhance_prompt
from storage import StorageManager
from video_processing import assemble_scene, concatenate_scenes, edit_video, add_transition
from workflows import build_image_workflow, build_video_workflow

# ── App 
app = FastAPI(title="CineForge — AI Video Editor & Generator", version="4.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

storage = StorageManager(settings.OUTPUT_DIR)

# Static mounts
app.mount("/files", StaticFiles(directory=str(settings.OUTPUT_DIR)), name="files")
app.mount("/ui",    StaticFiles(directory="frontend", html=True),    name="ui")


# ── Pydantic models 
class CreateProjectReq(BaseModel):
    title: str = "Untitled Project"

class SceneReq(BaseModel):
    prompt: str

class ConfirmReq(BaseModel):
    confirmed: bool   

class EditClipReq(BaseModel):
    clip_index: int
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    action: str 
    split_position: Optional[float] = None

class ReorderClipsReq(BaseModel):
    new_order: List[int]

class AddTransitionReq(BaseModel):
    clip_index_a: int
    clip_index_b: int
    transition_type: str = "fade" 
    duration: float = 0.5

class ExportSceneReq(BaseModel):
    quality: str = "high"  # high, medium, low


# ── Root 
@app.get("/")
async def root():
    return {"status": "ok", "ui": "/ui", "docs": "/docs", "version": "4.0.0"}


# ── Projects 
@app.post("/projects", status_code=201)
async def create_project(req: CreateProjectReq):
    return storage.create_project(title=req.title)


@app.get("/projects")
async def list_projects():
    return storage.list_projects()


@app.get("/projects/{pid}")
async def get_project(pid: str):
    p = storage.get_project(pid)
    if not p:
        raise HTTPException(404, "Project not found")
    return _hydrate_project(p)


def _hydrate_project(p: dict) -> dict:
    """Attach URL fields to all scenes inside a project dict."""
    p = dict(p)
    p["scenes"] = [_hydrate_scene(p["project_id"], s) for s in p.get("scenes", [])]
    return p


# ── Scene: create (AI Generation) 
@app.post("/projects/{pid}/scenes", status_code=201)
async def create_scene(pid: str, req: SceneReq, bg: BackgroundTasks):
    """
    Step 1 of the pipeline.
    Enhances the raw prompt via Groq, then kicks off txt2img in the background.
    Returns the scene stub immediately — poll GET /…/scenes/{idx} for updates.
    """
    if not storage.get_project(pid):
        raise HTTPException(404, "Project not found")

    try:
        clip_prompts = await enhance_prompt(req.prompt)
        img_prompt   = await enhance_image_prompt(req.prompt)
    except Exception as e:
        raise HTTPException(502, f"Prompt enhancement failed: {e}")

    scene     = storage.add_scene(pid, req.prompt, clip_prompts)
    scene_idx = scene["scene_idx"]
    storage.update_scene(pid, scene_idx,
                         enhanced_image_prompt=img_prompt,
                         status="generating_image")

    bg.add_task(_gen_source_image, pid, scene_idx, img_prompt)
    return _hydrate_scene(pid, storage.get_scene(pid, scene_idx))


# ── Scene: upload existing video (New Editing Feature) 
@app.post("/projects/{pid}/scenes/{scene_idx}/upload")
async def upload_video_to_scene(
    pid: str, 
    scene_idx: int, 
    video: UploadFile = File(...),
    position: int = Form(-1) 
):
    """
    Upload an existing video to use as a clip in this scene.
    """
    scene = storage.get_scene(pid, scene_idx)
    if not scene:
        raise HTTPException(404, "Scene not found")
    
    # Save uploaded video
    clip_path = storage.clip_path(pid, scene_idx, len(scene.get("clips", [])))
    with open(clip_path, "wb") as f:
        shutil.copyfileobj(video.file, f)
    
    storage.add_clip(pid, scene_idx, str(clip_path))
    
    # Update scene status if needed
    if scene["status"] == "pending":
        storage.update_scene(pid, scene_idx, status="editing")
    
    return {"status": "uploaded", "clip_index": len(scene.get("clips", [])) - 1}


# ── Scene: edit timeline (New Editing Feature) ────────────────────────────────
@app.post("/projects/{pid}/scenes/{scene_idx}/edit")
async def edit_scene_timeline(pid: str, scene_idx: int, req: EditClipReq, bg: BackgroundTasks):
    """
    Edit a clip in the timeline: trim, cut, split.
    """
    scene = storage.get_scene(pid, scene_idx)
    if not scene:
        raise HTTPException(404, "Scene not found")
    
    clip_path = storage.clip_path(pid, scene_idx, req.clip_index)
    if not clip_path.exists():
        raise HTTPException(404, "Clip not found")
    
    # Perform editing in background to avoid blocking
    bg.add_task(_edit_clip, pid, scene_idx, req, clip_path)
    
    return {"status": "editing", "action": req.action}


@app.post("/projects/{pid}/scenes/{scene_idx}/reorder")
async def reorder_clips(pid: str, scene_idx: int, req: ReorderClipsReq):
    """
    Reorder clips in the timeline.
    """
    scene = storage.get_scene(pid, scene_idx)
    if not scene:
        raise HTTPException(404, "Scene not found")
    
    clips = scene.get("clips", [])
    if len(clips) != len(req.new_order):
        raise HTTPException(400, "New order length must match number of clips")
    
    reordered = [clips[i] for i in req.new_order]
    storage.update_scene(pid, scene_idx, clips=reordered)
    
    return {"status": "reordered", "new_order": req.new_order}


@app.post("/projects/{pid}/scenes/{scene_idx}/transitions")
async def add_clip_transition(pid: str, scene_idx: int, req: AddTransitionReq, bg: BackgroundTasks):
    """
    Add a transition between two clips.
    """
    scene = storage.get_scene(pid, scene_idx)
    if not scene:
        raise HTTPException(404, "Scene not found")
    
    clip_a = storage.clip_path(pid, scene_idx, req.clip_index_a)
    clip_b = storage.clip_path(pid, scene_idx, req.clip_index_b)
    
    if not clip_a.exists() or not clip_b.exists():
        raise HTTPException(404, "One or both clips not found")
    
    # Create transitioned clip
    output_path = storage.transition_path(pid, scene_idx, req.clip_index_a, req.clip_index_b)
    bg.add_task(_add_transition, clip_a, clip_b, output_path, req.transition_type, req.duration)
    
    return {"status": "adding_transition", "output": str(output_path)}


# ── Scene: read 
@app.get("/projects/{pid}/scenes/{scene_idx}")
async def get_scene(pid: str, scene_idx: int):
    s = storage.get_scene(pid, scene_idx)
    if not s:
        raise HTTPException(404, "Scene not found")
    return _hydrate_scene(pid, s)


def _hydrate_scene(pid: str, s: dict) -> dict:
    """Attach *_url fields so the frontend can render images/videos directly."""
    s = dict(s)
    if s.get("source_image"):
        s["source_image_url"] = storage.rel_url(s["source_image"])
    s["clip_urls"] = [storage.rel_url(c) for c in s.get("clips", [])]
    if s.get("scene_video"):
        s["scene_video_url"] = storage.rel_url(s["scene_video"])
    return s


# ── Scene: confirm / reject source image 
@app.post("/projects/{pid}/scenes/{scene_idx}/confirm")
async def confirm_image(pid: str, scene_idx: int, req: ConfirmReq, bg: BackgroundTasks):
    scene = storage.get_scene(pid, scene_idx)
    if not scene:
        raise HTTPException(404, "Scene not found")
    if scene["status"] != "confirming":
        raise HTTPException(400, f"Expected status 'confirming', got '{scene['status']}'")

    if not req.confirmed:
        img_prompt = scene.get("enhanced_image_prompt") or scene["raw_prompt"]
        storage.update_scene(pid, scene_idx, status="generating_image", error=None)
        bg.add_task(_gen_source_image, pid, scene_idx, img_prompt)
        return {"status": "generating_image", "message": "Regenerating source image…"}

    storage.update_scene(pid, scene_idx, status="generating_clips", error=None)
    bg.add_task(_gen_clips, pid, scene_idx)
    return {"status": "generating_clips", "message": "Generating 5 clips…"}


# ── Scene: regenerate with new prompt
@app.post("/projects/{pid}/scenes/{scene_idx}/regenerate")
async def regenerate_image(pid: str, scene_idx: int, req: SceneReq, bg: BackgroundTasks):
    scene = storage.get_scene(pid, scene_idx)
    if not scene:
        raise HTTPException(404, "Scene not found")

    try:
        img_prompt = await enhance_image_prompt(req.prompt)
    except Exception as e:
        raise HTTPException(502, f"Prompt enhancement failed: {e}")

    storage.update_scene(pid, scene_idx,
                         raw_prompt=req.prompt,
                         enhanced_image_prompt=img_prompt,
                         status="generating_image",
                         error=None)
    bg.add_task(_gen_source_image, pid, scene_idx, img_prompt)
    return {"status": "generating_image"}


# ── Scene: export (New Feature) 
@app.post("/projects/{pid}/scenes/{scene_idx}/export")
async def export_scene(
    pid: str, 
    scene_idx: int, 
    req: ExportSceneReq,
    bg: BackgroundTasks
):
    """
    Export the edited scene with specified quality.
    """
    scene = storage.get_scene(pid, scene_idx)
    if not scene:
        raise HTTPException(404, "Scene not found")
    
    clips = scene.get("clips", [])
    if not clips:
        raise HTTPException(400, "No clips to export")
    
    clip_paths = [Path(c) for c in clips]
    output_path = storage.scene_video_path(pid, scene_idx)
    
    bg.add_task(_export_scene, clip_paths, output_path, req.quality)
    
    return {"status": "exporting", "output_url": storage.rel_url(output_path)}


# ── Finalize 
@app.post("/projects/{pid}/finalize")
async def finalize_project(pid: str, bg: BackgroundTasks):
    project = storage.get_project(pid)
    if not project:
        raise HTTPException(404, "Project not found")

    done = [s for s in project["scenes"]
            if s["status"] == "done" and s.get("scene_video")]
    if not done:
        raise HTTPException(400, "No completed scenes to finalize")

    storage.update_project(pid, final_status="processing")
    bg.add_task(_finalize, pid, done)
    return {"status": "processing", "scene_count": len(done)}


@app.get("/projects/{pid}/final-video")
async def download_final(pid: str):
    p = storage.get_project(pid)
    if not p:
        raise HTTPException(404, "Project not found")
    if not p.get("final_video"):
        raise HTTPException(404, "Final video not ready yet")
    return FileResponse(
        p["final_video"],
        media_type="video/mp4",
        filename=f"{p['title'].replace(' ', '_')}_final.mp4",
    )


# Background tasks 

async def _gen_source_image(pid: str, scene_idx: int, image_prompt: str):
    """
    Runs the txt2img workflow on ComfyUI.
    On success → status becomes 'confirming' so the UI can show the image.
    """
    try:
        workflow  = build_image_workflow(image_prompt)
        dest_path = storage.source_image_path(pid, scene_idx)
        await comfy.run_job(workflow, dest_path, timeout=1000)
        storage.update_scene(pid, scene_idx,
                             source_image=str(dest_path),
                             status="confirming")
    except Exception as e:
        storage.update_scene(pid, scene_idx, status="error", error=str(e))


async def _gen_clips(pid: str, scene_idx: int):
    """
    Generates 5 sequential 4-second clips for one scene.

    Clip chain:
      clip_0 ← source_image         + enhanced_prompts[0]
      clip_1 ← last_frame(clip_0)   + enhanced_prompts[1]
      clip_2 ← last_frame(clip_1)   + enhanced_prompts[2]
      clip_3 ← last_frame(clip_2)   + enhanced_prompts[3]
      clip_4 ← last_frame(clip_3)   + enhanced_prompts[4]

    FIX: extract_last_frame and assemble_scene use subprocess.run (blocking).
    Both are wrapped in asyncio.to_thread() to avoid freezing the event loop.
    """
    try:
        scene      = storage.get_scene(pid, scene_idx)
        prompts    = scene["enhanced_prompts"]
        seed_image = Path(scene["source_image"])

        for i in range(settings.CLIPS_PER_SCENE):
            clip_path = storage.clip_path(pid, scene_idx, i)
            workflow  = build_video_workflow(prompts[i])

            await comfy.run_job(
                workflow,
                dest_path=clip_path,
                timeout=10000,
                img_to_upload=seed_image,
            )

            storage.add_clip(pid, scene_idx, str(clip_path))
            storage.update_scene(pid, scene_idx, last_completed_clip=i)

            # Prepare seed for next clip — blocking ffmpeg off the event loop
            if i < settings.CLIPS_PER_SCENE - 1:
                last_frame = storage.last_frame_path(pid, scene_idx)
                await asyncio.to_thread(extract_last_frame, clip_path, last_frame)
                seed_image = last_frame

        # Assemble 5 clips → 20s scene.mp4 — blocking ffmpeg off the event loop
        clip_paths  = [storage.clip_path(pid, scene_idx, i)
                       for i in range(settings.CLIPS_PER_SCENE)]
        scene_video = storage.scene_video_path(pid, scene_idx)
        await asyncio.to_thread(assemble_scene, clip_paths, scene_video)

        storage.update_scene(pid, scene_idx,
                             scene_video=str(scene_video),
                             status="done")
    except Exception as e:
        storage.update_scene(pid, scene_idx, status="error", error=str(e))


async def _edit_clip(pid: str, scene_idx: int, req: EditClipReq, clip_path: Path):
    """
    Edit a clip: trim, cut, split.
    """
    try:
        output_path = storage.clip_path(pid, scene_idx, req.clip_index)
        output_path = output_path.with_stem(f"{output_path.stem}_edited")
        
        await asyncio.to_thread(
            edit_video, 
            clip_path, 
            output_path, 
            req.start_time, 
            req.end_time,
            req.action,
            req.split_position
        )
        
        # Replace original with edited version
        output_path.rename(clip_path)
        
        storage.update_scene(pid, scene_idx, last_edit=f"{req.action}_clip_{req.clip_index}")
    except Exception as e:
        storage.update_scene(pid, scene_idx, error=str(e))


async def _add_transition(clip_a: Path, clip_b: Path, output_path: Path, transition_type: str, duration: float):
    """
    Add transition between two clips.
    """
    try:
        await asyncio.to_thread(add_transition, clip_a, clip_b, output_path, transition_type, duration)
    except Exception as e:
        print(f"Transition failed: {e}")


async def _export_scene(clip_paths: List[Path], output_path: Path, quality: str):
    """
    Export scene with specified quality.
    """
    try:
        await asyncio.to_thread(assemble_scene, clip_paths, output_path)
    except Exception as e:
        print(f"Export failed: {e}")


async def _finalize(pid: str, done_scenes: list):
    """
    Concatenates all done scene videos with lightning-flash transitions.

    FIX: concatenate_scenes uses subprocess.run (blocking).
    Wrapped in asyncio.to_thread() to avoid freezing the event loop.
    """
    try:
        scene_paths = [Path(s["scene_video"]) for s in done_scenes]
        out_path    = storage.final_video_path(pid)
        await asyncio.to_thread(concatenate_scenes, scene_paths, out_path)

        storage.update_project(pid,
                               final_video=str(out_path),
                               final_video_url=storage.rel_url(out_path),
                               final_status="done")
    except Exception as e:
        storage.update_project(pid, final_status="error", final_error=str(e))
