"""
CineForge FastAPI Backend
=========================

Full pipeline per scene
────────────────────────
 1. POST /projects/{pid}/scenes
      → Groq enhances raw prompt → 5 clip prompts + 1 image prompt
      → Queues txt2img job (GPU_LOCK serialises with video jobs)
      → Returns immediately; poll for status

 2. GET  /projects/{pid}/scenes/{idx}
      → Poll until status == "confirming"
      → Response includes source_image_url for the UI to display
 
 3. POST /projects/{pid}/scenes/{idx}/confirm  { "confirmed": true }
      → Approved: starts 5-clip generation chain in background
    POST /projects/{pid}/scenes/{idx}/confirm  { "confirmed": false }
      → Rejected: regenerates source image (same prompt)
    POST /projects/{pid}/scenes/{idx}/regenerate  { "prompt": "..." }
      → Rejected with new description: re-enhances + regenerates

 4. Background clip loop (for each of 5 clips):
      upload seed image → img2vid → download clip
      extract last frame → use as seed for next clip
    → On completion: assemble 5 clips → 20s scene.mp4

 5. POST /projects/{pid}/finalize
      → Stitches all done scenes with lightning-flash transition → final.mp4
"""

import asyncio
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from comfy_client import comfy
from config import settings
from frame_extractor import extract_last_frame
from prompt_enhancer import enhance_image_prompt, enhance_prompt
from storage import StorageManager
from video_processing import assemble_scene, concatenate_scenes
from workflows import build_image_workflow, build_video_workflow

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="CineForge — AI Video Pipeline", version="3.0.0")

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


# ── Pydantic models ───────────────────────────────────────────────────────────
class CreateProjectReq(BaseModel):
    title: str = "Untitled Project"

class SceneReq(BaseModel):
    prompt: str

class ConfirmReq(BaseModel):
    confirmed: bool   # True → proceed with clip gen, False → regen image


# ── Root ──────────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {"status": "ok", "ui": "/ui", "docs": "/docs"}


# ── Projects ──────────────────────────────────────────────────────────────────
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


# ── Scene: create ─────────────────────────────────────────────────────────────
@app.post("/projects/{pid}/scenes", status_code=201)
async def create_scene(pid: str, req: SceneReq, bg: BackgroundTasks):
    """
    Step 1 of the pipeline.
    Enhances the raw prompt via Groq, then kicks off txt2img in the background.
    Returns the scene stub immediately — poll GET /…/scenes/{idx} for updates.
    """
    if not storage.get_project(pid):
        raise HTTPException(404, "Project not found")

    # Groq — both calls happen before we create the scene record
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


# ── Scene: read ───────────────────────────────────────────────────────────────
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


# ── Scene: confirm / reject source image ─────────────────────────────────────
@app.post("/projects/{pid}/scenes/{scene_idx}/confirm")
async def confirm_image(pid: str, scene_idx: int, req: ConfirmReq, bg: BackgroundTasks):
    """
    Step 3a.
    confirmed=true  → kick off 5-clip generation chain.
    confirmed=false → regenerate source image with the same enhanced prompt.
    """
    scene = storage.get_scene(pid, scene_idx)
    if not scene:
        raise HTTPException(404, "Scene not found")
    if scene["status"] != "confirming":
        raise HTTPException(400, f"Expected status 'confirming', got '{scene['status']}'")

    if not req.confirmed:
        # Re-generate image (same prompt)
        img_prompt = scene.get("enhanced_image_prompt") or scene["raw_prompt"]
        storage.update_scene(pid, scene_idx, status="generating_image", error=None)
        bg.add_task(_gen_source_image, pid, scene_idx, img_prompt)
        return {"status": "generating_image", "message": "Regenerating source image…"}

    # Start clip generation
    storage.update_scene(pid, scene_idx, status="generating_clips", error=None)
    bg.add_task(_gen_clips, pid, scene_idx)
    return {"status": "generating_clips", "message": "Generating 5 clips…"}


# ── Scene: regenerate with new prompt ─────────────────────────────────────────
@app.post("/projects/{pid}/scenes/{scene_idx}/regenerate")
async def regenerate_image(pid: str, scene_idx: int, req: SceneReq, bg: BackgroundTasks):
    """
    Step 3b — user provides a revised description.
    Re-enhances the prompt and regenerates the source image.
    """
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


# ── Finalize ──────────────────────────────────────────────────────────────────
@app.post("/projects/{pid}/finalize")
async def finalize_project(pid: str, bg: BackgroundTasks):
    """
    Step 5 — stitch all done scenes into the final film with lightning transitions.
    Runs in the background; poll GET /projects/{pid} for final_status.
    """
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
    return FileResponse(p["final_video"], media_type="video/mp4",
                        filename=f"{p['title'].replace(' ','_')}_final.mp4")


# ── Background tasks ──────────────────────────────────────────────────────────

async def _gen_source_image(pid: str, scene_idx: int, image_prompt: str):
    """
    Runs the txt2img workflow on ComfyUI (GPU_LOCK serialises with video jobs).
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

    All calls go through comfy.run_job() which holds GPU_LOCK, so they
    naturally queue behind any concurrent image-gen requests.
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

            # Prepare seed for next clip
            if i < settings.CLIPS_PER_SCENE - 1:
                last_frame = storage.last_frame_path(pid, scene_idx)
                extract_last_frame(clip_path, last_frame)
                seed_image = last_frame

        # Assemble 5 clips → 20s scene.mp4
        clip_paths  = [storage.clip_path(pid, scene_idx, i)
                       for i in range(settings.CLIPS_PER_SCENE)]
        scene_video = storage.scene_video_path(pid, scene_idx)
        assemble_scene(clip_paths, scene_video)

        storage.update_scene(pid, scene_idx,
                             scene_video=str(scene_video),
                             status="done")
    except Exception as e:
        storage.update_scene(pid, scene_idx, status="error", error=str(e))


async def _finalize(pid: str, done_scenes: list):
    """Concatenates all done scene videos with lightning-flash transitions."""
    try:
        scene_paths = [Path(s["scene_video"]) for s in done_scenes]
        out_path    = storage.final_video_path(pid)
        concatenate_scenes(scene_paths, out_path)

        storage.update_project(pid,
                               final_video=str(out_path),
                               final_video_url=storage.rel_url(out_path),
                               final_status="done")
    except Exception as e:
        storage.update_project(pid, final_status="error", final_error=str(e))
