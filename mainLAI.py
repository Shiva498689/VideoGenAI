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

app = FastAPI(title="CineForge — AI Video Pipeline", version="3.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

storage = StorageManager(settings.OUTPUT_DIR)

app.mount("/files", StaticFiles(directory=str(settings.OUTPUT_DIR)), name="files")
app.mount("/ui",    StaticFiles(directory="frontend", html=True),    name="ui")


class CreateProjectReq(BaseModel):
    title: str = "Untitled Project"

class SceneReq(BaseModel):
    prompt: str

class ConfirmReq(BaseModel):
    confirmed: bool   

@app.get("/")
async def root():
    return {"status": "ok", "ui": "/ui", "docs": "/docs"}


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
    p = dict(p)
    p["scenes"] = [_hydrate_scene(p["project_id"], s) for s in p.get("scenes", [])]
    return p


@app.post("/projects/{pid}/scenes", status_code=201)
async def create_scene(pid: str, req: SceneReq, bg: BackgroundTasks):
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


@app.get("/projects/{pid}/scenes/{scene_idx}")
async def get_scene(pid: str, scene_idx: int):
    s = storage.get_scene(pid, scene_idx)
    if not s:
        raise HTTPException(404, "Scene not found")
    return _hydrate_scene(pid, s)


def _hydrate_scene(pid: str, s: dict) -> dict:
    s = dict(s)
    if s.get("source_image"):
        s["source_image_url"] = storage.rel_url(s["source_image"])
    s["clip_urls"] = [storage.rel_url(c) for c in s.get("clips", [])]
    if s.get("scene_video"):
        s["scene_video_url"] = storage.rel_url(s["scene_video"])
    return s


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


async def _gen_source_image(pid: str, scene_idx: int, image_prompt: str):
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

            if i < settings.CLIPS_PER_SCENE - 1:
                last_frame = storage.last_frame_path(pid, scene_idx)
                await asyncio.to_thread(extract_last_frame, clip_path, last_frame)
                seed_image = last_frame

        clip_paths  = [storage.clip_path(pid, scene_idx, i)
                       for i in range(settings.CLIPS_PER_SCENE)]
        scene_video = storage.scene_video_path(pid, scene_idx)
        await asyncio.to_thread(assemble_scene, clip_paths, scene_video)

        storage.update_scene(pid, scene_idx,
                             scene_video=str(scene_video),
                             status="done")
    except Exception as e:
        storage.update_scene(pid, scene_idx, status="error", error=str(e))


async def _finalize(pid: str, done_scenes: list):
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
