"""Omni VACE server — FastAPI wrapping vLLM-Omni."""
import os, sys, time, json, base64, gc
import torch
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

os.environ.update({"HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1","DIFFUSERS_OFFLINE":"1"})

app = FastAPI()

MODEL = os.environ.get("OMNI_MODEL", "/models/wan2.1-vace-14b-diffusers")
PORT = int(os.environ.get("OMNI_PORT", "8083"))
_engine = None

class GenRequest(BaseModel):
    prompt: str; width: int=832; height: int=480; num_frames: int=81
    steps: int=18; cfg: float=5.0; seed: int=-1; fps: int=16

def get_engine():
    global _engine
    if _engine is None:
        t0 = time.perf_counter()
        from vllm_omni.entrypoints.omni import Omni
        _engine = Omni(model=MODEL)
        print(f"Omni engine loaded in {time.perf_counter()-t0:.1f}s", flush=True)
    return _engine

@app.get("/health")
async def health():
    return {"status":"ok" if _engine else "idle", "model":MODEL}

@app.post("/generate")
async def generate(req: GenRequest):
    omni = get_engine()
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams
    sp = OmniDiffusionSamplingParams(
        height=req.height, width=req.width, num_frames=req.num_frames,
        num_inference_steps=req.steps, guidance_scale=req.cfg,
        seed=req.seed if req.seed>=0 else None, fps=req.fps)
    t0 = time.perf_counter()
    out = omni.generate([req.prompt, ""], sp, use_async=False)
    elapsed = time.perf_counter()-t0
    frames = out[0].images
    mp4 = _frames_to_mp4(frames, req.fps)
    vram = torch.cuda.max_memory_allocated(0)/(1024*1024)
    return {"status":"success","output":{"type":"video","content":base64.b64encode(mp4).decode(),"format":"mp4","fps":req.fps},
            "metrics":{"latency_s":round(elapsed,2),"vram_peak_mb":int(vram),"steps":req.steps,"num_frames":req.num_frames}}

def _frames_to_mp4(frames, fps=16):
    import tempfile, subprocess, shutil
    from PIL import Image
    tmp = tempfile.mkdtemp()
    try:
        for i,f in enumerate(frames):
            arr = ((f*0.5+0.5).clamp(0,1)*255).to(torch.uint8).cpu().numpy()
            Image.fromarray(arr).save(f"{tmp}/{i:04d}.png")
        out = f"{tmp}/out.mp4"
        subprocess.run(["ffmpeg","-y","-framerate",str(fps),"-i",f"{tmp}/%04d.png",
                       "-c:v","libx264","-pix_fmt","yuv420p","-crf","5",out],
                      capture_output=True, check=True)
        with open(out,"rb") as f: return f.read()
    finally: shutil.rmtree(tmp, ignore_errors=True)

@app.post("/load")
async def load(): get_engine(); return {"status":"loaded"}
@app.post("/release")
async def release():
    global _engine
    if _engine: _engine.close(); _engine=None
    gc.collect(); torch.cuda.empty_cache()
    return {"status":"released"}

if __name__ == "__main__":
    print(f"Omni VACE server on port {PORT}", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=PORT)
