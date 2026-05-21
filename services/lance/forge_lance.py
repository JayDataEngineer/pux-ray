"""Lance Forge service — ByteDance unified multimodal (6 tasks).

Subprocess-managed GPU. Uses run_quant_eval.py for AWQ INT4 inference.

All 6 tasks:
  t2i         text → image
  t2v         text → video
  image_edit  text + image → image
  video_edit  text + video → video
  x2t_image   text + image → text
  x2t_video   text + video → text
"""
from __future__ import annotations

import base64
import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from services.forge_base import ForgeService

logger = logging.getLogger(__name__)

_LANCE_SRC = "/opt/lance"
_LANCE_QUANT = "/opt/lance-quant"

# (resolution, h, w, default_frames, native_task)
RES_CONF = {
    "t2i":         ("image_768res", 768, 768, 1,    "t2i"),
    "t2v":         ("video_480p",   480, 848, 50,   "t2v"),
    "i2v":         ("video_480p",   480, 848, 50,   "video_edit"),
    "image_edit":  ("image_768res", 768, 768, 1,    "image_edit"),
    "video_edit":  ("video_480p",   480, 848, 50,   "video_edit"),
    "x2t_image":   ("image_768res", 768, 768, 1,    "x2t_image"),
    "x2t_video":   ("video_480p",   480, 848, 50,   "x2t_video"),
}

# Task → which script handles it (run_quant_eval handles all)
_TASK = "task"


class LanceForgeService(ForgeService):
    vram_mb = 0
    service_name = "lance"
    default_model = "lance-video-awq"

    def __init__(self):
        super().__init__()
        self._mp: Path | None = None
        self._vp: Path | None = None
        self._ap: Path | None = None
        self._awq = False
        self._vid = False

    def load(self, model_name: str, quant: str | None = None) -> None:
        from registry.config import Config
        mr = Path(Config().models_root)
        self._vid = "video" in model_name
        self._awq = quant in ("int4", "awq") or "awq" in model_name
        sub = "Lance_3B_Video" if self._vid else "Lance_3B"
        awq_sub = "Lance-3B-Video-AWQ-INT4" if self._vid else "Lance-3B-AWQ-INT4"
        ckpt = mr / f"lance/{sub}"
        awq_d = mr / f"lance/{awq_sub}"
        if not ckpt.is_dir():
            raise FileNotFoundError(f"Lance checkpoint not found: {ckpt}")
        self._mp = ckpt
        self._vp = mr / "lance/Qwen2.5-VL-ViT"
        self._ap = awq_d if self._awq and awq_d.is_dir() else None
        self.model_name = model_name
        self._loaded = True
        logger.info("lance: %s awq=%s ckpt=%s", model_name, self._awq, ckpt)

    def unload(self) -> None:
        self._mp = self._vp = self._ap = None
        self._loaded = False
        self.model_name = None

    def infer(self, payload: dict) -> dict:
        task = payload.get(_TASK) or ("t2v" if self._vid else "t2i")
        if task not in RES_CONF:
            return {"status": "error", "error": f"unsupported task: {task}"}

        resolution, h, w, nf, native_task = RES_CONF[task]
        text = payload.get("text") or payload.get("input_prompt", "")
        image_b64 = payload.get("image", "")
        video_b64 = payload.get("video", "")

        with tempfile.TemporaryDirectory(prefix="lance_") as tmpdir:
            save_dir = Path(tmpdir) / "results"
            save_dir.mkdir()
            media_dir = Path(tmpdir) / "media"
            media_dir.mkdir()

            ex = {}

            if task in ("t2i", "t2v"):
                ex = {"000000.png" if task == "t2i" else "000000.mp4": text}

            elif task in ("image_edit", "video_edit"):
                is_vid = task == "video_edit"
                raw = video_b64 if is_vid else image_b64
                if not raw:
                    return {"status": "error", "error": f"{task} requires {'video' if is_vid else 'image'} input"}
                ext = ".mp4" if is_vid else ".png"
                src_path = media_dir / f"source{ext}"
                src_path.write_bytes(base64.b64decode(raw))
                ex["000000.png" if not is_vid else "000000.mp4"] = {
                    "interleave_array": [text, str(src_path), str(src_path)],
                    "element_dtype_array": ["text", "video" if is_vid else "image", "video" if is_vid else "image"],
                    "istarget_in_interleave": [0, 0, 1],
                }

            elif task == "i2v":
                first = payload.get("image", "") or payload.get("image_start", "")
                if not first:
                    return {"status": "error", "error": "i2v requires image_start"}
                fp = media_dir / "first.png"
                fp.write_bytes(base64.b64decode(first))
                last_b64 = payload.get("image_end", "")
                arr = [text, str(fp), str(fp), "000000.mp4"]
                dtypes = ["text", "image", "image", "video"]
                targets = [0, 0, 0, 1]
                if last_b64:
                    lp = media_dir / "last.png"
                    lp.write_bytes(base64.b64decode(last_b64))
                    arr = [text, str(fp), str(fp), str(lp), "000000.mp4"]
                    dtypes = ["text", "image", "image", "image", "video"]
                    targets = [0, 0, 0, 0, 1]
                ex["000000.mp4"] = {
                    "interleave_array": arr,
                    "element_dtype_array": dtypes,
                    "istarget_in_interleave": targets,
                }

            elif task in ("x2t_image", "x2t_video"):
                is_vid = task == "x2t_video"
                raw = video_b64 if is_vid else image_b64
                if not raw:
                    return {"status": "error", "error": f"{task} requires image or video input"}
                ext = ".mp4" if is_vid else ".png"
                src_path = media_dir / f"source{ext}"
                src_path.write_bytes(base64.b64decode(raw))
                question = text or payload.get("question", "Describe this image.")
                ex["000001"] = {
                    "interleave_array": [str(src_path), ["You are a helpful assistant.", question, ""]],
                    "element_dtype_array": ["video" if is_vid else "image", "text"],
                    "istarget_in_interleave": [0, 1],
                }

            prompt_file = Path(tmpdir) / "prompt.json"
            with open(prompt_file, "w") as f:
                for idx_key, data_val in ex.items():
                    json.dump({"index": idx_key, "data": data_val}, f)
                    f.write("\n")

            script_src, extra = self._resolve_script(native_task)
            if not script_src:
                return {"status": "error", "error": "inference script not found"}

            cmd = self._build_cmd(script_src, native_task, resolution, h, w, nf, prompt_file, save_dir, payload, extra)
            env = self._build_env()

            logger.info("lance: %s native_task=%s", script_src.name, native_task)
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=str(_LANCE_SRC), env=env)

            if result.returncode != 0:
                err = (result.stderr or result.stdout or "")[-2000:]
                logger.error("lance fail: %s", err[:500])
                return {"status": "error", "error": err}

            return self._encode_output(save_dir, task)

    def _resolve_script(self, task: str) -> tuple[Path | None, list[str]]:
        if self._awq:
            # Use the patched run_quant_eval.py from /opt/lance (has Lance.to() fix)
            s = Path(_LANCE_SRC) / "run_quant_eval.py"
            if s.exists():
                return s, ["--awq_dir", str(self._ap), "--mode", "ondemand"]
        for p in [Path(_LANCE_SRC) / "run_baseline.py"]:
            if p.exists():
                return p, []
        return None, []

    def _build_cmd(self, script, task, resolution, h, w, nf, prompt_file, save_dir, payload, extra):
        cmd = [
            sys.executable, str(script),
            "--task", task,
            "--model_path", str(self._mp),
            "--vit_path", str(self._vp),
            "--save_path_gen", str(save_dir),
            "--validation_num_timesteps", str(payload.get("num_timesteps", 30)),
            "--cfg_scale", str(payload.get("cfg_text_scale", 4.0)),
            "--seed", str(payload.get("seed", 42)),
            "--num_frames", str(payload.get("num_frames", nf)),
            "--video_height", str(payload.get("video_height", h)),
            "--video_width", str(payload.get("video_width", w)),
            "--resolution", resolution,
            "--example_json", str(prompt_file),
        ] + extra
        return cmd

    def _build_env(self):
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{_LANCE_SRC}:{_LANCE_QUANT}:{env.get('PYTHONPATH', '')}"
        env["POSITION_EMBEDDING_3D_VERSION"] = "v2"
        env["TORCH_COMPILE_DISABLE"] = "1"
        env["TORCHDYNAMO_DISABLE"] = "1"
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        return env

    def _encode_output(self, save_dir: Path, task: str) -> dict:
        if not save_dir.exists():
            return {"status": "error", "error": "no output directory"}
        files = sorted(save_dir.iterdir())
        if not files:
            return {"status": "error", "error": "no output files"}

        if task in ("x2t_image", "x2t_video"):
            for rf in files:
                if rf.name == "result.json":
                    with open(rf) as f:
                        data = json.load(f)
                    answer = data[0].get("answer", "") if data else ""
                    return {"status": "success", "data": base64.b64encode(answer.encode()).decode(), "media_type": "text/plain", "text": answer}
                if rf.suffix == ".json":
                    c = rf.read_text().strip()
                    return {"status": "success", "data": base64.b64encode(c.encode()).decode(), "media_type": "text/plain", "text": c}

        for rf in files:
            if rf.suffix in (".mp4", ".png", ".jpg", ".gif"):
                with open(rf, "rb") as f:
                    b = base64.b64encode(f.read()).decode()
                mt = "video/mp4" if rf.suffix == ".mp4" else "image/png"
                return {"status": "success", "data": b, "media_type": mt, "filename": rf.name}

        return {"status": "error", "error": f"no media files in {save_dir}"}
