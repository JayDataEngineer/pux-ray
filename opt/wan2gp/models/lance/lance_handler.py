"""Lance family handler — ByteDance unified multimodal model (AWQ INT4).

Calls run_quant_eval.py as subprocess, same pattern as the ComfyUI custom node.
Requires Lance source at /opt/lance and lance-quant at /opt/lance-quant.
"""

import base64
import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from models.base_handler import BaseFamilyHandler, _make_handler_cls

logger = logging.getLogger(__name__)

_LANCE_SRC = "/opt/lance"
_LANCE_QUANT = "/opt/lance-quant"

TASK_MAP = {
    "t2i": ("image", "image_768res", "image", False, 768, 768, 1),
    "t2v": ("video", "video_480p", "video", True, 480, 848, 50),
    "image_edit": ("image", "image_768res", "image", False, 768, 768, 1),
    "video_edit": ("video", "video_480p", "video", True, 480, 848, 50),
    "x2t_image": ("image", "image_768res", "image_understanding", False, 768, 768, 1),
    "x2t_video": ("video", "video_480p", "video_understanding", True, 480, 848, 50),
}


@_make_handler_cls
class family_handler(BaseFamilyHandler):
    SUPPORTED_TYPES = [
        "lance-image", "lance-video",
        "lance-image-awq", "lance-video-awq",
    ]
    FAMILY = "lance"
    FAMILY_INFOS = {"lance": (501, "Lance")}
    MODEL_DEF = {"image_outputs": False, "audio_only": False}
    DEFAULTS = {"prompt": "a cat walking on a sunny street, cinematic quality"}

    @staticmethod
    def load_model(model_filename, model_type, base_model_type, model_def,
                   quantizeTransformer=False, text_encoder_quantization=None,
                   dtype=None, VAE_dtype=None, profile=0, **kwargs):
        from registry.config import Config
        cfg = Config()
        mr = Path(cfg.models_root)

        is_video = "video" in model_type
        quant = kwargs.get("quant")
        is_awq = quant in ("int4", "awq") or "awq" in model_type

        if is_video:
            ckpt = mr / "lance/Lance_3B_Video"
            awq_d = mr / "lance/Lance-3B-Video-AWQ-INT4"
        else:
            ckpt = mr / "lance/Lance_3B"
            awq_d = mr / "lance/Lance-3B-AWQ-INT4"

        pipeline = _Pipeline(
            model_path=ckpt,
            vit_path=mr / "lance/Qwen2.5-VL-ViT",
            vae_path=mr / "lance/Wan2.2_VAE.pth",
            is_video=is_video,
            is_awq=is_awq and awq_d.is_dir(),
            awq_path=awq_d,
            lance_src=Path(_LANCE_SRC),
            lance_quant=Path(_LANCE_QUANT),
        )
        return pipeline, {"pipe": {}, "coTenantsMap": {}}


class _Pipeline:
    def __init__(self, model_path, vit_path, vae_path, is_video, is_awq,
                 awq_path, lance_src, lance_quant):
        self.model_path = model_path
        self.vit_path = vit_path
        self.vae_path = vae_path
        self.is_video = is_video
        self.is_awq = is_awq
        self.awq_path = awq_path
        self.lance_src = lance_src
        self.lance_quant = lance_quant
        self._dev = "cuda"

    @property
    def device(self):
        return self._dev

    def generate(self, *, input_prompt="", task=None, **kwargs):
        text = input_prompt or kwargs.get("text", "")
        if not text:
            return {"status": "error", "error": "text input required"}

        task = task or ("t2v" if self.is_video else "t2i")
        if task not in TASK_MAP:
            return {"status": "error", "error": f"unsupported task: {task}"}

        modality, resolution, sample_modality, vis, h, w, nf = TASK_MAP[task]
        seed = kwargs.get("seed", 42)
        num_frames = kwargs.get("num_frames", nf)
        video_height = kwargs.get("video_height", h)
        video_width = kwargs.get("video_width", w)
        num_timesteps = kwargs.get("num_timesteps", 30)
        cfg_scale = kwargs.get("cfg_text_scale", 4.0)
        mode = kwargs.get("mode", "ondemand")

        with tempfile.TemporaryDirectory(prefix="lance_infer_") as tmpdir:
            save_dir = Path(tmpdir) / "results"
            save_dir.mkdir()

            prompt_file = Path(tmpdir) / "prompt.json"
            with open(prompt_file, "w") as f:
                json.dump({"index": "000000.png", "data": text}, f)
                f.write("\n")

            extra = []
            if self.is_awq:
                script = self.lance_quant / "scripts/run_quant_eval.py"
                extra = ["--awq_dir", str(self.awq_path), "--mode", mode]
            else:
                script = self.lance_quant / "scripts/run_baseline.py"
                if not script.exists():
                    script = self.lance_src / "run_baseline.py"

            if not script.exists():
                return {"status": "error", "error": f"script not found: {script}"}

            cmd = [
                sys.executable, str(script),
                "--task", task,
                "--model_path", str(self.model_path),
                "--vit_path", str(self.vit_path),
                "--save_path_gen", str(save_dir),
                "--validation_num_timesteps", str(num_timesteps),
                "--cfg_scale", str(cfg_scale),
                "--seed", str(seed),
                "--num_frames", str(num_frames),
                "--video_height", str(video_height),
                "--video_width", str(video_width),
                "--resolution", resolution,
                "--example_json", str(prompt_file),
            ] + extra

            env = os.environ.copy()
            env["PYTHONPATH"] = f"{self.lance_src}:{self.lance_quant}:{env.get('PYTHONPATH', '')}"
            env["POSITION_EMBEDDING_3D_VERSION"] = "v2"
            env["TORCH_COMPILE_DISABLE"] = "1"
            env["TORCHDYNAMO_DISABLE"] = "1"
            env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

            logger.info("Running: %s %s ...", sys.executable, script.name)
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=600,
                cwd=str(self.lance_src), env=env,
            )

            stdout = result.stdout or ""
            stderr = result.stderr or ""
            for line in (stdout + stderr).split("\n"):
                if any(kw in line for kw in ("SUCCESS", "INFERENCE_DONE",
                                              "Validating:", "awq-loader",
                                              "quant-swap", "ERROR",
                                              "Traceback", "Error")):
                    logger.info("lance: %s", line.strip())

            if result.returncode != 0 and "SUCCESS" not in stdout:
                return {"status": "error", "error": stderr[-2000:] or stdout[-2000:]}

            return self._encode_output(save_dir, task)

    def _encode_output(self, save_dir, task):
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
                    return {
                        "status": "success",
                        "data": base64.b64encode(answer.encode()).decode(),
                        "media_type": "text/plain",
                        "text": answer,
                    }
                if rf.suffix == ".json":
                    content = rf.read_text().strip()
                    return {
                        "status": "success",
                        "data": base64.b64encode(content.encode()).decode(),
                        "media_type": "text/plain",
                        "text": content,
                    }

        for rf in files:
            if rf.suffix in (".mp4", ".png", ".jpg", ".gif"):
                with open(rf, "rb") as f:
                    data_b64 = base64.b64encode(f.read()).decode()
                media_type = "video/mp4" if rf.suffix == ".mp4" else "image/png"
                return {
                    "status": "success",
                    "data": data_b64,
                    "media_type": media_type,
                    "filename": rf.name,
                }

        return {"status": "error", "error": f"no media files in {save_dir}: {[f.name for f in files]}"}
