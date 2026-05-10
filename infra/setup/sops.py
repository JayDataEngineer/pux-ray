"""SOPS + AGE encryption setup for Flux CD secrets.

Automates:
  1. AGE keypair generation (config/age.key — gitignored)
  2. .sops.yaml public key update
  3. sops-age k8s secret for Flux decryption
  4. Render secrets from config/secrets.env into shared-infra.yaml
  5. Encrypt to shared-infra.enc.yaml (committed)

Usage:
    python -m infra.setup sops          # Check status
    python -m infra.setup sops --fix    # Generate/encrypt
"""
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
AGE_KEY_PATH = REPO_ROOT / "config" / "age.key"
SOPS_YAML_PATH = REPO_ROOT / ".sops.yaml"
SECRETS_ENV_PATH = REPO_ROOT / "config" / "secrets.env"
TEMPLATE_PATH = REPO_ROOT / "infra/flux/infra-secrets/shared-infra.template.yaml"
SECRETS_YAML_PATH = REPO_ROOT / "infra/flux/infra-secrets/shared-infra.yaml"
ENCRYPTED_PATH = REPO_ROOT / "infra/flux/infra-secrets/shared-infra.enc.yaml"
PLACEHOLDER = "AGE_PUBLIC_KEY_PLACEHOLDER"

GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
RESET = "\033[0m"


def _log(msg: str) -> None:
    print(f"  {GREEN}✓{RESET} {msg}")


def _warn(msg: str) -> None:
    print(f"  {YELLOW}!{RESET} {msg}")


def _err(msg: str) -> None:
    print(f"  {RED}✗{RESET} {msg}")


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kwargs)


def _read_env(path: Path) -> dict[str, str]:
    secrets = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition("=")
        secrets[key.strip()] = value.strip()
    return secrets


def _check_binary(name: str) -> bool:
    return shutil.which(name) is not None


def _get_public_key() -> str | None:
    if not AGE_KEY_PATH.exists():
        return None
    for line in AGE_KEY_PATH.read_text().splitlines():
        if line.startswith("# public key:"):
            return line.split(":", 1)[1].strip()
    return None


def ensure_age_keypair(fix: bool) -> str | None:
    pub = _get_public_key()
    if pub:
        _log(f"AGE keypair exists (public: {pub[:20]}...)")
        return pub
    if not fix:
        _warn("AGE keypair missing (config/age.key)")
        return None
    if not _check_binary("age-keygen"):
        _err("age-keygen not found. Install: sudo apt install age")
        return None
    AGE_KEY_PATH.parent.mkdir(parents=True, exist_ok=True)
    r = _run(["age-keygen", "-o", str(AGE_KEY_PATH)])
    if r.returncode != 0:
        _err(f"age-keygen failed: {r.stderr}")
        return None
    pub = _get_public_key()
    _log(f"Generated AGE keypair (public: {pub[:20]}...)")
    return pub


def ensure_sops_yaml(pub: str, fix: bool) -> bool:
    content = SOPS_YAML_PATH.read_text()
    if PLACEHOLDER not in content:
        if "age1" in content:
            _log(".sops.yaml has real public key")
            return True
        _warn(".sops.yaml has unexpected format")
        return False
    if not fix:
        _warn(".sops.yaml still has placeholder public key")
        return False
    content = content.replace(PLACEHOLDER, pub)
    SOPS_YAML_PATH.write_text(content)
    _log(f".sops.yaml updated with public key")
    return True


def ensure_sops_age_secret(fix: bool) -> bool:
    r = _run(["kubectl", "get", "secret", "sops-age", "-n", "flux-system"])
    if r.returncode == 0:
        _log("sops-age secret exists in flux-system")
        return True
    if not fix:
        _warn("sops-age secret missing in flux-system")
        return False
    if not AGE_KEY_PATH.exists():
        _err("config/age.key missing — run ensure_age_keypair first")
        return False
    r = _run([
        "kubectl", "create", "secret", "generic", "sops-age",
        "-n", "flux-system",
        f"--from-file=age.agekey={AGE_KEY_PATH}",
        "--dry-run=client", "-o", "yaml",
    ])
    if r.returncode != 0:
        _err(f"Secret generation failed: {r.stderr}")
        return False
    r2 = _run(["kubectl", "apply", "-f", "-"], input=r.stdout)
    if r2.returncode != 0:
        _err(f"Secret apply failed: {r2.stderr}")
        return False
    _log("sops-age secret created in flux-system")
    return True


def render_secrets_yaml(fix: bool) -> bool:
    if ENCRYPTED_PATH.exists():
        _log("shared-infra.enc.yaml exists (already encrypted)")
        return True
    if not SECRETS_ENV_PATH.exists():
        _warn("config/secrets.env missing — create from config/secrets.env.example")
        return False
    if not TEMPLATE_PATH.exists():
        _err("shared-infra.template.yaml missing")
        return False
    if not fix:
        _warn("shared-infra.enc.yaml missing (need to render + encrypt)")
        return False

    env = _read_env(SECRETS_ENV_PATH)
    template = TEMPLATE_PATH.read_text()

    for key, value in env.items():
        template = template.replace(f"{key}: CHANGE_ME", f"{key}: {value}")

    remaining = template.count("CHANGE_ME")
    if remaining:
        _warn(f"{remaining} CHANGE_ME placeholders remain (keys not in secrets.env)")

    SECRETS_YAML_PATH.write_text(template)
    _log(f"Rendered shared-infra.yaml ({len(env)} secrets from env)")
    return True


def encrypt_secrets_yaml(fix: bool) -> bool:
    if ENCRYPTED_PATH.exists():
        _log("shared-infra.enc.yaml exists")
        return True
    if not SECRETS_YAML_PATH.exists():
        _warn("shared-infra.yaml missing — run render first")
        return False
    if not fix:
        _warn("shared-infra.enc.yaml missing (need to encrypt)")
        return False
    if not _check_binary("sops"):
        _err("sops not found. Install: https://github.com/getsops/sops/releases")
        return False

    r = _run(["sops", "-e", str(SECRETS_YAML_PATH)])
    if r.returncode != 0:
        _err(f"sops encrypt failed: {r.stderr}")
        return False

    ENCRYPTED_PATH.write_text(r.stdout)
    SECRETS_YAML_PATH.unlink(missing_ok=True)
    _log(f"Encrypted → shared-infra.enc.yaml")
    return True


def main() -> int:
    fix = "--fix" in sys.argv
    print(f"SOPS Secrets Setup {'(fix mode)' if fix else '(check mode)'}")
    print()

    if not _check_binary("age-keygen") or not _check_binary("sops"):
        print("Missing tools. Install:")
        if not _check_binary("age-keygen"):
            print("  sudo apt install age")
        if not _check_binary("sops"):
            print("  sudo curl -L https://github.com/getsops/sops/releases/download/v3.9.0/sops-v3.9.0.linux.amd64 -o /usr/local/bin/sops && sudo chmod +x /usr/local/bin/sops")
        return 1

    ok = True
    pub = ensure_age_keypair(fix)
    if not pub:
        ok = False

    if pub:
        if not ensure_sops_yaml(pub, fix):
            ok = False
        if not ensure_sops_age_secret(fix):
            ok = False

    if not render_secrets_yaml(fix):
        ok = False

    if not encrypt_secrets_yaml(fix):
        ok = False

    print()
    if ok:
        _log("All SOPS checks passed")
    else:
        if fix:
            _err("Some steps failed — see errors above")
        else:
            _warn("Run with --fix to apply changes")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
