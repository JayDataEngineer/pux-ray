"""Host bootstrap for Tech Noir k3s + Flux cluster.

Configures host-level settings that k3s containerd and Docker need
before Flux can manage the rest. Idempotent — safe to re-run.

Usage:
    sudo python -m infra.setup bootstrap          # Check status
    sudo python -m infra.setup bootstrap --fix     # Apply fixes
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
HOSTS_PATH = Path("/etc/hosts")
REGISTRIES_PATH = Path("/etc/rancher/k3s/registries.yaml")
DOCKER_DAEMON_PATH = Path("/etc/docker/daemon.json")
REGISTRIES_EXAMPLE = REPO_ROOT / "config" / "registries.yaml.example"
FLUX_KUSTOMIZATION = REPO_ROOT / "infra/flux/clusters/forge/kustomization.yaml"

REGISTRY_HOST = "forge-reg.local"
REGISTRY_PORT = 30500

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


def check_hosts_entry(fix: bool) -> bool:
    entry = f"127.0.0.1 {REGISTRY_HOST}"
    content = HOSTS_PATH.read_text()
    if REGISTRY_HOST in content:
        _log(f"/etc/hosts has {REGISTRY_HOST} entry")
        return True
    if not fix:
        _warn(f"/etc/hosts missing {REGISTRY_HOST} entry")
        return False
    with open(HOSTS_PATH, "a") as f:
        f.write(f"\n{entry}\n")
    _log(f"Added '{entry}' to /etc/hosts")
    return True


def check_k3s_registries(fix: bool) -> bool:
    if REGISTRIES_PATH.exists() and REGISTRY_HOST in REGISTRIES_PATH.read_text():
        _log(f"registries.yaml configured ({REGISTRY_HOST})")
        return True
    if not fix:
        _warn("registries.yaml missing or outdated")
        return False
    if not REGISTRIES_EXAMPLE.exists():
        _err(f"config/registries.yaml.example missing")
        return False
    REGISTRIES_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(REGISTRIES_EXAMPLE, REGISTRIES_PATH)
    _log(f"Copied registries.yaml → {REGISTRIES_PATH}")
    return True


def check_docker_insecure_registry(fix: bool) -> bool:
    registry_url = f"{REGISTRY_HOST}:{REGISTRY_PORT}"
    if not DOCKER_DAEMON_PATH.exists():
        if not fix:
            _warn("/etc/docker/daemon.json not found")
            return False
        config = {"insecure-registries": [registry_url]}
        DOCKER_DAEMON_PATH.write_text(json.dumps(config, indent=2) + "\n")
        _log("Created /etc/docker/daemon.json with insecure registry")
        return True

    config = json.loads(DOCKER_DAEMON_PATH.read_text())
    insecure = config.get("insecure-registries", [])
    if registry_url in insecure:
        _log(f"docker daemon has insecure registry ({registry_url})")
        return True
    if not fix:
        _warn(f"docker daemon missing insecure registry ({registry_url})")
        return False

    insecure.append(registry_url)
    config["insecure-registries"] = insecure
    DOCKER_DAEMON_PATH.write_text(json.dumps(config, indent=2) + "\n")
    _log(f"Added {registry_url} to docker insecure-registries")
    return True


def check_flux_controllers(fix: bool) -> bool:
    r = _run(["kubectl", "get", "deploy", "source-controller", "-n", "flux-system"])
    if r.returncode == 0:
        _log("Flux controllers installed (source-controller running)")
        return True
    if not fix:
        _warn("Flux controllers not installed")
        return False
    r = _run(["flux", "install", "--components=source-controller,kustomize-controller"])
    if r.returncode != 0:
        _err(f"Flux install failed: {r.stderr}")
        return False
    _log("Flux controllers installed")
    return True


def check_github_token_secret(fix: bool) -> bool:
    r = _run(["kubectl", "get", "secret", "github-token", "-n", "flux-system"])
    if r.returncode == 0:
        _log("github-token secret exists in flux-system")
        return True
    if not fix:
        _warn("github-token secret missing (Flux can't pull private repo)")
        return False

    token = _run(["gh", "auth", "token"]).stdout.strip()
    if not token:
        token = input("Enter GitHub PAT (repo scope): ").strip()
    if not token:
        _err("No token provided")
        return False

    r = _run([
        "kubectl", "create", "secret", "generic", "github-token",
        "-n", "flux-system",
        f"--from-literal=password={token}",
        "--from-literal=username=JayDataEngineer",
        "--dry-run=client", "-o", "yaml",
    ])
    if r.returncode != 0:
        _err(f"Secret generation failed: {r.stderr}")
        return False
    r2 = _run(["kubectl", "apply", "-f", "-"], input=r.stdout)
    if r2.returncode != 0:
        _err(f"Secret apply failed: {r2.stderr}")
        return False
    _log("github-token secret created in flux-system")
    return True


def check_flux_bootstrap(fix: bool) -> bool:
    r = _run(["flux", "get", "kustomizations"])
    if r.returncode == 0 and "tech-noir" in _run([
        "kubectl", "get", "gitrepository", "tech-noir", "-n", "flux-system",
    ]).stdout:
        _log("Flux bootstrapped (tech-noir GitRepository exists)")
        return True
    if not fix:
        _warn("Flux not bootstrapped (no tech-noir GitRepository)")
        return False
    if not FLUX_KUSTOMIZATION.exists():
        _err(f"Flux kustomization missing: {FLUX_KUSTOMIZATION}")
        return False
    r = _run(["kubectl", "apply", "-f", str(FLUX_KUSTOMIZATION)])
    if r.returncode != 0:
        _err(f"Flux bootstrap failed: {r.stderr}")
        return False
    _log("Flux bootstrapped (applied kustomization.yaml)")
    return True


def check_sops(fix: bool) -> bool:
    from infra.setup.sops import main as sops_main
    old_argv = sys.argv
    sys.argv = ["sops"] + (["--fix"] if fix else [])
    try:
        return sops_main() == 0
    finally:
        sys.argv = old_argv


def main() -> int:
    fix = "--fix" in sys.argv
    print(f"Tech Noir Host Bootstrap {'(fix mode)' if fix else '(check mode)'}")
    print()

    checks = [
        ("Hosts entry", check_hosts_entry),
        ("K3s registries", check_k3s_registries),
        ("Docker insecure registry", check_docker_insecure_registry),
        ("Flux controllers", check_flux_controllers),
        ("GitHub token secret", check_github_token_secret),
        ("Flux bootstrap", check_flux_bootstrap),
        ("SOPS secrets", check_sops),
    ]

    results = {}
    for name, fn in checks:
        print(f"[{name}]")
        results[name] = fn(fix)
        print()

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    print(f"Results: {passed}/{total} checks passed")
    if passed < total:
        if fix:
            _err("Some fixes failed — see errors above")
        else:
            _warn("Run with --fix to apply changes")
    else:
        _log("All checks passed — host is ready")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
