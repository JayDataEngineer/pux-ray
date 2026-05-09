"""Read config/secrets.env and sync to k8s secrets in all namespaces.

Source of truth: config/secrets.env
Target: 'shared-infra' k8s secret in every namespace that needs it.

Usage: python infra/secrets_sync.py
"""

import subprocess
import sys
from pathlib import Path

NAMESPACES = ["infra", "mcp", "ai-services"]
SECRET_NAME = "shared-infra"
ENV_PATH = Path("config/secrets.env")


def read_env(path: Path) -> dict[str, str]:
    secrets = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition("=")
        secrets[key.strip()] = value.strip()
    return secrets


def sync(namespace: str, secrets: dict[str, str]) -> bool:
    args = [
        "kubectl", "create", "secret", "generic", SECRET_NAME,
        f"--namespace={namespace}",
        "--dry-run=client", "-o", "yaml",
    ]
    for key, value in secrets.items():
        args.append(f"--from-literal={key}={value}")

    gen = subprocess.run(args, capture_output=True, text=True)
    if gen.returncode != 0:
        print(f"  {namespace}: generate failed: {gen.stderr.strip()}")
        return False

    app = subprocess.run(
        ["kubectl", "apply", "-f", "-"],
        input=gen.stdout, capture_output=True, text=True,
    )
    if app.returncode != 0:
        print(f"  {namespace}: apply failed: {app.stderr.strip()}")
        return False

    status = "configured" if "configured" in app.stdout else "created"
    print(f"  {namespace}: {status} ({len(secrets)} keys)")
    return True


def main() -> None:
    if not ENV_PATH.exists():
        print(f"ERROR: {ENV_PATH} not found. Copy from config/secrets.env.example")
        sys.exit(1)

    secrets = read_env(ENV_PATH)
    if not secrets:
        print("ERROR: no secrets found in config/secrets.env")
        sys.exit(1)

    print(f"Syncing {len(secrets)} secrets...")
    ok = all(sync(ns, secrets) for ns in NAMESPACES)

    if ok:
        print("Done.")
    else:
        print("FAILED: some namespaces failed to sync.")
        sys.exit(1)


if __name__ == "__main__":
    main()
