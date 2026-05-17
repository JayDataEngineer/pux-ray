"""Tech Noir CLI — Flux-native service lifecycle management.

All K8s services are managed by Flux CD. This CLI verifies cluster health,
starts Docker services (non-K8s), and provides status/repair commands.

Usage:
    tech-noir boot              Verify k3s + Flux health, start Docker services
    tech-noir boot docker       Start Docker services only
    tech-noir status            Show Flux kustomizations + Docker service status
    tech-noir stop              Stop Docker services (K8s services managed by Flux)
    tech-noir stop <name>       Stop a specific Docker service
    tech-noir up <name>         Start a specific Docker service
    tech-noir heal              Force-reconcile all Flux Kustomizations
"""

from __future__ import annotations

import logging
import subprocess
import sys

from rich.console import Console
from rich.table import Table

from boot.health import Status
from boot.services import (
    ServiceType,
    all_services,
    get,
    get_all_status,
    start_service,
    stop_service,
    SERVICES,
)

console = Console()
logging.basicConfig(level=logging.INFO, format="%(message)s")


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kwargs)


def _flux_healthy() -> bool:
    """Check if Flux is installed and reconciling."""
    r = _run(["flux", "get", "kustomizations"])
    return r.returncode == 0


def _wait_flux_healthy(timeout: int = 300) -> bool:
    """Wait for all Flux Kustomizations to be Ready."""
    import time
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = _run(["flux", "get", "kustomizations", "-o", "json"])
        if r.returncode != 0:
            remaining = int(deadline - time.time())
            if remaining > 0:
                time.sleep(5)
            continue
        import json
        try:
            kustomizations = json.loads(r.stdout)
        except json.JSONDecodeError:
            time.sleep(5)
            continue
        all_ready = True
        for k in kustomizations:
            ready = False
            for c in k.get("status", {}).get("conditions", []):
                if c.get("type") == "Ready" and c.get("status") == "True":
                    ready = True
                    break
            if not ready:
                all_ready = False
                break
        if all_ready:
            return True
        remaining = int(deadline - time.time())
        if remaining <= 0:
            return False
        time.sleep(10)
    return False


def cmd_boot(target: str | None = None) -> None:
    """Verify Flux health and start Docker services.

    Args:
        target: "docker" for Docker services only,
                None for everything (Flux + Docker).
    """
    # Phase 1: Verify k3s
    r = _run(["kubectl", "get", "node"])
    if r.returncode != 0:
        console.print("[red]k3s is not running. Start it first: sudo systemctl start k3s[/red]")
        sys.exit(1)
    console.print("[green]k3s is running[/green]")

    # Phase 2: Verify Flux
    if not _flux_healthy():
        console.print("[yellow]Flux not healthy — bootstrapping...[/yellow]")
        r = _run(["kubectl", "apply", "-f", "infra/flux/clusters/forge/kustomization.yaml"])
        if r.returncode != 0:
            console.print(f"[red]Flux bootstrap failed: {r.stderr}[/red]")
            sys.exit(1)

    if target != "docker":
        console.print("[cyan]Waiting for Flux Kustomizations to be Ready...[/cyan]")
        if _wait_flux_healthy(timeout=300):
            console.print("[green]All Flux Kustomizations are Ready[/green]")
        else:
            console.print("[yellow]Some Flux Kustomizations not Ready (will self-heal)[/yellow]")

    # Phase 3: Start Docker services (non-K8s services like redshiftdb)
    if target == "docker" or target is None:
        docker_svcs = [s for s in all_services() if s.type == ServiceType.DOCKER]
        if docker_svcs:
            console.print(f"\n[cyan]Starting {len(docker_svcs)} Docker services...[/cyan]\n")
            for svc in docker_svcs:
                console.print(f"  {svc.label}...", end=" ")
                success = start_service(svc)
                if success:
                    console.print("[green]OK[/green]")
                else:
                    console.print("[red]FAILED[/red]")

    console.print()
    cmd_status()


def cmd_heal() -> None:
    """Force-reconcile all Flux Kustomizations."""
    console.print("[cyan]Force-reconciling all Flux Kustomizations...[/cyan]\n")

    r = _run(["flux", "get", "kustomizations", "-o", "json"])
    if r.returncode != 0:
        console.print("[red]Cannot list Flux Kustomizations. Is Flux installed?[/red]")
        sys.exit(1)

    import json
    try:
        kustomizations = json.loads(r.stdout)
    except json.JSONDecodeError:
        console.print("[red]Cannot parse Flux output[/red]")
        sys.exit(1)

    for k in kustomizations:
        name = k["metadata"]["name"]
        ns = k["metadata"]["namespace"]
        console.print(f"  Reconciling {name}...", end=" ")
        r = _run(["flux", "reconcile", "kustomization", name, "-n", ns, "--force"])
        if r.returncode == 0:
            console.print("[green]OK[/green]")
        else:
            console.print(f"[red]FAILED: {r.stderr.strip()}[/red]")

    console.print()
    cmd_status()


def cmd_up(name: str) -> None:
    """Start a specific service by name."""
    svc = get(name)
    if not svc:
        console.print(f"[red]Unknown service: {name}[/red]")
        console.print(f"Available: {', '.join(SERVICES.keys())}")
        return

    console.print(f"\nStarting [cyan]{svc.label}[/cyan]...")
    success = start_service(svc)
    if success:
        console.print("[green]Started[/green]")
    else:
        console.print("[red]Failed[/red]")


def cmd_stop(name: str | None = None) -> None:
    """Stop Docker services. K8s services are managed by Flux."""
    if name:
        svc = get(name)
        if not svc:
            console.print(f"[red]Unknown service: {name}[/red]")
            return
        console.print(f"Stopping [cyan]{svc.label}[/cyan]...")
        stop_service(svc)
        console.print("[green]Stopped[/green]")
        return

    console.print("\n[bold yellow]Stopping Docker services...[/bold yellow]")
    console.print("[dim](K8s services are managed by Flux — use flux suspend to pause)[/dim]\n")

    docker_svcs = [s for s in all_services() if s.type == ServiceType.DOCKER]
    for svc in reversed(docker_svcs):
        health = get_all_status().get(svc.name)
        if health and health.status != Status.STOPPED:
            console.print(f"  {svc.label}...", end=" ")
            stop_service(svc)
            console.print("[green]stopped[/green]")

    console.print()


def cmd_status() -> None:
    """Show Flux kustomization health + Docker service status."""
    # Flux Kustomizations
    r = _run(["flux", "get", "kustomizations"])
    if r.returncode == 0 and r.stdout.strip():
        flux_table = Table(title="Flux Kustomizations", show_header=True, header_style="bold")
        flux_table.add_column("Name", style="cyan")
        flux_table.add_column("Ready")
        flux_table.add_column("Status")
        flux_table.add_column("Message")

        import json
        rj = _run(["flux", "get", "kustomizations", "-o", "json"])
        if rj.returncode == 0:
            try:
                kustomizations = json.loads(rj.stdout)
                for k in kustomizations:
                    name = k["metadata"]["name"]
                    ready = "?"
                    status = "?"
                    message = ""
                    for c in k.get("status", {}).get("conditions", []):
                        if c.get("type") == "Ready":
                            ready = c.get("status", "?")
                            message = c.get("message", "")
                        if c.get("type") == "Reconciling":
                            status = c.get("status", "?")
                    ready_str = "[green]True[/green]" if ready == "True" else f"[red]{ready}[/red]"
                    status_str = "Reconciling" if status == "True" else "Idle"
                    flux_table.add_row(name, ready_str, status_str, message[:60])
            except json.JSONDecodeError:
                console.print(r.stdout)
        console.print(flux_table)
    else:
        console.print("[yellow]Flux not available (k3s may not be running)[/yellow]")

    # Docker services
    docker_svcs = [s for s in all_services() if s.type == ServiceType.DOCKER]
    if docker_svcs:
        docker_table = Table(title="Docker Services", show_header=True, header_style="bold")
        docker_table.add_column("Service", style="cyan")
        docker_table.add_column("Status")
        docker_table.add_column("Port")
        docker_table.add_column("Detail")

        status_styles = {
            Status.HEALTHY: "[green]running[/green]",
            Status.UNHEALTHY: "[yellow]degraded[/yellow]",
            Status.STOPPED: "[red]stopped[/red]",
            Status.UNKNOWN: "[dim]unknown[/dim]",
        }

        for svc in docker_svcs:
            result = get_all_status().get(svc.name)
            if result:
                status_str = status_styles.get(result.status, str(result.status))
                detail = result.detail
            else:
                status_str = "[dim]unknown[/dim]"
                detail = ""
            port_str = str(svc.port) if svc.port else "—"
            docker_table.add_row(svc.label, status_str, port_str, detail)

        console.print(docker_table)

    # Ray cluster quick check
    r = _run(["kubectl", "get", "rayservice", "-A"])
    if r.returncode == 0 and r.stdout.strip():
        console.print(f"\n[bold]Ray Services:[/bold]")
        console.print(r.stdout)

    # Pod summary
    r = _run(["kubectl", "get", "pods", "-A", "--field-selector=status.phase!=Running"])
    if r.returncode == 0 and "No resources" not in r.stdout:
        console.print(f"\n[bold yellow]Non-running pods:[/bold yellow]")
        console.print(r.stdout)


def main() -> None:
    """CLI entry point."""
    args = sys.argv[1:]

    if not args:
        cmd_status()
        return

    command = args[0]

    if command == "boot":
        target = args[1] if len(args) > 1 else None
        cmd_boot(target)
    elif command == "up":
        if len(args) < 2:
            console.print("[red]Usage: tech-noir up <service-name>[/red]")
            sys.exit(1)
        cmd_up(args[1])
    elif command == "stop":
        target = args[1] if len(args) > 1 else None
        cmd_stop(target)
    elif command == "status":
        cmd_status()
    elif command == "heal":
        cmd_heal()
    else:
        console.print(f"[red]Unknown command: {command}[/red]")
        console.print("Commands: boot, up, stop, status, heal")
        sys.exit(1)


if __name__ == "__main__":
    main()
