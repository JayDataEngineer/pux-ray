"""Tech Noir CLI — unified service lifecycle management.

Usage:
    tech-noir boot            Start all services
    tech-noir boot ray        Start Ray stack only (cluster + serve + ingress)
    tech-noir boot docker     Start all Docker services
    tech-noir up <name>       Start a specific service
    tech-noir stop            Stop all services
    tech-noir stop <name>     Stop a specific service
    tech-noir status          Show all service status
"""

from __future__ import annotations

import logging
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


def cmd_boot(target: str | None = None) -> None:
    """Start services.

    Args:
        target: "ray" for Ray stack, "docker" for Docker services,
                None for everything.
    """
    if target == "ray":
        svcs = [s for s in all_services() if s.type in (ServiceType.RAY, ServiceType.PROCESS)]
    elif target == "docker":
        svcs = [s for s in all_services() if s.type == ServiceType.DOCKER]
    elif target is None:
        svcs = all_services()
    else:
        console.print(f"[red]Unknown target: {target}[/red]")
        console.print("Use: boot, boot ray, or boot docker")
        return

    console.print(f"\n[bold cyan]Starting {len(svcs)} services...[/bold cyan]\n")

    for svc in svcs:
        console.print(f"  {svc.label}...", end=" ")
        success = start_service(svc)
        if success:
            console.print("[green]OK[/green]")
        else:
            console.print("[red]FAILED[/red]")

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
    """Stop services. If name is None, stop everything."""
    if name:
        svc = get(name)
        if not svc:
            console.print(f"[red]Unknown service: {name}[/red]")
            return
        console.print(f"Stopping [cyan]{svc.label}[/cyan]...")
        stop_service(svc)
        console.print("[green]Stopped[/green]")
        return

    console.print("\n[bold yellow]Stopping all services...[/bold yellow]\n")

    # Stop in reverse order
    svcs = list(reversed(all_services()))
    for svc in svcs:
        health = get_all_status().get(svc.name)
        if health and health.status != Status.STOPPED:
            console.print(f"  {svc.label}...", end=" ")
            stop_service(svc)
            console.print("[green]stopped[/green]")

    console.print()


def cmd_status() -> None:
    """Show status of all services."""
    statuses = get_all_status()

    table = Table(title="Tech Noir Services", show_header=True, header_style="bold")
    table.add_column("Service", style="cyan")
    table.add_column("Type")
    table.add_column("Status")
    table.add_column("Port")
    table.add_column("Detail")

    status_styles = {
        Status.HEALTHY: "[green]● running[/green]",
        Status.UNHEALTHY: "[yellow]● degraded[/yellow]",
        Status.STOPPED: "[red]○ stopped[/red]",
        Status.UNKNOWN: "[dim]? unknown[/dim]",
    }

    for svc in all_services():
        result = statuses.get(svc.name)
        if result:
            status_str = status_styles.get(result.status, str(result.status))
            detail = result.detail
        else:
            status_str = "[dim]? unknown[/dim]"
            detail = ""

        port_str = str(svc.port) if svc.port else "—"
        table.add_row(svc.label, svc.type.value, status_str, port_str, detail)

    console.print(table)


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
    else:
        console.print(f"[red]Unknown command: {command}[/red]")
        console.print("Commands: boot, up, stop, status")
        sys.exit(1)


if __name__ == "__main__":
    main()
