"""Tech Noir Ray — Infrastructure setup.

Usage:
    python -m infra.setup system          # Check system prerequisites
    python -m infra.setup system --fix    # Apply system fixes
    python -m infra.setup bootstrap       # Check host bootstrap (k3s + Flux)
    python -m infra.setup bootstrap --fix # Apply host bootstrap
    python -m infra.setup sops            # Check SOPS secrets
    python -m infra.setup sops --fix      # Generate AGE key + encrypt secrets
    python -m infra.setup all             # Set up bare-metal tool venvs
    python -m infra.setup docker          # Build Docker worker images
    python -m infra.setup ace-step        # Set up specific tool
"""

import sys


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    cmd = sys.argv[1]

    if cmd == "system":
        from infra.setup.system import main as system_main
        sys.exit(system_main())
    elif cmd == "bootstrap":
        from infra.setup.bootstrap import main as bootstrap_main
        sys.exit(bootstrap_main())
    elif cmd == "sops":
        from infra.setup.sops import main as sops_main
        sys.exit(sops_main())
    else:
        from infra.setup.venvs import main as venvs_main
        venvs_main()


main()
