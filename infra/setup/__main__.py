"""Tech Noir Ray — Infrastructure setup.

Usage:
    python -m infra.setup system          # Check system prerequisites
    python -m infra.setup system --fix    # Apply system fixes
    python -m infra.setup all             # Set up bare-metal tool venvs
    python -m infra.setup docker          # Build Docker worker images
    python -m infra.setup ace-step        # Set up specific tool
"""

import sys


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "system":
        from infra.setup.system import main as system_main
        sys.exit(system_main())
    else:
        from infra.setup.venvs import main as venvs_main
        venvs_main()


main()
