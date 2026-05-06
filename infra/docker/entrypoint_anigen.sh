#!/bin/bash
set -e

# Link models to AniGen's expected working directory
if [ -d /models/ckpts/ckpts ] && [ ! -e /opt/anigen/ckpts ]; then
    ln -s /models/ckpts/ckpts /opt/anigen/ckpts
    echo "Linked /opt/anigen/ckpts -> /models/ckpts/ckpts"
fi

exec python /opt/api.py "$@"
