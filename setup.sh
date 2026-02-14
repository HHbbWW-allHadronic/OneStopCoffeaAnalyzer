#!/usr/bin/env bash


container="/cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-dask-almalinux9:2025.10.2-py3.12"

if [[ -n "$SINGULARITY_NAME" ]] || [[ -n "$APPTAINER_NAME" ]]; then
    echo "Inside container: $APPTAINER_NAME"
    if [ -f "$HOME/.bashrc" ]; then
        . "$HOME/.bashrc"
    fi

    if ! command -v uv &> /dev/null; then
        echo "Error: 'uv' is not installed or not in PATH."
        echo "Please install 'uv' and try again."
        exit 1
    fi

    echo "Found uv: $(uv --version)"


    SYNC_CMD_EXTRAS=""
    if [[ "$IS_LPC" == "true" ]]; then
        echo "LPC environment detected. Enabling lpc and condor extras."
        SYNC_CMD_EXTRAS="$SYNC_CMD_EXTRAS --extra lpc --extra condor"
    fi

    if [ ! -d ".venv" ]; then
        echo "Creating virtual environment..."
        uv venv --clear --no-managed-python --relocatable --system-site-packages --link-mode copy
        echo "Syncing dependencies..."
        echo "Running: uv sync --no-managed-python --link-mode copy $SYNC_CMD_EXTRAS"
        uv sync --no-managed-python --link-mode copy $SYNC_CMD_EXTRAS
    fi
    
    exec /bin/bash 
else
    HOST=$(hostname)
    IS_LPC="false"
    if [[ "$HOST" == *"cmslpc"* ]]; then
        echo "Detected LPC host."
        IS_LPC="true"
    fi
    SCRIPT_PATH=$(realpath "$0")
    ARGS=""
    if [[ "$IS_LPC" == "true" ]]; then
        ARGS="-B $HOME,/uscmst1b_scratch,$(realpath "$HOME"),/uscms_data,/cvmfs,/etc/condor/,/usr/local/bin/cmslpc-local-conf.py,$(realpath .):$PWD"
    else
        ARGS="-B $(realpath .):$PWD,$(realpath "$HOME")" 
    fi

    echo "Entering container..."
    export IS_LPC
    apptainer exec $ARGS "$container" /bin/bash  "$SCRIPT_PATH"
fi 
