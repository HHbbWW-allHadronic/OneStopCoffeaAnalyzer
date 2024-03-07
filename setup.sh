#!/usr/bin/env bash

declare -A env_configs
env_configs[coffea,venv]="coffeaenv"
env_configs[coffea,container]="/cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-dask:latest-py3.10"
#env_configs[coffea,container]="/cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-dask:l"
#env_configs[coffea,container]="/cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-base:v2024.1.2"
#env_configs[coffea,container]="/cvmfs/unpacked.cern.ch/registry.hub.docker.com/cmssw/cs9:x86_64"
#env_configs[coffea,container]="/cvmfs/unpacked.cern.ch/registry.hub.docker.com/cmsml/cmsml:3.10"
if [[ $(hostname) =~ "fnal" ]]; then
    env_configs[coffea,extras]="lpcqueue"
else
    env_configs[coffea,extras]=""
fi

env_configs[torch,venv]="cmsmlenv"
env_configs[torch,extras]="torch"
env_configs[torch,apptainer_flags]="--nv"
env_configs[torch,container]="/cvmfs/unpacked.cern.ch/registry.hub.docker.com/cmsml/cmsml:3.10"
#env_configs[torch,container]="/cvmfs/unpacked.cern.ch/registry.hub.docker.com/fnallpc/fnallpc-docker:pytorch-2.0.0-cuda11.7-cudnn8-runtime-singularity"


function activate_venv(){
    local env=$1
    source "$env"/bin/activate
    local localpath="$VIRTUAL_ENV$(python3 -c 'import sys; print(f"/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages")')"
    export PYTHONPATH=${localpath}:$PYTHONPATH
    printf "Python path is %s\n" "$PYTHONPATH"
    printf "Python bin is %s\n" "$(which python)"
    printf "Python version is %s\n" "$(python3 --version)"
    printf "Using environment %s\n" "$VIRTUAL_ENV"
}

function version_info(){
    local packages_to_show=("coffea" "awkward" "dask-awkward" "dask")
    local package_info="$(pip3 show ${packages_to_show[@]} )"
    for package in ${packages_to_show[@]}; do
        awk -v package="$package" 'BEGIN{pat=package "$" } a==1{printf("%s: %s\n", package, $2); exit} $0~pat{a++}' <<< "$package_info"
    done  >&2 
}

function create_venv(){
    local env=$1
    local extras=$2
    export TMPDIR=$(mktemp -d -p .)
    
    trap 'rm -rf -- "$TMPDIR"' EXIT

    python3 -m venv --system-site-packages "$env"
    activate_venv "$env"


    printf "Created virtual environment %s\n" "$env"
    printf "Upgrading installation tools\n"
    python3 -m pip install setuptools pip wheel --upgrade
    printf "Installing project\n"
    if [[ -z $extras ]]; then
        python3 -m pip install .
    else
        python3 -m pip install ".[$extras]"
    fi

    pip3 install ipython --upgrade
    python3 -m ipykernel install --user --name "$env"
    pip3 install -I boost-histogram

    rm -rf $env/lib/*/site-packages/analyzer

    rm -rf $TMPDIR && unset TMPDIR

    sed -i "/PS1=/d" "$env"/bin/activate
    #for file in $NAME/bin/*; do
    #    sed -i '1s/#!.*python$/#!\/usr\/bin\/env python3/' "$file"
    #done
    #sed -i '40s/.*/VIRTUAL_ENV="$(cd "$(dirname "$(dirname "${BASH_SOURCE[0]}" )")" \&\& pwd)"/' $ENVNAME/bin/activate
    trap - EXIT
}




function rcmode(){
    K="\[\033[0;30m\]"    # black
    R="\[\033[0;31m\]"    # red
    G="\[\033[0;32m\]"    # green
    Y="\[\033[0;33m\]"    # yellow
    B="\[\033[0;34m\]"    # blue
    M="\[\033[0;35m\]"    # magenta
    C="\[\033[0;36m\]"    # cyan
    W="\[\033[0;37m\]"    # white
    EMK="\[\033[1;30m\]"
    EMR="\[\033[1;31m\]"
    EMG="\[\033[1;32m\]"
    EMY="\[\033[1;33m\]"
    EMB="\[\033[1;34m\]"
    EMM="\[\033[1;35m\]"
    EMC="\[\033[1;36m\]"
    EMW="\[\033[1;37m\]"
    BGK="\[\033[40m\]"
    BGR="\[\033[41m\]"
    BGG="\[\033[42m\]"
    BGY="\[\033[43m\]"
    BGB="\[\033[44m\]"
    BGM="\[\033[45m\]"
    BGC="\[\033[46m\]"
    BGW="\[\033[47m\]"
    NONE="\[\033[0m\]"    # unsets color to term's fg color
    local env=${env_configs[$1,venv]}
    local extras=${env_configs[$1,extras]}
    if [[ ! -d $env ]]; then
        printf "Virtual environment does not exist, creating virtual environment\n"
        create_venv "$env" "$extras"
    fi
    activate_venv "$env"

    [ -z "$PS1" ] && return

    PS1="${R}[APPTAINER\$( [[ ! -z \${VIRTUAL_ENV} ]] && echo "/\${VIRTUAL_ENV##*/}")]${M}[\t]${W}\u@${C}\h:${G}[\w]> ${NONE}"

    unset PROMPT_COMMAND

    HISTSIZE=50000
    HISTFILESIZE=20000
    export HISTCONTROL="erasedups:ignoreboth"
    export HISTTIMEFORMAT='%F %T '
    export HISTIGNORE=:"&:[ ]*:exit:ls:bg:fg:history:clear"
    shopt -s histappend
    shopt -s cmdhist &>/dev/null
    export HISTFILE=~/.bash_history
    export CONDOR_CONFIG="/srv/.condor_config"
    export JUPYTER_PATH=/srv/.local/$env/.jupyter
    export JUPYTER_RUNTIME_DIR=/srv/.local/$env/share/jupyter/runtime
    export JUPYTER_DATA_DIR=/srv/.local/$env/share/jupyter
    export IPYTHONDIR=/srv/.local/$env/.ipython
    export MPLCONFIGDIR=/srv/.local/$env/.mpl
    #export LD_LIBRARY_PATH=/opt/conda/lib/:$LD_LIBRARY_PATH
}


function startup_with_container(){
    local in_apptainer=${APPTAINER_COMMAND:-false}
    local container=${env_configs[$1,container]}
    local apptainer_flags=${env_configs[$1,apptainer_flags]}
    printf "Running in container mode %s\n" "$container"
    if [ "$in_apptainer"  = false ]; then
        if command -v condor_config_val &> /dev/null; then
            printf "Cloning HTCondor configuration\n"
            condor_config_val  -summary > .condor_config
        fi
        if [[ -e $HISTFILE ]]; then
            apptainer_flags="$apptainer_flags --bind $HISTFILE:/srv/.bash_eternal_history"
        fi
        apptainer exec \
                  --env "APPTAINER_WORKING_DIR=$PWD" \
                  --env "APPTAINER_IMAGE=$container" $apptainer_flags \
                  --bind /uscmst1b_scratch/ \
                  --bind /cvmfs \
                  --bind ${X509_USER_PROXY%/*} \
                  --bind ${PWD}:/srv \
                  --pwd /srv "$container" /bin/bash \
                  --rcfile <(printf "source setup.sh '$1' bashrc")
    else
        printf "Already in apptainer, nothing to do.\n"
    fi
}


function start_jupyter(){
    local port=${1:-8999}
    python3 -m jupyter lab --no-browser --port "$port" --allow-root
}



function main(){
    local config="$1"
    local mode="${2:-apptainer}"
    if [[ -z ${env_configs[$config,venv]} ]]; then
        printf "Not a valid environment %s\n" "$config"
        return 1
    fi
    case "$mode" in
        apptainer )
            startup_with_container $config
            ;;
        bashrc )
            rcmode $config
            ;;
        * )
            printf "Unknown mode\n"
            return 1
            ;;
    esac
}

main "$@"
