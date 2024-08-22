#!/usr/bin/env bash

application_root="/srv"
application_data="$application_root/.application_data"

declare -A env_configs

env_configs[coffea,venv]="coffeaenv"
env_configs[coffea,container]="/cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-dask-almalinux8:2024.8.1-py3.10"

if [[ $(hostname) =~ "fnal" ]]; then
    env_configs[coffea,extras]="lpcqueue"
else
    env_configs[coffea,extras]=""
fi

env_configs[torch,venv]="cmsmlenv"
env_configs[torch,extras]="torch"
if nvidia-modprobe 2> /dev/null; then 
    env_configs[torch,apptainer_flags]="--nv"
fi
env_configs[torch,container]="/cvmfs/unpacked.cern.ch/registry.hub.docker.com/cmsml/cmsml:3.10"

env_configs[jaxenv,venv]="jaxenv"
env_configs[jaxenv,extras]="torch"
env_configs[jaxenv,empty]="true"
if nvidia-modprobe 2> /dev/null; then 
    env_configs[jaxenv,apptainer_flags]="--nv"
fi
#env_configs[jaxenv,container]="/cvmfs/unpacked.cern.ch/registry.hub.docker.com/cmsml/cmsml:3.10"
#env_configs[jaxenv,container]="/cvmfs/unpacked.cern.ch/registry.hub.docker.com/fnallpc/fnallpc-docker:pytorch-2.0.0-cuda11.7-cudnn8-runtime-singularity"
env_configs[jaxenv,container]="/cvmfs/unpacked.cern.ch/registry.hub.docker.com/fnallpc/fnallpc-docker:tensorflow-2.12.0-gpu-singularity"


function box_out()
{
    local box_h="#"
    local box_v="#"
    local s=("$@") b w
    for l in "${s[@]}"; do
        ((w<${#l})) && { b="$l"; w="${#l}"; }
    done
    echo " ${box_h}${b//?/${box_h}}${box_h}
${box_v} ${b//?/ } ${box_v}"
    for l in "${s[@]}"; do
        printf "${box_v} %*s ${box_v}\n" "-$w" "$l"
    done
    echo "${box_v} ${b//?/ } ${box_v}
 ${box_h}${b//?/${box_h}}${box_h}"
}



function activate_venv(){
    local config_name=$1
    local env=${env_configs[$config_name,venv]}
    source $application_data/virtual_envs/"$env"/bin/activate
    local localpath="$VIRTUAL_ENV$(python3 -c 'import sys; print(f"/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages")')"
    export PYTHONPATH=${localpath}:$PYTHONPATH
}

function version_info(){
    local packages_to_show=("coffea" "awkward" "uproot" "dask" "distributed" "dask-awkward" "dask-histogram" "fsspec_xrootd")
    local package_info="$(pip3 show "${packages_to_show[@]}")"
    for package in "${packages_to_show[@]}"; do
        awk -v package="$package" 'BEGIN{pat="Name: " package } a==1{printf("%s: %s\n", package, $2); exit} $0~pat{a++}' \
            <<< "$package_info"
    done  >&2 

}


function create_venv(){
    local config_name=$1
    local env_name=${env_configs[$config_name,venv]}
    local env_path=$application_data/virtual_envs/${env_name}
    local extras=${env_configs[$config_name,extras]}

    export TMPDIR=$(mktemp -d -p .)
    export PIP_DOWNLOAD_CACHE=".pipcache"
    
    trap 'rm -rf -- "$TMPDIR"' EXIT

    python3 -m venv --system-site-packages "$env_path"
    activate_venv $1

    if [[ "${env_configs[$config_name,empty]:-X}" == "true" ]]; then
        return
    fi
    printf "Created virtual environment %s\n" "$env_path"
    printf "Upgrading installation tools\n"
    python3 -m pip install pip  --upgrade
    python3 -m pip install setuptools pip wheel --upgrade
    printf "Installing project\n"
    if [[ -z $extras ]]; then
        python3 -m pip install -U -e . 
    else
        python3 -m pip install -U -e ".[$extras]" 
    fi

    # pip3 install ipython --upgrade
    python3 -m ipykernel install --name "$env_name" --prefix "$application_data/envlocal/$env/"
    # pip3 install -I boost-histogram
    #rm -rf "$env_path"/lib/*/site-packages/analyzer
    rm -rf $TMPDIR && unset TMPDIR
    sed -i "/PS1=/d" "$env_path"/bin/activate
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


    [ -z "$PS1" ] && return

    local config_name=$1
    local env=${env_configs[$config_name,venv]}

    mkdir -p $application_data/envlocal/$env

    HISTSIZE=50000
    HISTFILESIZE=20000
    export HISTCONTROL="erasedups:ignoreboth"
    export HISTTIMEFORMAT='%F %T '
    export HISTIGNORE=:"&:[ ]*:exit:ls:bg:fg:history:clear"
    shopt -s histappend
    shopt -s cmdhist &>/dev/null
    export HISTFILE=/srv/.bash_eternal_history
    export CONDOR_CONFIG="$application_data/.condor_config"
    export JUPYTER_PATH=$application_data/envlocal/$env/share/jupyter
    export JUPYTER_RUNTIME_DIR=$application_data/envlocal/$env/share/jupyter/runtime
    export JUPYTER_DATA_DIR=$application_data/envlocal/$env/share/jupyter
    export IPYTHONDIR=$application_data/envlocal/$env/ipython
    export MPLCONFIGDIR=$application_data/envlocal/$env/mpl
    export MPLBACKEND="Agg"
    #export LD_LIBRARY_PATH=/opt/conda/lib/:$LD_LIBRARY_PATH

    #export POETRY_HOME=/srv/.local/poetry
    #if [[ ! -d $POETRY_HOME ]]; then
    #    curl -sSL https://install.python-poetry.org | python3 -
    #fi
    

    if [[ ! -d $application_data/virtual_envs/$env ]]; then
        printf "Virtual environment does not exist, creating virtual environment\n"
        create_venv "$1"
    fi
    activate_venv "$1"

    PS1="${R}[APPTAINER\$( [[ ! -z \${VIRTUAL_ENV} ]] && echo "/\${VIRTUAL_ENV##*/}")]${M}[\t]${W}\u@${C}\h:${G}[\w]> ${NONE}"
    unset PROMPT_COMMAND

    local localpath="$VIRTUAL_ENV$(python3 -c 'import sys; print(f"/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages")')"
    if [[ -d $localpath/argcomplete ]]; then
        source $localpath/argcomplete/bash_completion.d/_python-argcomplete
        eval "$(register-python-argcomplete analyzer)"
    fi

    welcome_message="
             Single Stop Analysis Framework
........................................................
  Python version is $(python3 --version)
  Using environment $VIRTUAL_ENV
........................................................
  Run the following command to get started      
  $ python3 -m analyzer --help
"
    IFS=$'\n' read -rd '' -a split_welcome_message <<<"$welcome_message"
    mkdir -p .private
    box_out "${split_welcome_message[@]}"
}



function startup_with_container(){
    local rel_data=${application_data#$application_root/}
    mkdir -p $rel_data
    local in_apptainer=${APPTAINER_COMMAND:-false}
    local container=${env_configs[$1,container]}
    local apptainer_flags=${env_configs[$1,apptainer_flags]}
    printf "Running in container mode %s\n" "$container"
    if [ "$in_apptainer"  = false ]; then
        if command -v condor_config_val &> /dev/null; then
            printf "Cloning HTCondor configuration\n"
            condor_config_val  -summary > $rel_data/.condor_config
        fi
        if [[ -e $HISTFILE ]]; then
            apptainer_flags="$apptainer_flags --bind $HISTFILE:/srv/.bash_eternal_history"
        fi
        if [[ $(hostname) =~ "fnal" ]]; then
            apptainer_flags="$apptainer_flags --bind /uscmst1b_scratch/"
        fi
        if [[ $(hostname) =~ "umn" ]]; then
            apptainer_flags="$apptainer_flags --bind /local/cms/user/"
        fi
        if [[ ! -z "${X509_USER_PROXY}" ]]; then
            apptainer_flags="$apptainer_flags --bind ${X509_USER_PROXY%/*}"
        fi
        if [[ -d "$HOME/.globus" ]]; then
            apptainer_flags="$apptainer_flags --bind $HOME/.globus" # --bind $HOME/.rnd"
        fi


        apptainer exec \
                  --env "APPTAINER_WORKING_DIR=$PWD" \
                  --env "APPTAINER_IMAGE=$container" $apptainer_flags \
                  --bind /cvmfs \
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
