#!/usr/bin/env bash

application_root="/srv"
application_data="$application_root/.application_data"
container="/cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-dask-almalinux8:2024.8.1-py3.10"
env_extras="lpcqueue"
virtual_env_path="$application_data/venv"

LPC_CONDOR_CONFIG=/etc/condor/config.d/01_cmslpc_interactive
LPC_CONDOR_LOCAL=/usr/local/bin/cmslpc-local-conf.py



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
    source $virtual_env_path/bin/activate
    local localpath="$VIRTUAL_ENV$(python3 -c 'import sys; print(f"/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages")')"
    export PYTHONPATH=${localpath}:$PYTHONPATH
}

function version_info(){
    local packages_to_show=("coffea" "awkward" "uproot" "dask" "distributed" "dask-awkward" "dask-histogram" "fsspec_xrootd")
    local package_info="$(python3 -m pip show "${packages_to_show[@]}")"
    for package in "${packages_to_show[@]}"; do
        awk -v package="$package" 'BEGIN{pat="Name: " package } a==1{printf("%s: %s\n", package, $2); exit} $0~pat{a++}' \
            <<< "$package_info"
    done  >&2 

}


function create_venv(){

    export TMPDIR=$(mktemp -d -p .)
    export PIP_DOWNLOAD_CACHE=".pipcache"
    
    trap 'rm -rf -- "$TMPDIR"' EXIT

    python3 -m venv --system-site-packages "$virtual_env_path"
    activate_venv 

    printf "Created virtual environment %s\n" "$env_path"
    printf "Upgrading installation tools\n"

    python3 -m pip install pip  --upgrade
    python3 -m pip install setuptools pip wheel --upgrade
    printf "Installing project\n"
    if [[ -z $env_extras ]]; then
        python3 -m pip install -U -e . 
    else
        python3 -m pip install -U -e ".[$env_extras]" 
    fi

    # pip3 install ipython --upgrade
    python3 -m ipykernel install --name "$env_name" --prefix "$application_data/local/"
    rm -rf $TMPDIR && unset TMPDIR
    sed -i "/PS1=/d" "$virtual_env_path"/bin/activate
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


    mkdir -p $application_data/local


    HISTSIZE=50000
    HISTFILESIZE=20000
    export HISTCONTROL="erasedups:ignoreboth"
    export HISTTIMEFORMAT='%F %T '
    export HISTIGNORE=:"&:[ ]*:exit:ls:bg:fg:history:clear"
    export HISTFILE=/srv/.bash_history
    shopt -s histappend
    shopt -s cmdhist &>/dev/null
    export JUPYTER_PATH=$application_data/local/share/jupyter
    export JUPYTER_RUNTIME_DIR=$application_data/local/share/jupyter/runtime
    export JUPYTER_DATA_DIR=$application_data/local/share/jupyter
    export IPYTHONDIR=$application_data/local/ipython
    export MPLCONFIGDIR=$application_data/local/mpl
    export MPLBACKEND="Agg"
    #export LD_LIBRARY_PATH=/opt/conda/lib/:$LD_LIBRARY_PATH

    #export POETRY_HOME=/srv/.local/poetry
    #if [[ ! -d $POETRY_HOME ]]; then
    #    curl -sSL https://install.python-poetry.org | python3 -
    #fi
    

    if [[ ! -d $virtual_env_path ]]; then
        printf "Virtual environment does not exist, creating virtual environment in $virtual_env_path\n"
        create_venv 
    fi
    activate_venv 

    PS1="${R}[APPTAINER\$( [[ ! -z \${VIRTUAL_ENV} ]] && echo "/\${VIRTUAL_ENV##*/}")]${M}[\t]${W}\u@${C}\h:${G}[\w]> ${NONE}"
    unset PROMPT_COMMAND

    PROMPT_COMMAND="history -a;$PROMPT_COMMAND"

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
    local apptainer_flags=""

    cat <<EOF > .cmslpc-local-conf
#!/bin/bash
python3 ${LPC_CONDOR_LOCAL}.orig | grep -v "LOCAL_CONFIG_FILE"
EOF
    chmod u+x .cmslpc-local-conf

    if [ "$in_apptainer"  = false ]; then
        # if command -v condor_config_val &> /dev/null; then
        #     condor_config_val  -summary > .condor_config
        # fi
        export CONDOR_CONFIG=${LPC_CONDOR_CONFIG}
        if [[ -e $HISTFILE ]]; then
            apptainer_flags="$apptainer_flags --bind $HISTFILE:/srv/.bash_history"
        fi
        if [[ $(hostname) =~ "fnal" ]]; then
            apptainer_flags="$apptainer_flags --bind /uscmst1b_scratch"
            apptainer_flags="$apptainer_flags --bind /cvmfs"
            apptainer_flags="$apptainer_flags --bind /cvmfs/grid.cern.ch/etc/grid-security:/etc/grid-security"
            apptainer_flags="$apptainer_flags --bind ${LPC_CONDOR_CONFIG}"
            apptainer_flags="$apptainer_flags --bind ${LPC_CONDOR_LOCAL}:${LPC_CONDOR_LOCAL}.orig"
            apptainer_flags="$apptainer_flags --bind .cmslpc-local-conf:${LPC_CONDOR_LOCAL}"
        fi
        if [[ $(hostname) =~ "umn" ]]; then
            apptainer_flags="$apptainer_flags --bind /local/cms/user/"
            apptainer_flags="$apptainer_flags --bind /cvmfs"
            apptainer_flags="$apptainer_flags --bind /cvmfs/grid.cern.ch/etc/grid-security:/etc/grid-security"
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
            --bind ${PWD}:/srv \
            --pwd /srv "$container" /bin/bash \
            --rcfile <(printf "source setup.sh bashrc")
    else
        printf "Already in apptainer, nothing to do.\n"
    fi
}

function start_jupyter(){
    local port=${1:-8999}
    python3 -m jupyter lab --no-browser --port "$port" --allow-root
}

function main(){
    local mode="${1:-apptainer}"
    case "$mode" in
        apptainer )
            startup_with_container 
            ;;
        bashrc )
            rcmode 
            ;;
        * )
            printf "Unknown mode\n"
            return 1
            ;;
    esac
}

main "$@"
