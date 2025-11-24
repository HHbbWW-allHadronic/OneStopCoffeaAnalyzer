#!/usr/bin/env bash

container="/cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-dask-almalinux8:2025.1.0-py3.10"

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




function create_venv(){
    uv venv
    uv sync
}

function rcmode(){
    create_venv 
    ulimit -n 4096

    welcome_message="
             Single Stop Analysis Framework
  Run the following command to get started      
  $ uv run python3 -m analyzer --help
"
    IFS=$'\n' read -rd '' -a split_welcome_message <<<"$welcome_message"
    box_out "${split_welcome_message[@]}"
}



function startup_with_container(){
    local in_apptainer=${APPTAINER_COMMAND:-false}
    
    local apptainer_flags=""

    if [ "$in_apptainer"  = false ]; then
        export CONDOR_CONFIG=${LPC_CONDOR_CONFIG}
        apptainer_flags="$apptainer_flags --bind $HOME/.local"
        if [[ -e $HISTFILE ]]; then
            apptainer_flags="$apptainer_flags --bind $HISTFILE:/srv/.bash_history"
        fi
        if [[ $(hostname) =~ "fnal" ]]; then
            apptainer_flags="$apptainer_flags --bind /uscmst1b_scratch"
            apptainer_flags="$apptainer_flags --bind /cvmfs"
            apptainer_flags="$apptainer_flags --bind /uscms/homes/"
            apptainer_flags="$apptainer_flags --bind /storage"
            apptainer_flags="$apptainer_flags --bind /cvmfs/grid.cern.ch/etc/grid-security:/etc/grid-security"
            if [[ ! $(hostname) =~ "gpu" ]]; then
                apptainer_flags="$apptainer_flags --bind ${LPC_CONDOR_CONFIG}"
                apptainer_flags="$apptainer_flags --bind ${LPC_CONDOR_LOCAL}:${LPC_CONDOR_LOCAL}.orig"
                apptainer_flags="$apptainer_flags --bind .cmslpc-local-conf:${LPC_CONDOR_LOCAL}"
                cat <<EOF > .cmslpc-local-conf
#!/bin/bash
python3 ${LPC_CONDOR_LOCAL}.orig | grep -v "LOCAL_CONFIG_FILE"
EOF
                chmod u+x .cmslpc-local-conf
            fi
        fi
        if [[ $(hostname) =~ "umn" ]]; then
            apptainer_flags="$apptainer_flags --bind /local/cms/user/"
            apptainer_flags="$apptainer_flags --bind /cvmfs"
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
            --bind ${PWD} \
            "$container" /bin/bash
    else
        printf "Already in apptainer, nothing to do.\n"
    fi
}
function main(){
    startup_with_container 
}

main "$@"
