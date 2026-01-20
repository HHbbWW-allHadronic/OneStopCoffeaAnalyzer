#!/usr/bin/env bash

container="/cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-dask-almalinux9:2025.10.2-py3.12"

apptainer exec -B $HOME,/uscmst1b_scratch,$(realpath $HOME),/uscms_data/d3/,$(realpath .),/cvmfs,/etc/condor/,/usr/local/bin/cmslpc-local-conf.py "$container" /bin/bash 
