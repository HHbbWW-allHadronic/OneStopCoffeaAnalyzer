#!/usr/bin/env bash

source /cvmfs/sft.cern.ch/lcg/views/LCG_108/x86_64-el9-gcc14-opt/setup.sh
set -eux 
g++ -O3 -std=c++20 $(root-config --cflags --libs) scripts/signal_splitting/split_signals.cpp
./a.out $1 $2
xrdcp -r split_signals/ root://cmseos.fnal.gov///store/user/ckapsiak/SingleStop/raw_official_samples/
echo "JOB COMPLETED SUCCESSFULL"
