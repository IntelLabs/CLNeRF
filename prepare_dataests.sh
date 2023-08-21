#!/bin/sh
cd scripts/data_prepare
bash prepare_nerfpp.sh
bash prepare_SynthNeRF.sh
bash prepare_WAT.sh
cd ../..