#!/bin/bash

mkdir tmp

echo "Computing cochlear response for a tone"
python CochlearModel3D_semiFFT.py
python anim_BMmotion.py

rm -Rf ./tmp