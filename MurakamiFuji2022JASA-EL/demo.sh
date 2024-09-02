#!/bin/bash

mkdir ./stim
mkdir ./data

echo "Generate signal."
python generate_stim.py

echo "Compute cochlear responses of the 1D model for a pure tone"
python Simulation_PureTone_Cochlea1D.py
echo "Compute cochlear responses of the 2D model for a pure tone"
python Simulation_PureTone_Cochlea2D.py

echo "Compute cochlear responses of the 1D model for two tones"
python Simulation_TwoTone_Cochlea1D.py
echo "Compute cochlear responses of the 2D model for two tones"
python Simulation_TwoTone_Cochlea2D.py

echo "Compare Frequency and Suppression Tuning Curves"
python plot_compare_TC_Cochlea.py

#rm -Rf ./stim
#rm -Rf ./data