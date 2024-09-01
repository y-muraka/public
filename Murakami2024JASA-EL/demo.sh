#!/bin/bash

python resample_wavfile.py

mkdir tmp
mkdir tmp/fft
mkdir tmp/direct

cd fft
make

for num_seg in 256 512 1024
do
    for Lp in 0 40 80
    do
        echo FFT $num_seg $Lp
        ./CochlearModel_1D_fft ../wav/converted/1000Hz.wav $num_seg $Lp
    done
done

cd ..

cd direct
make

for num_seg in 256 512 1024
do
    for Lp in 0 40 80
    do
        echo Direct $num_seg $Lp
        ./CochlearModel_1D_direct ../wav/converted/1000Hz.wav $num_seg $Lp
    done
done

cd ..

python demo_plot.py