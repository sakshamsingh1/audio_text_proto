#!/bin/sh

pip install -r requirements.txt

conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
pip uninstall numba
pip install numba 
pip install chardet