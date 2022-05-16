#!/usr/bin/env bash

#git clone https://github.com/Wickstrom/Quantus.git
pip install -r Quantus/requirements.txt
apt-get update
apt-get install ffmpeg libsm6 libxext6  -y
pip install termcolor

python3 in-pretrain.py