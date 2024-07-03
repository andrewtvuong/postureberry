#!/bin/sh

sudo dpkg -i /home/pi/posture/libedgetpu1-std_16.0tf2.13.1-2.bookworm_arm64.deb
sudo apt-get install -f
pip install -r requirements.txt --break-system-packages
