#!/bin/bash
PI_USER="pi"
PI_HOST=192.168.43.229
PI_DIR="~/thermal_photos/"
MAC_DIR="$HOME/research_project/solar-thermal-pipeline/data/incoming/"


echo "Syncing..."

while true; do
 rsync -avz --quiet "${PI_USER}@${PI_HOST}":"${PI_DIR}" "$MAC_DIR"
 echo "Done"
 sleep 10
done
