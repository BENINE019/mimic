#!/usr/bin/env bash
# Convert phone clips in clips/raw/ to 720p H.264 MP4 (clips/cNN.mp4)
# and extract mono 16 kHz narration audio (clips/audio/cNN.wav).
# Files are numbered in name order; rename the raw files first if order matters.
set -euo pipefail
mkdir -p clips/audio
i=0
for f in clips/raw/*; do
  [ -f "$f" ] || continue
  i=$((i + 1))
  n=$(printf "c%02d" "$i")
  echo "$f -> clips/$n.mp4"
  ffmpeg -loglevel error -y -i "$f" -vf "scale=-2:720,fps=30" -c:v libx264 -crf 23 -an "clips/$n.mp4"
  ffmpeg -loglevel error -y -i "$f" -vn -ac 1 -ar 16000 "clips/audio/$n.wav"
done
echo "Converted $i clips."
