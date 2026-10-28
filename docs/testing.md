# SAM 3.1 tracking sanity check

Decides between the two hackathon ideas. If SAM 3.1 keeps object identities
while your hand passes over them, the "teach robots from phone video" idea is
viable. If it keeps losing them, stay with the laundry agent.

Runs in Ubuntu (WSL), using the venv you already set up.

## 1. Film the clip

- Phone on a tripod or propped against something, landscape, looking down at a table.
- Three everyday objects with different colours (for example: a red cup, a blue sock, a white bowl).
- 15 to 25 seconds. Pick each object up, move it somewhere else, put it down.
  Let your hand and arm fully cover at least one object at some point.
- Daylight or a bright lamp, no flash.

## 2. Put the clip where the script can see it

Inside Ubuntu:

```bash
mkdir -p ~/tracktest/clips
cp "/mnt/c/Users/ADMIN/Downloads/IMG_xxxx.MOV" ~/tracktest/clips/
```

Phone videos are usually HEVC and too large. Convert to 720p H.264 first
(cheaper to upload, and well under the API's frame limits):

```bash
sudo apt install -y ffmpeg
cd ~/tracktest
ffmpeg -i clips/IMG_xxxx.MOV -vf "scale=-2:720,fps=30" -c:v libx264 -crf 23 -an clips/test.mp4
```

## 3. Install and run

Copy the four files from this folder into `~/tracktest/`, then:

```bash
cd ~/tracktest
source ~/venv/bin/activate
pip install -r requirements.txt
export MODEL_API_KEY="paste-your-key-from-dev.meta.ai"

python track_test.py clips/test.mp4 \
  --concept "red cup" --concept "blue sock" --concept "white bowl" \
  --output out/test
```

One API request per concept (that is how the SAM API works: one short noun
phrase per request). A 20 s, 30 fps clip is about 600 frames, so three
concepts cost well under one dollar of your credits.

The script prints a table like:

```
concept           id   first  last  seen  span  gap  verdict
red cup           0        0   598   571   599   11  OK
blue sock         0        0   240   233   241    4  LOST (gap > 15)
blue sock         1      291   598   301   308    2  LOST (gap > 15)
```

and writes `out/test/overlay.mp4` with coloured masks and `concept #id` labels.

## 4. How to read the result

- **PASS:** each object has one id for the whole clip, gaps of a few frames at most.
  Switch to the video-to-robot-demo idea.
- **FAIL:** an object gets a new id after your hand covers it (like `blue sock` above),
  or the mask jumps to a different object. Stay with the laundry agent, or budget a
  day for re-identification logic before committing.

Keep the overlay video either way; it is useful footage for the demo.

## Notes

- `sam_api.py` is Meta's own helper from the Model API cookbook (MIT licence).
- Reuse an upload with `--file-id file-...` (printed after the first run) to try
  different concept wordings without uploading again.
- If a concept returns nothing, try a plainer phrase: "cup" instead of "red ceramic mug".
