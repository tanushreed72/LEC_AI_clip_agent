# Topic-Based Video Clip Agent

An automated video-clipping agent that takes a video file and a list of
requested topics or keywords, uses speech-to-text to identify timestamped
speech segments, selects the most relevant contiguous sections for each
topic, and trims them to natural audio boundaries rather than arbitrary
timestamps.

The agent supports any number of requested topics in a single run, generating
one clip for each topic that can be matched to a transcript section. The
assessment demonstration uses three topics, exceeding the minimum requirement
of two distinct clips.

The agent is designed around **idempotent execution**: when the same video
and topic list are processed again, it recognises previously generated clips,
reuses the cached transcript, skips re-encoding, and produces no duplicate
files or partial overwrites.

The current implementation uses deterministic rule-based topic chaptering:
token-overlap keyword scoring combined with a monotonic dynamic-programming
pass. This was chosen to avoid an external LLM/API dependency and keep the
assessment reproducible. The architecture can be extended with an optional
LLM-based semantic chaptering step for improved matching when topic wording
differs substantially from the transcript.

## Pipeline

```
video -> speech-to-text (word/sentence timestamps)
      -> topic chaptering (decides the full time range each topic covers
         across the whole transcript)
      -> acoustic silence detection (independent ffmpeg pass)
      -> snap each chapter's start/end to the nearest real silence
      -> ffmpeg extraction (atomic write)
      -> persistent JSON state (atomic write)
```

The pipeline uses multiple external tools in sequence. **Faster-Whisper**
produces timestamped transcript segments, which are consumed by the topic
chaptering step. **FFmpeg** is then used for acoustic silence detection and for
the final clip extraction using the selected chapter boundaries.

### Idempotency

Each clip is identified by a content-addressable ID:
`sha1(video_hash : topic : raw_chapter_start : raw_chapter_end)`, computed
from the pre-snap chapter boundaries. Before generating a clip, the agent
checks whether that ID is already in `clip_state.json` **and** whether the
file it points to actually exists on disk — a state entry alone is never
trusted blindly. `reconcile_state()` also drops any state entry whose file
has gone missing and reports (without deleting) any untracked `.mp4` files
it finds. All writes (state file, clip file) go to a temp path first and
are atomically renamed into place, so a crash mid-write can't leave a
partial/corrupt file behind.

## Setup

### 1. System dependency: ffmpeg

`ffmpeg` and `ffprobe` must be on your `PATH`. They are **not** installable
via pip.

```bash
# macOS
brew install ffmpeg

# Debian/Ubuntu
sudo apt-get update && sudo apt-get install ffmpeg

# Windows: download a build from https://ffmpeg.org/download.html and add
# its bin/ folder to PATH
```

Verify with `ffmpeg -version` and `ffprobe -version`.

### 2. Python dependencies

Python 3.9+ recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` installs `faster-whisper` (the speech-to-text engine)
along with its own dependencies. If you'd rather install it directly:

```bash
pip install faster-whisper
```

The first transcription run will automatically download the `small`
Whisper model (a few hundred MB) and cache it locally.

### 3. Add your video

Place the input video under `videos/`, e.g. `videos/demo_clip.mp4`. Output
folders (`videos/`, `clips/`, `transcript_cache/`) are created
automatically on first run if they don't exist.

## Running it

**`clip_agent.py` is the file to run.** It's a plain Python script — no
notebook environment needed.

```bash
# Default: uses videos/demo_clip.mp4 and the three demo topics below,
# and runs the agent TWICE to demonstrate idempotency.

# Run the bundled demo twice to demonstrate idempotency:
python clip_agent.py

# Run your own video with your own topics:
python clip_agent.py videos/your_video.mp4 --topics "topic one" "topic two"

# Run your own video once:
python clip_agent.py videos/your_video.mp4 --topics "topic one" "topic two" --once
```

Default demo topics (phrased to match the sample coursework video):

```
welcome and demonstration of the coursework
deliverable 1 database normalisation and stored procedure
deliverable 2 business intelligence solution using SSAS
```

### What you should see

**Run 1** transcribes the video (cached afterwards), detects real silence
intervals, chapters the transcript into one time range per topic, snaps
each range's edges to nearby silence, and writes one `.mp4` per topic into
`clips/<video_stem>/`, plus `clip_state.json`.

**Run 2** (same input) reuses the cached transcript, re-derives the same
chapters and the same clip IDs, finds each clip already on disk, and skips
every single one — the script asserts that the set of files on disk is
identical before and after, and that both runs produced the same clip IDs.
You should see `[skip-cached] ... (no re-encode)` for every topic and
`OK: run 2 touched 0 new files.` at the end.

## Output layout

```
videos/                 your input video(s)
transcript_cache/        cached transcripts, keyed by video sha256
clips/<video_stem>/      output clips, one .mp4 per topic
clip_state.json          persistent record of every clip generated
```

## Demonstrated result

The agent was tested end-to-end on a real video.

- **Run 1:** transcribed the video and generated 3 topic-based clips.
- **Run 2:** reused the cached transcript and detected all 3 existing clips.
- No clips were re-encoded or duplicated on the second run.
- The clip IDs and output file set remained identical between runs.

The second run completed with:

`OK: run 2 touched 0 new files. Clip count stable at 3.`


## Known limitation

Topic chaptering here is rule-based (token-overlap keyword scoring per
sentence + a monotonic dynamic-programming pass), not semantic. On the
sample video it places the deliverable-1 → deliverable-2 boundary about 24
seconds later than the true transition, because the transcript says "D2"
and "BI solution" while the topic label spells out "deliverable 2 business
intelligence solution" — no shared vocabulary for the matcher to grab onto
right at the true boundary. Each topic's chaptering confidence is logged,
and a low score is a signal to rephrase the topic using words that
actually appear in the video, not something to silently trust.


## What I'd do next with more time

1. **Swap keyword chaptering for an LLM call.** Send the whole indexed,
   timestamped transcript plus the topic list in a single call and ask for
   the sentence-index range each topic covers. This directly fixes the
   limitation above (an LLM can tell "D2" means "deliverable 2" without
   shared vocabulary) and would let topic phrasing be arbitrary. Because
   the idempotency design keys clip IDs off the chaptering *output*, not a
   run counter, changing the chaptering method would correctly produce new
   clip IDs and new clips rather than silently reusing stale ones.
2. **Automated tests**: unit tests for `keyword_score`, the monotonic DP,
   and `snap_chapter_boundary`, plus an integration test that runs the
   full agent twice on a short fixture video in CI and asserts zero new
   files on the second run (currently this is only demonstrated
   interactively, not asserted automatically in a test suite).
3. **Better CLI ergonomics**: a topics file (one per line / JSON) instead
   of `--topics` on the command line, and flags for the silence-detection
   thresholds and snap window instead of editing constants.
4. **Batch mode** across a folder of videos — `run_agent` already keys
   everything by video hash, so this is mostly plumbing.
5. **Automatic topic discovery**: an LLM pass over the transcript could
   propose topics instead of requiring them upfront.

## Repo contents

- `clip_agent.py` — **the file to run.** Everything under "Running it"
  above refers to this script.
- `clip_agent.ipynb` — the same logic in notebook form, kept in the repo
  for demonstration and walkthrough purposes only (step-by-step narrative,
  inline explanations of each design decision). It is **not** required to
  run the agent — use `clip_agent.py` for that.
- `requirements.txt` — Python dependencies (ffmpeg is a separate system
  dependency, see Setup).
