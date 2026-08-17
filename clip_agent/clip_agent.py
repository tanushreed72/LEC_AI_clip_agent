#!/usr/bin/env python3
"""
Topic-Based Video Clip Agent
=============================

Pipeline: video -> speech-to-text (word/sentence-level timestamps) ->
topic chaptering (rule-based) -> acoustic silence boundary snap ->
ffmpeg extraction -> persistent state.

Given a video file and a list of topics/keywords, this agent:
  1. Transcribes the video with faster-whisper (word + sentence timestamps),
     caching the transcript on disk keyed by the video's sha256 so an
     unchanged file is never re-transcribed.
  2. Detects real silence intervals in the source audio with
     `ffmpeg -af silencedetect` (independent of the transcript).
  3. Chapters the transcript: decides the full contiguous time range each
     topic covers, using token-overlap keyword scoring per sentence
     resolved with a monotonic dynamic-programming pass (a topic is
     assumed to be discussed once, in the order given).
  4. Snaps each chapter's start/end to the nearest real silence interval,
     so the cut lands on a natural pause rather than a hard timestamp.
  5. Extracts each clip with ffmpeg (re-encoded, not stream-copied, so the
     cut lands exactly on the chosen boundary) and writes it atomically
     (`.part` file + rename).
  6. Records every clip in a JSON state file, keyed by a content-addressable
     ID (sha1 of video hash + topic + raw chapter boundaries). Re-running
     the agent on the same video/topics recomputes the same IDs, sees the
     files already exist, and skips re-encoding entirely -- this is the
     idempotency guarantee: run it twice, get zero new files.

See README.md for setup and run instructions, and for the documented
limitation of the rule-based chaptering approach.
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# --------------------------------------------------------------------------
# 0. Paths / setup
# --------------------------------------------------------------------------

VIDEO_DIR = Path("videos")
OUTPUT_DIR = Path("clips")
STATE_FILE = Path("clip_state.json")
TRANSCRIPT_CACHE_DIR = Path("transcript_cache")

for d in (VIDEO_DIR, OUTPUT_DIR, TRANSCRIPT_CACHE_DIR):
    d.mkdir(exist_ok=True)


def sha256_file(path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# --------------------------------------------------------------------------
# 1. Speech-to-text with word- and sentence-level timestamps
# --------------------------------------------------------------------------
# `faster-whisper` gives per-word and per-sentence start/end times.
# Transcripts are cached on disk keyed by the video file's sha256, so
# re-running on an unchanged file never re-transcribes -- the expensive
# step, and the first idempotency win.
#
# The `faster_whisper` import is lazy (only happens on a cache miss), so a
# cached re-run still works in an environment where the package isn't
# installed, instead of failing on an unconditional top-level import.


def get_model():
    from faster_whisper import WhisperModel
    return WhisperModel("small", device="cpu", compute_type="int8")


def transcribe(video_path: Path) -> dict:
    video_hash = sha256_file(video_path)
    cache_path = TRANSCRIPT_CACHE_DIR / f"{video_hash}.json"
    if cache_path.exists():
        print(f"[cache] transcript already exists for {video_path.name} -> reusing, no re-transcription")
        return json.loads(cache_path.read_text())

    print(f"[whisper] transcribing {video_path.name} ...")
    model = get_model()
    segments, info = model.transcribe(str(video_path), word_timestamps=True)

    words, seg_list = [], []
    for seg in segments:
        seg_list.append({"start": seg.start, "end": seg.end, "text": seg.text.strip()})
        for w in (seg.words or []):
            if w.word.strip():
                words.append({"word": w.word.strip(), "start": w.start, "end": w.end})

    result = {
        "video_hash": video_hash,
        "video_path": str(video_path),
        "language": info.language,
        "words": words,
        "segments": seg_list,
    }
    cache_path.write_text(json.dumps(result, indent=2))
    return result


# --------------------------------------------------------------------------
# 2. Acoustic silence detection (ground truth for boundary snapping)
# --------------------------------------------------------------------------
# Independent of Whisper: `ffmpeg silencedetect` finds real silence
# intervals in the source audio. Used at the end of the pipeline to snap a
# topic's (coarse) start/end time onto the nearest real pause, so the final
# cut doesn't land mid-word.


def find_silences(video_path, noise_db=-30, min_dur=0.08) -> List[Tuple[float, float]]:
    """Run ffmpeg's silencedetect filter and return (silence_start, silence_end) pairs in seconds."""
    cmd = ["ffmpeg", "-i", str(video_path), "-af",
           f"silencedetect=noise={noise_db}dB:d={min_dur}", "-f", "null", "-"]
    log = subprocess.run(cmd, capture_output=True, text=True).stderr
    starts = [float(x) for x in re.findall(r"silence_start:\s*(-?[\d.]+)", log)]
    ends = [float(x) for x in re.findall(r"silence_end:\s*(-?[\d.]+)", log)]
    return list(zip(starts, ends))


def snap_chapter_boundary(t: float, silences: List[Tuple[float, float]], window: float = 6.0) -> float:
    """
    Snap a coarse chapter boundary to the MIDPOINT of the nearest real silence
    interval within `window` seconds. Chapter boundaries come from sentence-
    level timestamps (already natural sentence breaks), so they're usually
    close to a real pause already; this snap removes the last bit of drift so
    the cut lands cleanly on silence rather than a fraction of a second into
    the next/previous word. `window` is wide (6s) because a *chapter*
    boundary can be off by a couple of seconds even when it's correctly
    chosen -- sentence timestamps aren't perfectly tight to the pause the
    way a single word's is.
    """
    best, best_d = None, window + 1e-9
    for s, e in silences:
        mid = (s + e) / 2
        d = abs(mid - t)
        if d <= window and d < best_d:
            best, best_d = mid, d
    return best if best is not None else t


# --------------------------------------------------------------------------
# 3. Topic chaptering: decide the FULL span each topic covers
# --------------------------------------------------------------------------
# Rule-based chaptering: token-overlap keyword scoring per sentence,
# resolved with a monotonic DP: each sentence is assigned to whichever
# topic scores highest, under the constraint that the topic index can only
# stay the same or advance by one as time moves forward (topics are
# assumed to occur once each, in the order given). This is a meaningfully
# better rule-based approach than "classify each sentence independently"
# (which flip-flops badly on shared vocabulary), but it is still lexical,
# not semantic -- see README.md for the documented limitation and the
# suggested LLM-based next step.

STOPWORDS = {"a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with", "this", "that",
             "is", "are", "we", "our", "its", "it's", "as", "by", "from", "which", "was", "were"}


def _tokenize(s: str) -> List[str]:
    return [t for t in re.findall(r"[a-z0-9']+", s.lower()) if t not in STOPWORDS]


def keyword_score(text: str, topic: str) -> float:
    """Token-overlap score (fraction of the topic's words found in the text),
    plus a bonus if the topic appears as a literal substring."""
    text_tokens = set(_tokenize(text))
    topic_tokens = _tokenize(topic)
    if not topic_tokens:
        return 0.0
    hits = sum(1 for t in topic_tokens if t in text_tokens)
    coverage = hits / len(topic_tokens)
    phrase_bonus = 1.0 if topic.lower() in text.lower() else 0.0
    return coverage + phrase_bonus


def rule_based_chapter_topics(segments: List[Dict], topics: List[str]):
    """Rule-based chaptering: monotonic-DP keyword chaptering. Returns
    (spans, confidence) where confidence is each topic's average per-sentence
    keyword_score over its assigned span -- a low value is a signal the match
    is weak and the caller should warn rather than silently trust it."""
    n, K = len(segments), len(topics)
    emission = [[keyword_score(seg["text"], t) for t in topics] for seg in segments]
    NEG = float("-inf")
    dp = [[NEG] * K for _ in range(n)]
    back = [[None] * K for _ in range(n)]
    dp[0][0] = emission[0][0]
    for i in range(1, n):
        for j in range(K):
            best_prev, best_val = None, NEG
            if dp[i - 1][j] > best_val:
                best_val, best_prev = dp[i - 1][j], j
            if j > 0 and dp[i - 1][j - 1] > best_val:
                best_val, best_prev = dp[i - 1][j - 1], j - 1
            if best_prev is not None:
                dp[i][j] = best_val + emission[i][j]
                back[i][j] = best_prev
    end_j = max(range(K), key=lambda j: dp[n - 1][j])
    labels = [None] * n
    labels[n - 1] = end_j
    for i in range(n - 1, 0, -1):
        labels[i - 1] = back[i][labels[i]]

    spans, confidence = {}, {}
    i = 0
    while i < n:
        j = labels[i]
        start_i = i
        while i < n and labels[i] == j:
            i += 1
        end_i = i - 1
        topic = topics[j]
        span = (segments[start_i]["start"], segments[end_i]["end"])
        if topic not in spans or (span[1] - span[0]) > (spans[topic][1] - spans[topic][0]):
            spans[topic] = span
            confidence[topic] = sum(emission[k][j] for k in range(start_i, end_i + 1)) / (end_i - start_i + 1)
    return spans, confidence


def chapter_topics(segments: List[Dict], topics: List[str]) -> Dict[str, Tuple[float, float]]:
    print("[chapter] using rule-based monotonic keyword chaptering")
    spans, confidence = rule_based_chapter_topics(segments, topics)
    for t, c in confidence.items():
        if c < 0.3:
            print(f"[warn] low keyword-confidence ({c:.2f}) chaptering topic {t!r} -- "
                  f"topic wording may differ from transcript wording")
    return spans


# --------------------------------------------------------------------------
# 4. Persistent state + idempotency
# --------------------------------------------------------------------------
# - Clip ID = sha1(video_hash : topic : raw_start : raw_end), computed from
#   the pre-snap chapter boundaries, so identity is stable even if
#   silence-snap parameters are retuned later.
# - State writes are atomic (.tmp + os.replace).
# - Clip files are written to .part then atomically renamed.
# - Before skipping a "known" clip, we check the file actually exists on
#   disk -- state saying "done" isn't trusted blindly.
# - reconcile_state() drops any state entry whose file is missing, and
#   reports (without deleting) any .mp4 on disk that isn't tracked by state.


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}


def save_state(state: dict):
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True))
    os.replace(tmp, STATE_FILE)


def make_clip_id(video_hash: str, topic: str, start: float, end: float) -> str:
    key = f"{video_hash}:{topic.lower().strip()}:{round(start, 2)}:{round(end, 2)}"
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def reconcile_state():
    """Detect drift between state.json and what's actually on disk."""
    state = load_state()
    changed = False
    for clip_id, entry in list(state.items()):
        if not Path(entry["output_path"]).exists():
            print(f"[reconcile] state points to missing file, dropping entry: {entry['output_path']}")
            del state[clip_id]
            changed = True
    known = {Path(e["output_path"]).name for e in state.values()}
    on_disk = {p.name for p in OUTPUT_DIR.glob("**/*.mp4")}
    orphans = on_disk - known
    if orphans:
        print(f"[reconcile] untracked files in {OUTPUT_DIR}/: {sorted(orphans)}")
    if changed:
        save_state(state)
    return state


# --------------------------------------------------------------------------
# 5. Boundary quality check + clip extraction (ffmpeg, atomic)
# --------------------------------------------------------------------------
# Re-encodes (rather than stream-copies) so the cut lands exactly on the
# chosen boundary instead of snapping to the nearest keyframe, writes to a
# .part file and atomically renames it, and a post-hoc check confirms the
# shipped clip actually starts/ends near silence.


def verify_clean_boundaries(clip_path: Path, noise_db: float = -30, edge_tol: float = 0.08):
    sil = find_silences(clip_path, noise_db=noise_db, min_dur=0.05)
    dur = float(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(clip_path)],
        capture_output=True, text=True).stdout.strip())
    starts_clean = any(s <= edge_tol for s, e in sil)
    ends_clean = any(e >= dur - edge_tol for s, e in sil)
    return starts_clean, ends_clean


def extract_clip(video_path: Path, start: float, end: float, out_path: Path,
                  pad_start: float = 0.25, pad_end: float = 0.35):
    s = max(0.0, start - pad_start)
    d = (end - start) + pad_start + pad_end
    tmp_out = out_path.with_suffix(out_path.suffix + ".part")
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{s:.3f}", "-i", str(video_path), "-t", f"{d:.3f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        "-f", "mp4",
        str(tmp_out),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not tmp_out.exists():
        if tmp_out.exists():
            tmp_out.unlink()
        raise RuntimeError(f"ffmpeg failed for {out_path.name}: {result.stderr[-800:]}")
    os.replace(tmp_out, out_path)


# --------------------------------------------------------------------------
# 6. Orchestrator
# --------------------------------------------------------------------------
# Calls the tools in sequence, each depending on the previous one's output:
# silence detection -> transcription -> topic chaptering (depends on the
# transcript) -> boundary snap -> (state check) -> ffmpeg extraction ->
# state write.


def safe_filename(topic: str) -> str:
    """Convert a topic into a safe, readable filename."""
    name = topic.lower().strip()
    name = re.sub(r"[^\w\s-]", "", name)
    name = re.sub(r"\s+", "_", name)
    return name[:100]


def run_agent(video_path: str, topics: List[str]) -> List[Dict]:
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    video_hash = sha256_file(video_path)
    state = reconcile_state()

    silences = find_silences(video_path)
    print(f"[silence] {len(silences)} real silence intervals detected in source audio")

    transcript = transcribe(video_path)
    segments = transcript["segments"]

    spans = chapter_topics(segments, topics)

    results = []
    for topic in topics:
        if topic not in spans:
            print(f"[skip] no chapter found for topic: {topic!r}")
            continue
        raw_start, raw_end = spans[topic]

        # clip_id identity is tied to the RAW (pre-snap) chapter boundaries --
        # stable across runs even if snap_chapter_boundary's window is retuned.
        clip_id = make_clip_id(video_hash, topic, raw_start, raw_end)
        video_folder = OUTPUT_DIR / video_path.stem
        video_folder.mkdir(parents=True, exist_ok=True)
        filename = f"{safe_filename(topic)}_{clip_id[:8]}.mp4"
        out_path = video_folder / filename

        existing = state.get(clip_id)
        if existing and Path(existing["output_path"]).exists():
            print(f"[skip-cached] '{topic}' already generated -> {out_path.name} (no re-encode)")
            results.append(existing)
            continue

        snapped_start = max(0.0, snap_chapter_boundary(raw_start, silences))
        snapped_end = snap_chapter_boundary(raw_end, silences)

        print(f"[generate] '{topic}' -> {raw_start:.2f}s-{raw_end:.2f}s "
              f"(snapped: {snapped_start:.2f}s-{snapped_end:.2f}s) -> {out_path.name}")
        extract_clip(video_path, snapped_start, snapped_end, out_path)

        starts_clean, ends_clean = verify_clean_boundaries(out_path)
        if not (starts_clean and ends_clean):
            print(f"[warn] '{topic}': boundary check found no near-silence at "
                  f"{'start' if not starts_clean else 'end'} of {out_path.name} "
                  f"-- source audio may genuinely have no pause near this content")

        entry = {
            "clip_id": clip_id,
            "topic": topic,
            "video": str(video_path),
            "video_hash": video_hash,
            "start": raw_start,
            "end": raw_end,
            "snapped_start": snapped_start,
            "snapped_end": snapped_end,
            "output_path": str(out_path),
            "output_sha256": sha256_file(out_path),
            "boundary_check_passed": bool(starts_clean and ends_clean),
            "created_at": time.time(),
        }
        state[clip_id] = entry
        save_state(state)  # persist after every clip, not just at the end
        results.append(entry)

    return results


# --------------------------------------------------------------------------
# 7. CLI / demo entry point
# --------------------------------------------------------------------------
# Default topics/video are for a real ~214s recording of a
# database-systems coursework walkthrough with three natural sections.
# Running with no arguments demonstrates the idempotency requirement: run 1
# generates the clips, run 2 (same input) produces zero new files.

DEFAULT_VIDEO = "videos/demo_clip.mp4"
DEFAULT_TOPICS = [
    "welcome and demonstration of the coursework",
    "deliverable 1 database normalisation and stored procedure",
    "deliverable 2 business intelligence solution using SSAS",
]


def print_results(run_label: str, results: List[Dict]) -> None:
    print(f"\n{run_label} produced/confirmed {len(results)} clips:")
    for r in results:
        dur = r["snapped_end"] - r["snapped_start"]
        print(f"  - {r['topic']!r}: {Path(r['output_path']).name}  "
              f"({r['snapped_start']:.1f}s-{r['snapped_end']:.1f}s, {dur:.1f}s long)")


def main():
    parser = argparse.ArgumentParser(
        description="Topic-based video clip agent (rule-based chaptering, idempotent).")
    parser.add_argument("video", nargs="?", default=DEFAULT_VIDEO,
                         help=f"Path to the input video (default: {DEFAULT_VIDEO})")
    parser.add_argument("--topics", nargs="+", default=None,
                         help="List of topic strings to clip. Defaults to the demo topics "
                              "for the sample coursework video.")
    parser.add_argument("--once", action="store_true",
                         help="Run the agent once instead of twice. By default the script "
                              "runs twice on the same input to demonstrate idempotency "
                              "(second run produces zero new files).")
    args = parser.parse_args()

    topics = args.topics if args.topics else DEFAULT_TOPICS

    print("=" * 20, "RUN 1", "=" * 20)
    run1 = run_agent(args.video, topics)
    print_results("Run 1", run1)

    if args.once:
        return

    files_before = sorted(p.name for p in OUTPUT_DIR.glob("**/*.mp4"))

    print("\n" + "=" * 20, "RUN 2 (same input)", "=" * 20)
    run2 = run_agent(args.video, topics)
    print_results("Run 2", run2)

    files_after = sorted(p.name for p in OUTPUT_DIR.glob("**/*.mp4"))

    assert files_before == files_after, "Idempotency violated: file set changed between runs!"
    assert [r["clip_id"] for r in run1] == [r["clip_id"] for r in run2], "Clip IDs differ between runs!"
    print(f"\nOK: run 2 touched 0 new files. Clip count stable at {len(files_after)}.")
    print("Files:", files_after)

    print("\n" + "=" * 20, "BOUNDARY QUALITY", "=" * 20)
    for r in run2:
        status = "OK" if r.get("boundary_check_passed") else "REVIEW"
        print(f"[{status}] '{r['topic']}' -> {Path(r['output_path']).name}  "
              f"({r['snapped_start']:.2f}s-{r['snapped_end']:.2f}s)")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"\n[error] input video not found: {e}", file=sys.stderr)
        print("Place your video under videos/ and pass its path as the first argument, "
              "e.g.: python clip_agent.py videos/your_video.mp4", file=sys.stderr)
        sys.exit(1)
