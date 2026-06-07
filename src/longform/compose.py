"""
Compositor: assembles a finished iceberg script into a 1080p MP4 with ffmpeg.

Per section it places the sourced shots (real clips, or Ken-Burns AI images, or
slates) timed to that section's voiceover, burns captions from the TTS word
timings (rendered with ImageMagick, never Pillow), cross-dissolves between shots,
ducks a royalty-free music bed under the VO, and concatenates everything. It then
embeds YouTube chapter markers from the tier structure and appends a credits
slate for any attribution-required assets.

Resilience is the rule: every ffmpeg stage is wrapped, and on failure the
pipeline keeps the last good output. A single bad shot/clip/caption never crashes
the render. Output MP4 + metadata go to review_queue/pending/<run_id>/; nothing
publishes.
"""

import json
import os
import subprocess
from typing import Optional

W, H, FPS = 1920, 1080, 30
XFADE = 0.5  # cross-dissolve seconds


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested)
# --------------------------------------------------------------------------- #
def build_caption_chunks(word_timings, max_words: int = 7, max_secs: float = 3.5) -> list:
    """Groups per-word timings into readable caption chunks {text, start, end}."""
    chunks, current = [], []
    for word in word_timings or []:
        if not current:
            current = [word]
            continue
        span = word["end"] - current[0]["start"]
        if len(current) >= max_words or span > max_secs:
            chunks.append(_chunk(current))
            current = [word]
        else:
            current.append(word)
    if current:
        chunks.append(_chunk(current))
    return chunks


def _chunk(words) -> dict:
    return {
        "text": " ".join(w["word"] for w in words).strip(),
        "start": round(words[0]["start"], 3),
        "end": round(words[-1]["end"], 3),
    }


def build_chapters(sections) -> list:
    """
    Cumulative chapter markers from [{title, duration}] -> [{title, start, end}].
    """
    chapters, cursor = [], 0.0
    for section in sections:
        duration = max(0.0, float(section.get("duration", 0.0)))
        chapters.append(
            {"title": section.get("title", "Chapter"), "start": round(cursor, 3),
             "end": round(cursor + duration, 3)}
        )
        cursor += duration
    return chapters


def _hhmmss(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def format_chapters_for_description(chapters) -> str:
    """YouTube description chapter lines (must start at 0:00)."""
    return "\n".join(f"{_hhmmss(c['start'])} {c['title']}" for c in chapters)


def build_credits(assets) -> list:
    """Unique attribution credit lines for attribution-required assets."""
    credits = []
    for asset in assets or []:
        if asset.get("attribution_required") and asset.get("attribution"):
            line = asset["attribution"]
            if line not in credits:
                credits.append(line)
    return credits


def ffmetadata_chapters(chapters) -> str:
    """Serializes chapters to an ffmetadata file body (milliseconds)."""
    lines = [";FFMETADATA1"]
    for chapter in chapters:
        lines += [
            "[CHAPTER]", "TIMEBASE=1/1000",
            f"START={int(chapter['start'] * 1000)}",
            f"END={int(chapter['end'] * 1000)}",
            f"title={chapter['title']}",
        ]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# ffmpeg plumbing (defensive)
# --------------------------------------------------------------------------- #
def _ffprobe_path(ffmpeg_path: str) -> str:
    base = os.path.basename(ffmpeg_path)
    if "ffmpeg" in base:
        return os.path.join(os.path.dirname(ffmpeg_path), base.replace("ffmpeg", "ffprobe"))
    return "ffprobe"


def _run(cmd, timeout=600) -> bool:
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=timeout)
        return result.returncode == 0
    except Exception:
        return False


def probe_duration(path: str, ffmpeg_path: str = "ffmpeg") -> Optional[float]:
    """Media duration in seconds via ffprobe, or None."""
    try:
        out = subprocess.run(
            [_ffprobe_path(ffmpeg_path), "-v", "error", "-show_entries",
             "format=duration", "-of", "default=nw=1:nk=1", path],
            capture_output=True, timeout=60, text=True,
        )
        return float(out.stdout.strip())
    except Exception:
        return None


def render_shot_clip(asset: dict, duration: float, out_path: str, ffmpeg_path: str) -> Optional[str]:
    """Renders one shot to a silent 1080p clip: Ken-Burns for stills, scale/crop
    (looped) for video. Returns the path or None."""
    duration = max(0.8, float(duration))
    kind = asset.get("kind", "image")
    path = asset.get("path")
    if not path or not os.path.exists(path):
        return None

    if kind == "video":
        vf = (f"scale={W}:{H}:force_original_aspect_ratio=increase,"
              f"crop={W}:{H},setsar=1,fps={FPS},format=yuv420p")
        cmd = [ffmpeg_path, "-y", "-stream_loop", "-1", "-i", path, "-t", f"{duration:.3f}",
               "-vf", vf, "-an", "-c:v", "libx264", "-preset", "veryfast",
               "-pix_fmt", "yuv420p", out_path]
    else:
        frames = int(duration * FPS)
        vf = (f"scale={W*2}:{H*2}:force_original_aspect_ratio=increase,crop={W*2}:{H*2},"
              f"zoompan=z='min(zoom+0.0008,1.5)':d={frames}:s={W}x{H}:fps={FPS},"
              f"format=yuv420p")
        cmd = [ffmpeg_path, "-y", "-loop", "1", "-i", path, "-t", f"{duration:.3f}",
               "-vf", vf, "-c:v", "libx264", "-preset", "veryfast",
               "-pix_fmt", "yuv420p", out_path]
    return out_path if _run(cmd) else None


def concat_clips(clips, out_path: str, ffmpeg_path: str) -> Optional[str]:
    """Concatenates clips (re-encode for safety). Hard cuts."""
    clips = [c for c in clips if c and os.path.exists(c)]
    if not clips:
        return None
    if len(clips) == 1:
        return clips[0]
    list_file = out_path + ".txt"
    with open(list_file, "w", encoding="utf-8") as handle:
        for clip in clips:
            handle.write(f"file '{os.path.abspath(clip)}'\n")
    cmd = [ffmpeg_path, "-y", "-f", "concat", "-safe", "0", "-i", list_file,
           "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
           "-r", str(FPS), out_path]
    return out_path if _run(cmd) else None


def xfade_clips(clips, durations, out_path: str, ffmpeg_path: str) -> Optional[str]:
    """Cross-dissolves a list of clips. Falls back to concat on any failure."""
    clips = [c for c in clips if c and os.path.exists(c)]
    if len(clips) < 2:
        return concat_clips(clips, out_path, ffmpeg_path)
    try:
        inputs = []
        for clip in clips:
            inputs += ["-i", clip]
        # chain xfade: offset accumulates (durations - overlap)
        filt = ""
        prev = "0:v"
        offset = max(0.1, durations[0] - XFADE)
        for index in range(1, len(clips)):
            label = f"x{index}"
            filt += (f"[{prev}][{index}:v]xfade=transition=fade:duration={XFADE}:"
                     f"offset={offset:.3f}[{label}];")
            prev = label
            offset += max(0.1, durations[index] - XFADE)
        filt = filt.rstrip(";")
        cmd = [ffmpeg_path, "-y", *inputs, "-filter_complex", filt, "-map", f"[{prev}]",
               "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", out_path]
        if _run(cmd):
            return out_path
    except Exception:
        pass
    return concat_clips(clips, out_path, ffmpeg_path)


def caption_png(text: str, out_path: str, magick_path: str) -> Optional[str]:
    """Renders a transparent caption strip with ImageMagick (not Pillow)."""
    if not text:
        return None
    cmd = [
        magick_path, "-background", "none", "-gravity", "center",
        "-fill", "white", "-stroke", "black", "-strokewidth", "3",
        "-font", "Arial-Bold", "-pointsize", "48", "-size", f"{int(W*0.86)}x",
        f"caption:{text}", out_path,
    ]
    if _run(cmd, timeout=60) and os.path.exists(out_path):
        return out_path
    # retry without explicit font
    cmd = [c for c in cmd if c not in ("-font", "Arial-Bold")]
    return out_path if (_run(cmd, timeout=60) and os.path.exists(out_path)) else None


def burn_captions(video: str, chunks, work_dir: str, out_path: str,
                  ffmpeg_path: str, magick_path: str) -> str:
    """Overlays timed caption PNGs onto a video. Returns video unchanged on
    failure (never crashes the render)."""
    try:
        items = []
        for index, chunk in enumerate(chunks):
            png = caption_png(chunk["text"], os.path.join(work_dir, f"cap{index}.png"), magick_path)
            if png:
                items.append((png, chunk["start"], chunk["end"]))
        if not items:
            return video
        inputs = ["-i", video]
        for png, _s, _e in items:
            inputs += ["-i", png]
        filt, prev = "", "0:v"
        for index, (_png, start, end) in enumerate(items, start=1):
            label = f"c{index}"
            filt += (f"[{prev}][{index}:v]overlay=x=(W-w)/2:y=H-h-90:"
                     f"enable='between(t,{start:.3f},{end:.3f})'[{label}];")
            prev = label
        filt = filt.rstrip(";")
        cmd = [ffmpeg_path, "-y", *inputs, "-filter_complex", filt, "-map", f"[{prev}]",
               "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", out_path]
        if _run(cmd):
            return out_path
    except Exception:
        pass
    return video


def mux_audio(video: str, audio: Optional[str], duration: float, out_path: str,
              ffmpeg_path: str) -> str:
    """Attaches the section VO (or silence) to a silent section video."""
    if audio and os.path.exists(audio):
        cmd = [ffmpeg_path, "-y", "-i", video, "-i", audio, "-map", "0:v", "-map", "1:a",
               "-c:v", "copy", "-c:a", "aac", "-shortest", out_path]
    else:  # silent track
        cmd = [ffmpeg_path, "-y", "-i", video, "-f", "lavfi", "-i",
               f"anullsrc=channel_layout=stereo:sample_rate=44100", "-map", "0:v",
               "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-t", f"{duration:.3f}", out_path]
    return out_path if _run(cmd) else video


def mix_music(video: str, music: Optional[str], out_path: str, ffmpeg_path: str) -> str:
    """Ducks a music bed under the existing VO. Returns video unchanged on
    failure or when there's no music."""
    if not music or not os.path.exists(music):
        return video
    try:
        filt = "[1:a]volume=0.16,aloop=loop=-1:size=2e9[m];[0:a][m]amix=inputs=2:duration=first:dropout_transition=2[a]"
        cmd = [ffmpeg_path, "-y", "-i", video, "-i", music, "-filter_complex", filt,
               "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-shortest", out_path]
        if _run(cmd):
            return out_path
    except Exception:
        pass
    return video


def embed_chapters(video: str, chapters, out_path: str, ffmpeg_path: str) -> str:
    """Embeds chapter markers via an ffmetadata file."""
    try:
        meta = video + ".ffmeta"
        with open(meta, "w", encoding="utf-8") as handle:
            handle.write(ffmetadata_chapters(chapters))
        cmd = [ffmpeg_path, "-y", "-i", video, "-i", meta, "-map_metadata", "1",
               "-codec", "copy", out_path]
        if _run(cmd):
            return out_path
    except Exception:
        pass
    return video


def _first_music(music_dir: str) -> Optional[str]:
    if not music_dir or not os.path.isdir(music_dir):
        return None
    for name in sorted(os.listdir(music_dir)):
        if name.lower().endswith((".mp3", ".wav", ".m4a", ".ogg")):
            return os.path.join(music_dir, name)
    return None


def concat_av(clips, out_path: str, ffmpeg_path: str) -> Optional[str]:
    """Concatenates clips preserving BOTH video and audio (re-encode)."""
    clips = [c for c in clips if c and os.path.exists(c)]
    if not clips:
        return None
    if len(clips) == 1:
        return clips[0]
    list_file = out_path + ".txt"
    with open(list_file, "w", encoding="utf-8") as handle:
        for clip in clips:
            handle.write(f"file '{os.path.abspath(clip)}'\n")
    cmd = [ffmpeg_path, "-y", "-f", "concat", "-safe", "0", "-i", list_file,
           "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-r", str(FPS), out_path]
    return out_path if _run(cmd) else None


def render_credits_clip(credits_lines, out_path: str, ffmpeg_path: str,
                        magick_path: str, seconds: float = 5.0) -> Optional[str]:
    """A short closing credits clip (ImageMagick text) with a silent track."""
    if not credits_lines:
        return None
    text = "Footage & music credits\n\n" + "\n".join(credits_lines)
    png = out_path + ".png"
    cmd = [magick_path, "-size", f"{int(W*0.8)}x{int(H*0.8)}", "-background", "#141821",
           "-fill", "white", "-gravity", "center", "-font", "Arial-Bold",
           "-pointsize", "34", f"caption:{text}", "-extent", f"{W}x{H}", png]
    if not (_run(cmd, timeout=60) and os.path.exists(png)):
        cmd = [c for c in cmd if c not in ("-font", "Arial-Bold")]
        if not (_run(cmd, timeout=60) and os.path.exists(png)):
            return None
    clip = render_shot_clip({"kind": "image", "path": png}, seconds, out_path + ".silent.mp4", ffmpeg_path)
    if not clip:
        return None
    return mux_audio(clip, None, seconds, out_path, ffmpeg_path)


def _estimate_duration(text: str, wpm: int = 155) -> float:
    words = len((text or "").split())
    return max(2.0, words / max(1, wpm) * 60.0)


def produce(
    script: dict,
    run_id: str,
    out_dir: str,
    source_shot_fn,
    tts_fn,
    ffmpeg_path: str = "ffmpeg",
    magick_path: str = "magick",
    music_dir: Optional[str] = None,
    db_path: Optional[str] = None,
    work_dir: Optional[str] = None,
) -> dict:
    """
    Renders ``script`` into a 1080p MP4 in ``out_dir``. Never raises.

    Args:
        source_shot_fn(shot_text, shot_index, section_title, dest_dir) -> asset
            dict ({kind, path, source, license, attribution_required,
            attribution, ...}). Sourcing + license logging happen inside it.
        tts_fn(text, out_path) -> {audio_path, word_timings, provider}.

    Returns a report dict: mp4_path, duration, chapters, chapters_text, credits,
    real_clips, ai_fallbacks, slates, failed_shots, sections.
    """
    work_dir = work_dir or os.path.join(out_dir, "_work")
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    # Build the narrated sections from the script.
    sections = [("Introduction", script.get("cold_hook", ""),
                 [f"{script.get('iceberg_topic', 'the topic')} ominous intro"])]
    for tier in script.get("tiers", []):
        shots = tier.get("shot_list") or [tier.get("entry_title", "")]
        sections.append((f"{tier.get('label', '')}: {tier.get('entry_title', '')}".strip(": "),
                         tier.get("narration", ""), shots))
    sections.append(("Conclusion", script.get("final_payoff", ""),
                     [f"{script.get('iceberg_topic', 'the topic')} dark finale"]))

    section_videos, chapter_specs = [], []
    counts = {"real_clips": 0, "ai_fallbacks": 0, "slates": 0, "failed_shots": 0}
    all_assets = []
    shot_counter = 0

    for sec_index, (title, narration, shots) in enumerate(sections):
        sec_dir = os.path.join(work_dir, f"sec{sec_index}")
        os.makedirs(sec_dir, exist_ok=True)

        # VO
        audio_path, word_timings = None, []
        try:
            tts = tts_fn(narration, os.path.join(sec_dir, "vo.mp3")) or {}
            audio_path = tts.get("audio_path")
            word_timings = tts.get("word_timings") or []
        except Exception:
            pass
        duration = probe_duration(audio_path, ffmpeg_path) if audio_path else None
        if not duration:
            duration = _estimate_duration(narration)

        # Shots -> clips
        shots = shots or [title]
        per_shot = max(1.2, duration / len(shots))
        clips, clip_durs = [], []
        for shot_text in shots:
            try:
                asset = source_shot_fn(shot_text, shot_counter, title, sec_dir)
            except Exception:
                asset = None
            shot_counter += 1
            if not asset or not asset.get("path") or not os.path.exists(asset["path"]):
                counts["failed_shots"] += 1
                continue
            all_assets.append(asset)
            src = asset.get("source", "")
            if src == "ai_generated":
                counts["ai_fallbacks"] += 1
            elif src == "slate":
                counts["slates"] += 1
            else:  # any real sourced asset (pexels/pixabay video, openverse image)
                counts["real_clips"] += 1
            clip = render_shot_clip(asset, per_shot, os.path.join(sec_dir, f"clip{len(clips)}.mp4"), ffmpeg_path)
            if clip:
                clips.append(clip)
                clip_durs.append(per_shot)
            else:
                counts["failed_shots"] += 1

        if not clips:  # guarantee at least a slate so the section exists
            from . import sourcer

            slate = sourcer.make_slate(os.path.join(sec_dir, "slate.png"))
            clip = render_shot_clip({"kind": "image", "path": slate}, duration,
                                    os.path.join(sec_dir, "clip0.mp4"), ffmpeg_path)
            if clip:
                clips, clip_durs = [clip], [duration]

        seg_video = xfade_clips(clips, clip_durs, os.path.join(sec_dir, "seg.mp4"), ffmpeg_path)
        if not seg_video:
            continue
        seg_video = burn_captions(seg_video, build_caption_chunks(word_timings), sec_dir,
                                  os.path.join(sec_dir, "seg_cap.mp4"), ffmpeg_path, magick_path)
        seg_av = mux_audio(seg_video, audio_path, duration, os.path.join(sec_dir, "seg_av.mp4"), ffmpeg_path)
        section_videos.append(seg_av)
        actual = probe_duration(seg_av, ffmpeg_path) or duration
        chapter_specs.append({"title": title, "duration": actual})

    # Assemble
    chapters = build_chapters(chapter_specs)
    credits = build_credits(all_assets)
    base = concat_av(section_videos, os.path.join(work_dir, "joined.mp4"), ffmpeg_path)

    if base and credits:
        credit_clip = render_credits_clip(credits, os.path.join(work_dir, "credits.mp4"),
                                          ffmpeg_path, magick_path)
        if credit_clip:
            joined = concat_av([base, credit_clip], os.path.join(work_dir, "joined2.mp4"), ffmpeg_path)
            base = joined or base

    if base:
        music = _first_music(music_dir)
        base = mix_music(base, music, os.path.join(work_dir, "music.mp4"), ffmpeg_path)
        final = os.path.join(out_dir, f"{run_id}_final.mp4")
        base = embed_chapters(base, chapters, final, ffmpeg_path) or base
        if base != final and os.path.exists(base):  # embed failed; copy last good
            import shutil
            shutil.copyfile(base, final)
            base = final

    duration = probe_duration(base, ffmpeg_path) if base else None
    chapters_text = format_chapters_for_description(chapters)

    return {
        "mp4_path": base, "duration": duration, "chapters": chapters,
        "chapters_text": chapters_text, "credits": credits,
        "real_clips": counts["real_clips"], "ai_fallbacks": counts["ai_fallbacks"],
        "slates": counts["slates"], "failed_shots": counts["failed_shots"],
        "sections": len(section_videos), "assets": all_assets,
    }
