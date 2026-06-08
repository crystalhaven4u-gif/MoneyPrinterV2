"""
Iceberg-video thumbnail generator (the classic, instantly-recognizable look).

The genre's thumbnail is an actual iceberg with a waterline across the frame,
a small bright tip above and a vast mass plunging into black water, ominous
lighting, and a bold high-contrast title. This builds that:

  1. A dramatic 1280x720 iceberg BASE -- AI-generated when an image backend is
     reachable (image_providers.generate_image, free Pollinations path), with a
     fully PROCEDURAL ImageMagick iceberg as the fallback so the thumbnail is a
     real iceberg even when AI is unavailable (never a gray placeholder).
  2. A composite pass (ImageMagick, not Pillow): bold title text plunging over
     the ice, optional faint tier labels down the side.

Three variants are produced for the human to pick. Nothing here publishes.
"""

import os
import subprocess
from typing import Callable, Optional

from . import image_providers

THUMB_W, THUMB_H = 1280, 720

# Three distinct treatments. Each tweaks the palette, the title placement, and
# whether faint tier labels run down the side -- so the reviewer has real
# choices, not three near-identical frames.
VARIANTS = (
    {
        "variant_id": "classic_centered",
        "palette": ("#2b4456", "#01030a", "#cfeeff", "#5fa9d6"),
        "title_gravity": "north", "title_offset": 40, "tier_labels": False,
        "ai_extra": "centered iceberg, symmetrical, moody fog",
    },
    {
        "variant_id": "abyss_tiers",
        "palette": ("#243a4a", "#010206", "#bfe6ff", "#4f93bd"),
        "title_gravity": "south", "title_offset": 36, "tier_labels": True,
        "ai_extra": "deep abyss below, faint layers descending, volumetric god rays",
    },
    {
        "variant_id": "cold_dread",
        "palette": ("#1d3340", "#000308", "#e6f6ff", "#3f7fa6"),
        "title_gravity": "north", "title_offset": 44, "tier_labels": True,
        "ai_extra": "icy cyan rim light, near-black water, dread",
    },
)


def _run(cmd, timeout=120) -> bool:
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=timeout)
        return result.returncode == 0
    except Exception:
        return False


def ai_prompt_for(topic: str, extra: str) -> str:
    """The AI base prompt for a dramatic, on-genre iceberg image."""
    return (
        f"a single dramatic photoreal iceberg, waterline across the middle of the "
        f"frame, small bright tip above dark foreboding ocean and a vast ice mass "
        f"plunging into pitch-black depths below, {extra}, themed around {topic}, "
        f"cinematic, high contrast, ominous lighting, no text, 16:9"
    )


# --------------------------------------------------------------------------- #
# Procedural iceberg base (ImageMagick) -- the always-available fallback
# --------------------------------------------------------------------------- #
def procedural_base(out_path: str, magick: str, palette) -> Optional[str]:
    """Draws a classic iceberg (gradient water + tip above / mass below the
    waterline, fading into the abyss) with ImageMagick. Returns the path or None.
    """
    sky, abyss, ice_light, ice_deep = palette
    waterline = 250  # upper third: small tip, huge underwater mass

    # Above-water jagged tip and the large underwater mass (tapering to depth).
    tip = ("polygon 545,%d 600,150 645,185 690,120 735,170 760,%d"
           % (waterline, waterline))
    mass = ("polygon 545,%d 760,%d 880,360 905,470 820,600 690,690 560,600 "
            "415,470 430,360" % (waterline, waterline))

    cmd = [
        magick,
        # 1) water/sky gradient (light at top -> black abyss)
        "-size", f"{THUMB_W}x{THUMB_H}", f"gradient:{sky}-{abyss}",
        # 2) iceberg layer drawn on transparency, then faded into the abyss
        "(", "-size", f"{THUMB_W}x{THUMB_H}", "xc:none",
        "-fill", ice_light, "-draw", tip,
        "-fill", ice_deep, "-draw", mass,
        # re-light the tip on top of the mass so the waterline reads
        "-fill", ice_light, "-draw", tip,
        "(", "-size", f"{THUMB_W}x{THUMB_H}", f"gradient:white-{abyss}", ")",
        "-compose", "CopyOpacity", "-composite",
        ")",
        "-compose", "over", "-composite",
        # 3) faint waterline highlight + a cold vignette
        "-fill", "none", "-stroke", ice_light, "-strokewidth", "2",
        "-draw", f"line 0,{waterline} {THUMB_W},{waterline}",
        "-stroke", "none",
        out_path,
    ]
    if _run(cmd) and os.path.exists(out_path):
        return out_path
    return None


# --------------------------------------------------------------------------- #
# Base = AI when reachable, else procedural
# --------------------------------------------------------------------------- #
def generate_base(
    topic: str,
    variant: dict,
    out_path: str,
    magick: str,
    generate_image_fn: Optional[Callable] = None,
    quality: str = "standard",
    config: Optional[dict] = None,
    session=None,
    db_path: Optional[str] = None,
) -> dict:
    """Produces the iceberg base image. Tries AI first; on a flagged placeholder
    (or any failure) draws the procedural iceberg. Returns {path, source}."""
    gen = generate_image_fn or image_providers.generate_image
    try:
        result = gen(
            prompt=ai_prompt_for(topic, variant["ai_extra"]),
            aspect_ratio="16:9", output_path=out_path, quality=quality,
            config=config, session=session, db_path=db_path,
        )
        path = getattr(result, "path", None)
        is_placeholder = getattr(result, "is_placeholder", True)
        if path and os.path.exists(path) and not is_placeholder:
            return {"path": path, "source": "ai_generated"}
    except Exception:
        pass
    drawn = procedural_base(out_path, magick, variant["palette"])
    if drawn:
        return {"path": drawn, "source": "procedural"}
    return {"path": None, "source": "failed"}


# --------------------------------------------------------------------------- #
# Title + tier-label compositing (ImageMagick)
# --------------------------------------------------------------------------- #
def _title_lines(topic: str) -> str:
    name = topic.strip()
    if name.lower().startswith("the "):
        name = name[4:]
    return f"THE {name.upper()}\nICEBERG"


def compose_thumbnail(
    base_path: str, topic: str, variant: dict, out_path: str, magick: str,
    tier_labels: Optional[list] = None,
) -> Optional[str]:
    """Overlays the bold title (and optional faint tier labels) onto the base."""
    title = _title_lines(topic)
    cmd = [
        magick, base_path, "-resize", f"{THUMB_W}x{THUMB_H}^",
        "-gravity", "center", "-extent", f"{THUMB_W}x{THUMB_H}",
    ]
    # Faint tier labels descending the left edge (only when the variant wants
    # them and we were given labels).
    if variant.get("tier_labels") and tier_labels:
        labels = [str(label) for label in tier_labels if str(label).strip()][:6]
        step = int((THUMB_H - 140) / max(1, len(labels)))
        cmd += ["-gravity", "northwest", "-font", "Arial", "-pointsize", "26",
                "-fill", "#add6f0aa", "-stroke", "none"]
        for index, label in enumerate(labels):
            y = 110 + index * step
            cmd += ["-annotate", f"+34+{y}", label]

    # Bold high-contrast title: black outline pass then white fill.
    cmd += ["-gravity", variant.get("title_gravity", "north"),
            "-font", "Arial-Bold", "-pointsize", "104"]
    offset = variant.get("title_offset", 40)
    cmd += ["-strokewidth", "12", "-stroke", "black", "-fill", "none",
            "-annotate", f"+0+{offset}", title,
            "-stroke", "none", "-fill", "white", "-annotate", f"+0+{offset}", title]
    cmd.append(out_path)

    if _run(cmd) and os.path.exists(out_path):
        return out_path
    # Retry without an explicit font (some installs lack Arial-Bold).
    no_font = [a for a in cmd if a not in ("-font", "Arial-Bold", "Arial")]
    return out_path if (_run(no_font) and os.path.exists(out_path)) else None


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def render_iceberg_thumbnails(
    topic: str,
    out_dir: str,
    magick_path: str,
    tier_labels: Optional[list] = None,
    generate_image_fn: Optional[Callable] = None,
    quality: str = "standard",
    config: Optional[dict] = None,
    session=None,
    db_path: Optional[str] = None,
) -> list:
    """Renders the 3 iceberg-thumbnail variants into ``out_dir``.

    Returns a list of {variant_id, path, base_source} (base_source is
    "ai_generated" or "procedural"). Non-blocking: a variant that fails to
    composite still returns its base path so a render never crashes.
    """
    os.makedirs(out_dir, exist_ok=True)
    results = []
    for index, variant in enumerate(VARIANTS, start=1):
        base_path = os.path.join(out_dir, f"ice_thumb_{index}_{variant['variant_id']}_base.png")
        final_path = os.path.join(out_dir, f"ice_thumb_{index}_{variant['variant_id']}.png")
        base = generate_base(
            topic, variant, base_path, magick_path,
            generate_image_fn=generate_image_fn, quality=quality,
            config=config, session=session, db_path=db_path,
        )
        path = None
        if base["path"]:
            path = compose_thumbnail(base["path"], topic, variant, final_path,
                                     magick_path, tier_labels=tier_labels)
        results.append({
            "variant_id": variant["variant_id"],
            "path": path or base["path"],
            "base_source": base["source"],
        })
    return results
