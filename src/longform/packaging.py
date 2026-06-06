"""
Packaging for iceberg deep-dives: titles, thumbnails, and the review-queue item.

- Titles: 5 variants built from the proven iceberg formula plus 'specific_number'
  and 'curiosity_gap' variants (see title_formulas.json), styled after the real
  winners in farm.db (intensifiers, closing brackets, depth challenges).
- Thumbnails: 3 tiered-iceberg concepts, each rendered at 1280x720 by calling
  generate_image() from image_providers.py (the free Pollinations path) and then
  overlaying big, readable PIL text.
- Everything (script, all hook variants + scores, all title variants, thumbnail
  concepts) is written into a single review-queue item under
  review_queue/pending/<run_id>/; the chosen title/hook are logged in the ledger.

Nothing here publishes -- it only stages a human review item.
"""

import json
import os
from datetime import datetime, timezone
from typing import Callable, Optional

from . import image_providers, storage

_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_REVIEW_BASE = os.path.join(_ROOT_DIR, "review_queue", "pending")
THUMB_SIZE = (1280, 720)


# --------------------------------------------------------------------------- #
# Titles
# --------------------------------------------------------------------------- #
def _display_topic(topic: str) -> str:
    """Title-cases a topic unless it already carries deliberate capitals."""
    topic = (topic or "").strip()
    if topic and topic == topic.lower():
        return topic.title()
    return topic


def _odd_near(value: Optional[int]) -> int:
    """Nearest 'specific, odd' count from the formula rules (7/9/13 feel)."""
    candidates = [7, 9, 13]
    if not value or value < 5:
        return 9
    # round value up to an odd number, capped at 13.
    odd = value if value % 2 == 1 else value + 1
    return min(odd, 13) if odd >= 7 else min(candidates, key=lambda c: abs(c - odd))


def build_title_variants(topic: str, entry_count: Optional[int] = None) -> list:
    """
    Builds 5 title variants for the iceberg on ``topic``.

    Returns a list of {"formula", "title"} dicts: three using the 'iceberg'
    formula (base, intensified, depth-challenge), one 'specific_number', and one
    'curiosity_gap'.
    """
    name = _display_topic(topic)
    number = _odd_near(entry_count)
    return [
        {"formula": "iceberg", "title": f"The {name} Iceberg Explained"},
        {
            "formula": "iceberg",
            "title": f"The DISTURBING {name} Iceberg (It Gets Worse)",
        },
        {
            "formula": "specific_number",
            "title": f"{number} Layers of the {name} Iceberg That Go Too Deep",
        },
        {
            "formula": "curiosity_gap",
            "title": f"The {name} Secret Hiding at the Bottom of the Iceberg",
        },
        {
            "formula": "iceberg",
            "title": f"The {name} Iceberg Nobody Has Reached the Bottom Of",
        },
    ]


# --------------------------------------------------------------------------- #
# Thumbnail concepts
# --------------------------------------------------------------------------- #
def build_thumbnail_concepts(topic: str) -> list:
    """Builds 3 distinct tiered-iceberg thumbnail concepts for ``topic``."""
    name = _display_topic(topic)
    upper = name.upper()
    return [
        {
            "concept_id": "split_waterline",
            "visual_prompt": (
                f"dramatic photoreal iceberg split at the waterline, small bright "
                f"tip above calm water and a vast dark mass plunging into black "
                f"depths below, themed around {name}, cinematic, high contrast, "
                f"ominous, 16:9 YouTube thumbnail"
            ),
            "overlay_text": f"THE {upper}\nICEBERG",
            "palette": "icy cyan over near-black deep",
        },
        {
            "concept_id": "tiered_glow",
            "visual_prompt": (
                f"a towering iceberg cross-section divided into glowing horizontal "
                f"tiers descending into darkness, each tier dimmer and more "
                f"unsettling, {name} motifs, volumetric light, eerie, cinematic, "
                f"16:9 YouTube thumbnail"
            ),
            "overlay_text": f"{upper}\nHOW DEEP?",
            "palette": "teal glow fading to abyssal black",
        },
        {
            "concept_id": "abyss_focus",
            "visual_prompt": (
                f"extreme low-angle looking down the bottom tier of an iceberg into "
                f"a pitch-black abyss, a single faint unsettling detail about {name} "
                f"barely visible at the very bottom, dread, fog, cinematic, 16:9 "
                f"YouTube thumbnail"
            ),
            "overlay_text": f"THE BOTTOM OF\nTHE {upper} ICEBERG",
            "palette": "deep blue-black, single cold highlight",
        },
    ]


# --------------------------------------------------------------------------- #
# Rendering (generate_image + PIL text overlay)
# --------------------------------------------------------------------------- #
def _font_path() -> Optional[str]:
    candidate = os.path.join(_ROOT_DIR, "fonts", "bold_font.ttf")
    return candidate if os.path.exists(candidate) else None


def _fit_font(draw, text_lines, font_path, max_width, max_height):
    """Largest font size at which all lines fit the box; falls back to default."""
    from PIL import ImageFont

    if not font_path:
        return ImageFont.load_default()
    for size in range(160, 28, -6):
        font = ImageFont.truetype(font_path, size)
        widths, heights = [], []
        for line in text_lines:
            bbox = draw.textbbox((0, 0), line, font=font, stroke_width=max(2, size // 18))
            widths.append(bbox[2] - bbox[0])
            heights.append(bbox[3] - bbox[1])
        if max(widths) <= max_width and (sum(heights) + 14 * len(text_lines)) <= max_height:
            return font
    return ImageFont.truetype(font_path, 34)


def overlay_text(base_image_path: str, text: str, out_path: str) -> str:
    """
    Draws large, high-contrast text onto a 1280x720 thumbnail.

    The base image is cover-fit to 1280x720; text is bottom-centered with a thick
    dark stroke for readability over any background.
    """
    from PIL import Image, ImageDraw

    image = Image.open(base_image_path).convert("RGB")
    # Cover-fit to exactly 1280x720.
    target_w, target_h = THUMB_SIZE
    scale = max(target_w / image.width, target_h / image.height)
    resized = image.resize((round(image.width * scale), round(image.height * scale)))
    left = (resized.width - target_w) // 2
    top = (resized.height - target_h) // 2
    canvas = resized.crop((left, top, left + target_w, top + target_h))

    draw = ImageDraw.Draw(canvas)
    lines = [line for line in (text or "").split("\n") if line.strip()] or [""]
    font = _fit_font(draw, lines, _font_path(), int(target_w * 0.92), int(target_h * 0.5))

    stroke = max(3, getattr(font, "size", 40) // 14)
    line_heights = [
        draw.textbbox((0, 0), line, font=font, stroke_width=stroke)[3]
        - draw.textbbox((0, 0), line, font=font, stroke_width=stroke)[1]
        for line in lines
    ]
    gap = 14
    block_height = sum(line_heights) + gap * (len(lines) - 1)
    y = target_h - block_height - 48  # sit in the lower third
    for line, line_h in zip(lines, line_heights):
        bbox = draw.textbbox((0, 0), line, font=font, stroke_width=stroke)
        x = (target_w - (bbox[2] - bbox[0])) // 2
        draw.text(
            (x, y),
            line,
            font=font,
            fill=(255, 255, 255),
            stroke_width=stroke,
            stroke_fill=(0, 0, 0),
        )
        y += line_h + gap

    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    canvas.save(out_path, "PNG")
    return out_path


def render_thumbnail(
    concept: dict,
    output_path: str,
    generate_image_fn: Optional[Callable] = None,
    do_overlay: bool = True,
    quality: str = "standard",
    config: Optional[dict] = None,
    session=None,
    db_path: Optional[str] = None,
) -> dict:
    """
    Renders one thumbnail: generate a 1280x720 base via generate_image(), then
    overlay the concept's text.

    Args:
        generate_image_fn: defaults to image_providers.generate_image. Injected
            (mocked) in tests so no real network call happens.

    Returns:
        dict: concept_id, path, provider, is_placeholder, overlay_text,
        visual_prompt.
    """
    gen = generate_image_fn or image_providers.generate_image
    base_path = output_path
    if do_overlay:
        root, ext = os.path.splitext(output_path)
        base_path = f"{root}_base{ext or '.png'}"

    result = gen(
        prompt=concept["visual_prompt"],
        aspect_ratio="16:9",
        output_path=base_path,
        quality=quality,
        config=config,
        session=session,
        db_path=db_path,
    )

    final_path = output_path
    if do_overlay:
        overlay_text(result.path, concept.get("overlay_text", ""), output_path)
    else:
        final_path = result.path

    return {
        "concept_id": concept.get("concept_id"),
        "path": final_path,
        "provider": getattr(result, "provider", None),
        "is_placeholder": getattr(result, "is_placeholder", None),
        "overlay_text": concept.get("overlay_text", ""),
        "visual_prompt": concept["visual_prompt"],
    }


# --------------------------------------------------------------------------- #
# Review-queue item
# --------------------------------------------------------------------------- #
def submit_to_review(
    run_id: str,
    topic: str,
    script: dict,
    hooks_result: dict,
    titles: list,
    concepts: list,
    base_dir: Optional[str] = None,
    render: bool = True,
    generate_image_fn: Optional[Callable] = None,
    do_overlay: bool = True,
    config: Optional[dict] = None,
    session=None,
    db_path: Optional[str] = None,
) -> dict:
    """
    Writes a single review item with the script, all hook + title variants, and
    rendered thumbnails, then logs the chosen title/hook in the ledger.

    Returns a summary dict including the item directory and chosen fields.
    """
    item_dir = os.path.join(base_dir or DEFAULT_REVIEW_BASE, run_id)
    os.makedirs(item_dir, exist_ok=True)

    thumbnails = []
    if render:
        for index, concept in enumerate(concepts, start=1):
            out = os.path.join(item_dir, f"thumb_{index}_{concept['concept_id']}.png")
            thumbnails.append(
                render_thumbnail(
                    concept,
                    out,
                    generate_image_fn=generate_image_fn,
                    do_overlay=do_overlay,
                    config=config,
                    session=session,
                    db_path=db_path,
                )
            )
    else:
        thumbnails = [dict(concept, path=None) for concept in concepts]

    best_hook = (hooks_result or {}).get("best", {})
    chosen_title = titles[0]["title"] if titles else None
    chosen_hook = best_hook.get("hook")

    package = {
        "run_id": run_id,
        "niche": "iceberg_deepdive",
        "topic": topic,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "pending",
        "prompt_version": script.get("prompt_version"),
        "chosen": {"title": chosen_title, "hook": chosen_hook},
        "script": script,
        "hook_variants": (hooks_result or {}).get("variants", []),
        "title_variants": titles,
        "thumbnails": thumbnails,
    }

    with open(os.path.join(item_dir, "package.json"), "w", encoding="utf-8") as handle:
        json.dump(package, handle, indent=2)

    # Ledger: finalize the run row + record every title variant.
    storage.update_creative_run_choices(
        run_id=run_id,
        chosen_title=chosen_title,
        chosen_hook=chosen_hook,
        review_item_path=item_dir,
        db_path=db_path,
    )
    created_at = datetime.now(timezone.utc).isoformat()
    for index, variant in enumerate(titles):
        storage.log_title_variant(
            {
                "run_id": run_id,
                "created_at": created_at,
                "formula": variant.get("formula"),
                "title": variant.get("title"),
                "is_chosen": int(index == 0),
            },
            db_path=db_path,
        )

    return {
        "run_id": run_id,
        "item_dir": item_dir,
        "chosen_title": chosen_title,
        "chosen_hook": chosen_hook,
        "thumbnails": thumbnails,
        "package_path": os.path.join(item_dir, "package.json"),
    }


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def generate_iceberg_package(
    topic: str,
    run_id: str,
    llm: Optional[Callable] = None,
    target_minutes: int = 10,
    db_path: Optional[str] = None,
    base_dir: Optional[str] = None,
    generate_image_fn: Optional[Callable] = None,
    n_hooks: int = 8,
) -> dict:
    """
    End-to-end: script -> hooks -> titles -> thumbnails -> review item.

    Returns the submit_to_review summary plus the script and hooks results.
    """
    from . import hooks as hooks_module
    from . import script as script_module

    script = script_module.generate_script(
        topic, llm=llm, target_minutes=target_minutes, db_path=db_path, run_id=run_id
    )
    hooks_result = hooks_module.run_hook_engine(
        topic, llm=llm, n=n_hooks, db_path=db_path, run_id=run_id
    )
    titles = build_title_variants(topic, entry_count=script["entry_count"])
    concepts = build_thumbnail_concepts(topic)

    summary = submit_to_review(
        run_id=run_id,
        topic=topic,
        script=script,
        hooks_result=hooks_result,
        titles=titles,
        concepts=concepts,
        base_dir=base_dir,
        generate_image_fn=generate_image_fn,
        db_path=db_path,
    )
    summary["script"] = script
    summary["hooks_result"] = hooks_result
    return summary
