#!/usr/bin/env python3
"""
Review-queue CLI.

Lists items awaiting review and lets a human approve or reject them. Approved
items move to ``review_queue/approved/`` where the publisher can pick them up.
Nothing is ever published from here directly.

Usage:
    python src/review.py                 # interactive review of pending items
    python src/review.py list            # list pending items
    python src/review.py list --state approved
    python src/review.py show <video_id>
    python src/review.py approve <video_id>
    python src/review.py reject <video_id>
"""

import argparse
import os
import sys

# Allow running both as "python src/review.py" and "python -m review".
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import review_queue
from status import error, info, success, warning
from termcolor import colored

try:
    from prettytable import PrettyTable
except Exception:  # pragma: no cover - prettytable is a declared dependency
    PrettyTable = None


def _short(value, length: int = 50) -> str:
    text = str(value or "")
    return text if len(text) <= length else text[: length - 1] + "…"


def _print_items(items: list, state: str) -> None:
    if not items:
        info(f"No items in '{state}'.")
        return

    if PrettyTable is not None:
        table = PrettyTable()
        table.field_names = ["#", "Video ID", "Lane", "Title", "Platforms"]
        table.align = "l"
        for index, item in enumerate(items, start=1):
            metadata = item["metadata"]
            platforms = ", ".join(metadata.get("target_platforms", []) or [])
            table.add_row(
                [
                    index,
                    item["video_id"],
                    metadata.get("lane", ""),
                    _short(metadata.get("title", ""), 40),
                    platforms,
                ]
            )
        print(table)
    else:  # pragma: no cover
        for index, item in enumerate(items, start=1):
            metadata = item["metadata"]
            print(f"{index}. {item['video_id']} - {metadata.get('title', '')}")


def _print_detail(item: dict) -> None:
    metadata = item["metadata"]
    print(colored(f"\nVideo ID: {item['video_id']}", "cyan"))
    print(colored(f"State:    {item['state']}", "cyan"))
    print(colored(f"File:     {item['video_path']}", "cyan"))
    print(f"  Lane:            {metadata.get('lane', '')}")
    print(f"  Title:           {metadata.get('title', '')}")
    print(f"  Description:     {_short(metadata.get('description', ''), 120)}")
    print(f"  Hashtags:        {', '.join(metadata.get('hashtags', []) or [])}")
    print(f"  Target platforms:{', '.join(metadata.get('target_platforms', []) or [])}")
    print(f"  Hook id:         {metadata.get('hook_id', '')}")
    print(f"  Prompt version:  {metadata.get('prompt_version', '')}")
    print(f"  utm_campaign:    {metadata.get('utm_campaign', '')}")


def _find_item(video_id: str, state: str):
    for item in review_queue.list_items(state):
        if item["video_id"] == video_id:
            return item
    return None


def cmd_list(args) -> int:
    items = review_queue.list_items(args.state)
    _print_items(items, args.state)
    return 0


def cmd_show(args) -> int:
    item = _find_item(args.video_id, args.state)
    if item is None:
        error(f"No '{args.state}' item found for id '{args.video_id}'.")
        return 1
    _print_detail(item)
    return 0


def cmd_approve(args) -> int:
    try:
        item = review_queue.approve(args.video_id)
    except FileNotFoundError as exc:
        error(str(exc))
        return 1
    success(f"Approved {item['video_id']} -> {item['video_path']}")
    return 0


def cmd_reject(args) -> int:
    try:
        item = review_queue.reject(args.video_id)
    except FileNotFoundError as exc:
        error(str(exc))
        return 1
    warning(f"Rejected {item['video_id']}.")
    return 0


def cmd_interactive(_args) -> int:
    items = review_queue.list_pending()
    if not items:
        info("No items awaiting review. Nothing to do.")
        return 0

    info(f"{len(items)} item(s) awaiting review.")
    for item in items:
        _print_detail(item)
        while True:
            choice = input(
                colored("  [a]pprove / [r]eject / [s]kip / [q]uit: ", "magenta")
            ).strip().lower()
            if choice in {"a", "approve"}:
                review_queue.approve(item["video_id"])
                success("  Approved.")
                break
            if choice in {"r", "reject"}:
                review_queue.reject(item["video_id"])
                warning("  Rejected.")
                break
            if choice in {"s", "skip", ""}:
                info("  Skipped.")
                break
            if choice in {"q", "quit"}:
                info("Stopping review.")
                return 0
            warning("  Please choose a, r, s, or q.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Review-queue CLI for the content engine.")
    subparsers = parser.add_subparsers(dest="command")

    list_parser = subparsers.add_parser("list", help="List queue items.")
    list_parser.add_argument(
        "--state",
        default=review_queue.PENDING,
        choices=review_queue._STATES,
        help="Queue state to list (default: pending).",
    )
    list_parser.set_defaults(func=cmd_list)

    show_parser = subparsers.add_parser("show", help="Show one item's details.")
    show_parser.add_argument("video_id")
    show_parser.add_argument(
        "--state",
        default=review_queue.PENDING,
        choices=review_queue._STATES,
    )
    show_parser.set_defaults(func=cmd_show)

    approve_parser = subparsers.add_parser("approve", help="Approve a pending item.")
    approve_parser.add_argument("video_id")
    approve_parser.set_defaults(func=cmd_approve)

    reject_parser = subparsers.add_parser("reject", help="Reject a pending item.")
    reject_parser.add_argument("video_id")
    reject_parser.set_defaults(func=cmd_reject)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if getattr(args, "command", None) is None:
        return cmd_interactive(args)

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
