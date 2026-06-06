#!/usr/bin/env python3
"""
Report the long-form LLM setup: active provider, system RAM, installed Ollama
models, and the recommended local model. NEVER pulls weights -- it only prints
the suggested `ollama pull` command for you to run if you choose to.

    python scripts/llm_doctor.py
"""

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))

from longform import llm


def main() -> int:
    info = llm.describe_local_setup()
    rec = info["recommendation"]

    print("=" * 64)
    print("LONG-FORM LLM DOCTOR")
    print("=" * 64)
    print(f"Active provider : {info['active_provider']}")
    print(f"System RAM      : {info['ram_gb']} GB" if info["ram_gb"] else "System RAM      : unknown")
    print(f"Ollama models   : {', '.join(info['installed_ollama_models']) or '(none / Ollama not running)'}")
    print()
    print(f"Recommended local model : {rec['recommended']}")
    print(f"  why            : {rec['reason']}")
    print(f"  installed?     : {'yes' if rec['already_installed'] else 'no'}")
    if rec["pull_needed"]:
        print(f"  to install     : ollama pull {rec['recommended']}")
        print("  (not pulled automatically -- multi-GB download; run it yourself)")
    print()
    print("For best script quality (free): set llm.provider=openai_compatible and")
    print("export GROQ_API_KEY=<your free key from https://console.groq.com>.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
