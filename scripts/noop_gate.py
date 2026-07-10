#!/usr/bin/env python3
"""A gate that always passes — the canonical green fixture gate.

Task gates must be `make <target>` or `python[3] <repo-relative .py>` (no
inline -c/-m code), so tests point at this script instead of `python -c`.
"""
import sys

if __name__ == "__main__":
    sys.exit(0)
