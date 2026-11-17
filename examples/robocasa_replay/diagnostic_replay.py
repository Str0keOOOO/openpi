"""Backward-compatible entrypoint for the canonical Human replay pipeline.

All full-episode and segmented replay runs use replay_human.py and its single
official reset/action-execution path.
"""
from replay_human import main


if __name__ == "__main__":
    main()
