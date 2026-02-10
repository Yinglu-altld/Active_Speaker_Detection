"""
2-Speaker DOA Test Script

This script tests the DOA (Direction of Arrival) system with two speakers
positioned at different angles. It validates that the system can:
1. Detect two separate speakers
2. Accurately estimate their azimuth angles
3. Switch attention between them based on speech activity

Usage:
    python -m furhat_asd.tools.doa_test --config config.2speaker_test.json

Or run directly:
    python run_2speaker_test.py
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

# Add src to path
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from furhat_asd.config import load_config
from furhat_asd.tools.doa_test import run_doa_test


async def run_2speaker_test() -> None:
    """Run DOA test configured for 2-speaker scenario."""
    # Load config
    cfg = load_config("config.2speaker_test.json")
    
    # Setup logging to see detailed DOA estimates
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    log = logging.getLogger("2speaker_test")
    log.info("=" * 60)
    log.info("2-SPEAKER DOA TEST")
    log.info("=" * 60)
    log.info("Setup Instructions:")
    log.info("1. Position SPEAKER 1 at ~90° (right side)")
    log.info("2. Position SPEAKER 2 at ~270° (left side)")
    log.info("3. Have speaker 1 speak first, then speaker 2")
    log.info("4. Try overlapping speech to test speaker switching")
    log.info("5. Press Ctrl+C to stop the test")
    log.info("=" * 60)
    log.info("")
    
    # Run DOA test
    await run_doa_test(cfg)


def main() -> None:
    """Entry point."""
    try:
        asyncio.run(run_2speaker_test())
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
        return


if __name__ == "__main__":
    main()
