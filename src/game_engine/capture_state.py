"""Optimized screen capture with per-frame OCR of the in-game percentage."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import mss
import numpy as np
import pytesseract

from actions import RandomActionEngine

OUTPUT_DIR = Path(__file__).resolve().parent / "frames"
CAPTURE_SIZE = {"width": 1280, "height": 720}


def _centered_bounds() -> Dict[str, int]:
    """Return capture bounds for a 1280x720 region centered on the main display.

    Returns:
        Dict[str, int]: A dictionary with keys ``top``, ``left``, ``width``,
        and ``height`` describing the capture rectangle. Falls back to the
        origin if screen info is unavailable.
    """
    try:
        from AppKit import NSScreen

        screen = NSScreen.mainScreen()
        if not screen:
            return {"top": 0, "left": 0, **CAPTURE_SIZE}

        frame = screen.frame()
        width = int(frame.size.width)
        height = int(frame.size.height)
        left = max((width - CAPTURE_SIZE["width"]) // 2, 0)
        top = max((height - CAPTURE_SIZE["height"]) // 2, 0)
        return {"top": top, "left": left, **CAPTURE_SIZE}
    except Exception as e:
        print(f"Error getting screen bounds: {e}")
        return {"top": 0, "left": 0, **CAPTURE_SIZE}


def _extract_percentage(frame_array: np.ndarray) -> Optional[float]:
    """Run fast OCR on a cropped percentage region.

    Args:
        frame_array: BGRA image array containing only the percentage area.

    Returns:
        float | None: Parsed percentage (0-100) if OCR succeeds, otherwise
        ``None``.
    """
    try:
        # Convert BGRA to grayscale (much faster than PIL)
        gray = cv2.cvtColor(frame_array, cv2.COLOR_BGRA2GRAY)
        
        # Threshold for better OCR (white text on dark background)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # OCR with optimized config
        text = pytesseract.image_to_string(
            binary,
            config="--psm 7 --oem 1 -c tessedit_char_whitelist=0123456789.%"
        )
        
        text = text.strip().replace("%", "").replace(" ", "")
        if not text:
            return None
            
        percentage = float(text)
        if percentage > 100:
            percentage = percentage / 100  # Handle OCR errors like 1020 -> 102.0
        return percentage
    except Exception as e:
        print(f"Error extracting percentage: {e}")
        return None

def capture_state(
    max_seconds: float = 10.0,
    debug: bool = False,
) -> Tuple[np.ndarray, float]:
    """Capture frames for a duration with per-frame OCR.

    Args:
        max_seconds: Duration to capture in seconds.
        debug: If True, write captured frames to disk for debugging.

    Returns:
        Tuple[np.ndarray, float]: ``(screen_frame, percentage)``.
    """
    if debug:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    screen_bounds = _centered_bounds()
    percentage_bounds = {
        "top": screen_bounds["top"], 
        "left": screen_bounds["left"] + screen_bounds["width"] // 2 - 120, 
        "width": 200, 
        "height": 50
    }

    start = time.perf_counter()
    frame_count = 0
    successful_ocr = 0

    sct = mss.mss()

    try:
        while time.perf_counter() - start < max_seconds:


            screen_frame = np.array(sct.grab(screen_bounds))
            percentage_frame = np.array(sct.grab(percentage_bounds))

            pct_value = _extract_percentage(percentage_frame)
            
            if debug:
                engine = RandomActionEngine()
                engine.execute()
                cv2.imwrite(
                    str(OUTPUT_DIR / f"frame_{frame_count:05d}.png"),
                    cv2.cvtColor(screen_frame, cv2.COLOR_BGRA2BGR),
                )
                cv2.imwrite(
                    str(OUTPUT_DIR / f"percentage_{frame_count:05d}.png"),
                    cv2.cvtColor(percentage_frame, cv2.COLOR_BGRA2BGR),
                )
                print(f"Frame {frame_count:05d}: {pct_value:.2f}%")

            frame_count += 1

    finally:
        sct.close()

    if debug:
        elapsed = time.perf_counter() - start
        fps = frame_count / elapsed if elapsed > 0 else 0
        ocr_success_rate = (successful_ocr / frame_count * 100) if frame_count > 0 else 0

        print(f"\nCaptured {frame_count} frames in {elapsed:.2f}s (~{fps:.1f} FPS)")
        print(f"OCR success rate: {ocr_success_rate:.1f}% ({successful_ocr}/{frame_count})")

    return screen_frame, pct_value


if __name__ == "__main__":
    capture_state(max_seconds=5.0, debug=False)
