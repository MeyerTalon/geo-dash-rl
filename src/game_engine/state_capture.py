"""Optimized screen capture with per-frame OCR of the in-game percentage."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import mss
import numpy as np
import pytesseract

OUTPUT_DIR = Path(__file__).resolve().parent / "frames"
CAPTURE_SIZE = {"width": 1280, "height": 720}


class StateCapture:
    """Class for capturing game state frames with OCR percentage extraction."""

    def __init__(self, output_dir: Path | None = None, debug: bool = False) -> None:
        """Initialize the state capture engine.

        Args:
            output_dir: Directory to save frames if debug=True. Defaults to frames/.
            debug: If True, save captured frames to disk.
        """
        self.output_dir = output_dir or OUTPUT_DIR
        self.debug = debug
        self.sct = mss.mss()
        self.screen_bounds = self._centered_bounds()
        self.percentage_bounds = {
            "top": self.screen_bounds["top"],
            "left": self.screen_bounds["left"] + self.screen_bounds["width"] // 2 - 120,
            "width": 200,
            "height": 50,
        }
        self.frame_count = 0

    def _centered_bounds(self) -> Dict[str, int]:
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

    def _extract_percentage(self, frame_array: np.ndarray) -> Optional[float]:
        """Run fast OCR on a cropped percentage region.

        Args:
            frame_array: BGRA image array containing only the percentage area.

        Returns:
            float: Parsed percentage (0-100) if OCR succeeds, otherwise -1.0.
        """
        try:
            # Convert BGRA to grayscale (much faster than PIL)
            gray = cv2.cvtColor(frame_array, cv2.COLOR_BGRA2GRAY)

            # Threshold for better OCR (white text on dark background)
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

            # OCR with optimized config
            text = pytesseract.image_to_string(
                binary,
                config="--psm 7 --oem 1 -c tessedit_char_whitelist=0123456789.%",
            )

            text = text.strip().replace("%", "").replace(" ", "")
            if not text:
                # Return -1.0 to indicate OCR error
                return -1.0

            percentage = float(text)
            if percentage > 100:
                percentage = percentage / 100  # Handle OCR errors like 1020 -> 102.0
            return percentage
        except Exception as e:
            print(f"Error extracting percentage: {e}")
            return -1.0

    def capture_frame(self) -> Tuple[np.ndarray, Optional[float]]:
        """Capture a single frame with percentage extraction.

        Returns:
            Tuple[np.ndarray, Optional[float]]: ``(screen_frame, percentage)``.
                screen_frame is BGRA format, percentage is 0-100 or None.
        """
        screen_frame = np.array(self.sct.grab(self.screen_bounds))
        percentage_frame = np.array(self.sct.grab(self.percentage_bounds))

        pct_value = self._extract_percentage(percentage_frame)

        if self.debug:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(
                str(self.output_dir / f"frame_{self.frame_count:05d}.png"),
                cv2.cvtColor(screen_frame, cv2.COLOR_BGRA2BGR),
            )
            cv2.imwrite(
                str(self.output_dir / f"percentage_{self.frame_count:05d}.png"),
                cv2.cvtColor(percentage_frame, cv2.COLOR_BGRA2BGR),
            )


        self.frame_count += 1
        return screen_frame, pct_value

    def capture_frames(
        self,
        max_seconds: float = 10.0,
    ) -> Tuple[List[np.ndarray], List[Optional[float]], Dict[str, float]]:
        """Capture frames for a specified duration with per-frame OCR.

        Args:
            max_seconds: Duration to capture in seconds.

        Returns:
            Tuple containing:
                - List of screen frames (BGRA format)
                - List of percentage values (0-100 or None)
                - Dictionary with capture statistics (fps, ocr_success_rate, etc.)
        """
        if self.debug:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        start = time.perf_counter()
        frames = []
        percentages = []
        successful_ocr = 0

        try:
            while time.perf_counter() - start < max_seconds:
                screen_frame, pct_value = self.capture_frame()
                frames.append(screen_frame)
                percentages.append(pct_value)

                if pct_value is not None:
                    successful_ocr += 1

        finally:
            self.sct.close()

        elapsed = time.perf_counter() - start
        fps = len(frames) / elapsed if elapsed > 0 else 0
        ocr_success_rate = (successful_ocr / len(frames) * 100) if len(frames) > 0 else 0

        stats = {
            "fps": fps,
            "ocr_success_rate": ocr_success_rate,
            "total_frames": len(frames),
            "successful_ocr": successful_ocr,
            "elapsed_seconds": elapsed,
        }

        if self.debug:
            print(f"\nCaptured {len(frames)} frames in {elapsed:.2f}s (~{fps:.1f} FPS)")
            print(f"OCR success rate: {ocr_success_rate:.1f}% ({successful_ocr}/{len(frames)})")

        return frames, percentages, stats

    def __enter__(self) -> "StateCapture":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - cleanup resources."""
        self.sct.close()

    def __del__(self) -> None:
        """Cleanup on deletion."""
        try:
            self.sct.close()
        except Exception:
            pass


if __name__ == "__main__":
    # Example usage
    with StateCapture(debug=False) as capture:
        # Capture a single frame
        frame, percentage = capture.capture_frame()
        print(f"Single frame captured, percentage: {percentage}")

        # Capture multiple frames
        frames, percentages, stats = capture.capture_frames(max_seconds=5.0)
        print(f"Captured {len(frames)} frames")
        print(f"Stats: {stats}")
