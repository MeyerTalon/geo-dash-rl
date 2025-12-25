"""Level selection for Geometry Dash via Steam."""

from __future__ import annotations

import subprocess
import time
from typing import Dict, Tuple

from AppKit import NSScreen
from pynput.mouse import Button, Controller as MouseController


class LevelSelect:
    """Class for selecting and launching Geometry Dash levels."""

    def __init__(self) -> None:
        """Initialize the level selector."""
        self.level_functions: Dict[str, callable] = {
            "Stereo Madness": self.open_stereo_madness,
            # TODO: Add more level functions
            # "back_on_track": self.open_back_on_track,
            # "polargeist": self.open_polargeist,
        }

    def _screen_center(self) -> Tuple[int, int]:
        """Return the center coordinates of the primary macOS screen.

        Returns:
            Tuple of (x, y) center coordinates.
        """
        main = NSScreen.mainScreen()
        frame = main.frame()
        width = int(frame.size.width)
        height = int(frame.size.height)
        return (width // 2, height // 2)

    def open_stereo_madness(self) -> None:
        """Launch Geometry Dash via Steam, wait for it to load, then click twice at the center.

        Launches the game, waits for it to load, then clicks twice at the center
        of the primary screen (macOS) to navigate to Stereo Madness level.
        """
        try:
            # macOS typical: process is "Geometry Dash"
            # Other platforms may require different logic
            subprocess.run(["pkill", "-f", "Geometry Dash"], check=False)
            time.sleep(1)  # Give process time to close
        except Exception as e:
            pass

        # Launch via Steam
        subprocess.run(["open", "steam://rungameid/322170"])

        time.sleep(8)  # Give game time to fully launch

        # Click twice in the center of a 1280x720 window
        try:
            center_x, center_y = self._screen_center()
            center_y -= 100
            mouse: MouseController = MouseController()
            for _ in range(2):
                mouse.position = (center_x, center_y)
                time.sleep(0.1)
                mouse.press(Button.left)
                mouse.release(Button.left)
                time.sleep(1)
        except Exception as e:
            print(f"Failed to click: {e}")

    def select(self, level_name: str) -> None:
        """Select and launch a level by name.

        Args:
            level_name: Name of the level to launch (e.g., "stereo_madness").

        Raises:
            ValueError: If the level name is not found.
        """
        if level_name not in self.level_functions:
            available = ", ".join(self.level_functions.keys())
            raise ValueError(
                f"Unknown level: {level_name}. Available levels: {available}"
            )

        level_func = self.level_functions[level_name]
        level_func()

    # TODO: Implement additional level functions
    # def open_back_on_track(self) -> None:
    #     """Launch Back on Track level."""
    #     pass
    #
    # def open_polargeist(self) -> None:
    #     """Launch Polargeist level."""
    #     pass


if __name__ == "__main__":
    selector = LevelSelect()
    selector.select("stereo_madness")
