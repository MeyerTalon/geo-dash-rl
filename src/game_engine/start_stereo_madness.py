"""Start Geometry Dash Stereo Madness level via Steam."""

from __future__ import annotations

import subprocess
import time
from typing import Tuple

from AppKit import NSScreen
from pynput.mouse import Button, Controller as MouseController


def _screen_center() -> Tuple[int, int]:
  """
  Return the center coordinates of the primary macOS screen.
  """
  main = NSScreen.mainScreen()
  frame = main.frame()
  width = int(frame.size.width)
  height = int(frame.size.height)
  return (width // 2, height // 2)



def open_stereo_madness() -> None:
  """
  Launch Geometry Dash via Steam, wait for it to load,
  move the window to the top-left corner, then click twice
  at the center of the primary screen (macOS).
  """
  try:
    # macOS typical: process is "Geometry Dash"
    # Other platforms may require different logic
    subprocess.run(["pkill", "-f", "Geometry Dash"], check=False)
    time.sleep(1)  # Give process time to close
  except Exception as e:
    pass
  
  # Launch via Steam
  subprocess.run(['open', 'steam://rungameid/322170'])
  
  time.sleep(8)  # Give game time to fully launch

  # Click twice in the center of a 1280x720 window
  try:
    center_x, center_y = _screen_center()
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

if __name__ == "__main__":
    open_stereo_madness()