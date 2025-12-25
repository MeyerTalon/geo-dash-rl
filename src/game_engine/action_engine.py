"""Action engine for controlling game inputs via mouse."""

from __future__ import annotations

import time
from enum import Enum
from typing import Tuple
import random
from AppKit import NSScreen
from pynput.mouse import Button, Controller as MouseController


class Action(Enum):
    """Action types for game control."""
    TAP = 0
    HOLD = 1
    NO_ACTION = 2


class ActionEngine:
    """Engine for executing game actions via mouse control.

    Provides methods for TAP, HOLD, and NO_ACTION operations at the screen center.
    """

    def __init__(self, click_position: Tuple[int, int] | None = None) -> None:
        """Initialize the action engine.

        Args:
            click_position: Optional (x, y) position to click. If None, uses screen center.
        """
        self.mouse = MouseController()
        if click_position is None:
            self.click_position = self._get_screen_center()
        else:
            self.click_position = click_position
        self.is_holding = False

    def _get_screen_center(self) -> Tuple[int, int]:
        """Get the center coordinates of the primary macOS screen.

        Returns:
            Tuple of (x, y) center coordinates.
        """
        try:
            main = NSScreen.mainScreen()
            frame = main.frame()
            width = int(frame.size.width)
            height = int(frame.size.height)
            center_x = width // 2
            center_y = height // 2
            # Adjust for Geometry Dash UI (similar to start_stereo_madness.py)
            center_y -= 100
            return (center_x, center_y)
        except Exception as e:
            print(f"Error getting screen center: {e}")
            return (640, 360)  # Fallback to default

    def tap(self, duration: float = 0.05) -> None:
        """Execute a quick tap (press and release).

        Args:
            duration: Time in seconds to hold the button down. Default 0.05s.
        """
        try:
            self.mouse.position = self.click_position
            time.sleep(0.01)  # Small delay for position to update
            self.mouse.press(Button.left)
            time.sleep(duration)
            self.mouse.release(Button.left)
        except Exception as e:
            print(f"Failed to tap: {e}")

    def hold(self) -> None:
        """Start holding the mouse button down."""
        if not self.is_holding:
            try:
                self.mouse.position = self.click_position
                time.sleep(0.01)  # Small delay for position to update
                self.mouse.press(Button.left)
                self.is_holding = True
            except Exception as e:
                print(f"Failed to start hold: {e}")

    def release(self) -> None:
        """Release the mouse button (end a hold)."""
        if self.is_holding:
            try:
                self.mouse.release(Button.left)
                self.is_holding = False
            except Exception as e:
                print(f"Failed to release: {e}")

    def no_action(self) -> None:
        """Execute no action (do nothing)."""
        # If currently holding, release
        if self.is_holding:
            self.release()

    def execute(self, action: Action) -> Action:
        """Execute an action based on the Action enum.

        Args:
            action: The action to execute (TAP, HOLD, or NO_ACTION).
        """
        if action == Action.TAP:
            self.tap()
        elif action == Action.HOLD:
            self.hold()
        elif action == Action.NO_ACTION:
            self.no_action()
        else:
            raise ValueError(f"Unknown action: {action}")
        return action

    def set_click_position(self, x: int, y: int) -> None:
        """Update the click position.

        Args:
            x: X coordinate.
            y: Y coordinate.
        """
        self.click_position = (x, y)

    def __del__(self) -> None:
        """Cleanup: release button if still holding."""
        if self.is_holding:
            try:
                self.mouse.release(Button.left)
            except Exception:
                pass


class RandomActionEngine(ActionEngine):
    """Engine for executing random actions."""
    def __init__(self, click_position: Tuple[int, int] | None = None) -> None:
        super().__init__(click_position)

    def execute(self) -> Action:
        """Execute a random action.
        Returns:
            The action executed.
        """
        action = random.choice([Action.TAP, Action.HOLD, Action.NO_ACTION])
        super().execute(action)
        return action


if __name__ == "__main__":
    # Example usage
    engine = ActionEngine()
    
    print("Testing TAP...")
    engine.execute(Action.TAP)
    time.sleep(0.5)
    
    print("Testing HOLD...")
    engine.execute(Action.HOLD)
    time.sleep(0.5)
    
    print("Testing NO_ACTION (releases hold)...")
    engine.execute(Action.NO_ACTION)
    time.sleep(0.5)
    
    print("Done!")
