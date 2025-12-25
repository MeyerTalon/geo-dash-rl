"""Miscellaneous game engine test script."""

from __future__ import annotations

import time

from action_engine import RandomActionEngine
from level_select import LevelSelect
from state_capture import StateCapture


if __name__ == "__main__":
    # Launch the level
    level_selector = LevelSelect()
    level_selector.select("Stereo Madness")
    
    # Wait a bit for the game to be ready
    time.sleep(2)
    
    # Create action engine for random actions
    action_engine = RandomActionEngine()
    
    # Capture state while executing random actions
    with StateCapture(debug=True) as capture:
        max_seconds = 60.0 * 1  # 1 minutes
        start_time = time.perf_counter()
        last_action_time = start_time
        action_interval = 0.01  # Execute action every 0.5 seconds
        
        frames = []
        percentages = []
        
        while time.perf_counter() - start_time < max_seconds:
            # Capture a frame
            frame, percentage = capture.capture_frame()
            frames.append(frame)
            percentages.append(percentage)


            
            # Execute random action at intervals
            current_time = time.perf_counter()
            if current_time - last_action_time >= action_interval:
                action = action_engine.execute()
                pct_str = f"{percentage:.2f}%" if percentage is not None else "N/A"
                print(
                    f"Frame {capture.frame_count:05d} | "
                    f"Action: {action.name:<10} | "
                    f"OCR Percentage: {pct_str}"
                )
                last_action_time = current_time
    
    print(f"Test completed!")
    print(f"Captured {len(frames)} frames")
    print(f"Highest percentage: {max(percentages)}%")
    print(f"Average percentage: {sum(percentages) / len(percentages):.2f}%")
    elapsed = time.perf_counter() - start_time
    fps = len(frames) / elapsed if elapsed > 0 else 0
    print(f"Average FPS: {fps:.2f}")
