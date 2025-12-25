from capture_state import capture_state
from start_stereo_madness import open_stereo_madness
from actions import RandomActionEngine

if __name__ == "__main__":
    open_stereo_madness()
    capture_state(
        max_seconds=20.0, 
        debug=True
    )