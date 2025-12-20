import time
from pynput.mouse import Button, Controller as MouseController

# Install: pip install pynput

def main():
    mouse = MouseController()
    frame_count = 0

    print("Starting in 3 seconds... Focus on your game window!")
    time.sleep(3)

    try:
        while True:  # Single loop
            if frame_count % 100 == 0:
                # Move, tiny jiggle, then explicit press/release with short delays
                mouse.position = (400, 400)
                time.sleep(0.02)
                mouse.position = (401, 400)
                time.sleep(0.01)
                mouse.position = (400, 400)
                time.sleep(0.01)
                mouse.press(Button.left)
                time.sleep(0.02)
                mouse.release(Button.left)
                print(f"Frame {frame_count}: clicked at (400, 400)")

            frame_count += 1
            time.sleep(1/60)  # ~60 FPS frame cadence
    except KeyboardInterrupt:
        print(f"Stopped at frame {frame_count}")


if __name__ == "__main__":
    main()