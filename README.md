# geo-dash-rl

## Pre-Requisites 
To set up the project, perform the following steps:

### 0. Install Geometry Dash (Steam) and Tesseract (OCR)

- Install Geometry Dash from Steam.
- Install Tesseract (used for OCR of the on-screen percentage). On macOS:

  ```bash
  brew install tesseract
  ```

---

### 1. Create and Activate the Conda Environment

1. Ensure you have [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/individual) installed.
2. From the root of the repository, create the environment using the provided `environment.yml`:

    ```bash
    conda env create -f environment.yml
    ```

3. Activate the environment:

    ```bash
    conda activate geo-dash-rl
    ```

---


### 2. Configure Geometry Dash Window Size

To ensure automation scripts work correctly, **set your Geometry Dash window size to exactly 1280x720**:

1. Open Geometry Dash.
2. Go to the in-game settings (click the gear icon).
3. Find the "Resolution" or "Window Size" option (typically under Video/Graphics settings).
4. Set the window to **1280x720**.
5. Make sure the game is in windowed mode (not fullscreen), and the window is positioned fully on screen.
6. In the level settings menu ensure that show percentage, and only show percentage, is enabled.

*Automation scripts assume the window is at (0, 0) on your desktop. If you encounter issues, manually drag the window to the top-left corner of your primary monitor.*



## Commands

**Run all commands from the geo-dash-rl directory.**
```bash
python src/game_engine/start_stereo_madness.py
```
```bash
python src/game_engine/collect_data.py
```