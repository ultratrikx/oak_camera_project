# 🥊 OAK Boxing Trainer

A first-person boxing game using OAK-1 AF camera with pose detection, ESP32 health management, and immersive audio.

## Features

-   **Real-time pose detection** using MediaPipe and OAK-1 AF camera
-   **Smart hit zones** that reposition automatically when health changes from any source
-   **ESP32 health system** with remote health monitoring and setting
-   **Campaign mode** with boss battles and cutscenes
-   **Background music** with volume controls
-   **Fullscreen support** for immersive gaming

## Hit Zone System

The game features an intelligent hit zone system that:

-   **Repositions dynamically** - Hit zones move to new random positions whenever health changes
-   **Multi-source health tracking** - Responds to health changes from both in-game hits and ESP32 external damage
-   **Visual feedback** - Square hit zones with pulsing effects and color coding
-   **Combo system** - Rewards consecutive hits with score multipliers

## Controls

-   `q` - Quit game
-   `r` - Reset current boss
-   `c` - Reset entire campaign
-   `f` - Toggle fullscreen
-   `+` - Increase music volume
-   `-` - Decrease music volume
-   `m` - Toggle music on/off
-   `ESC` - Exit fullscreen

## Setup

1. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

2. Connect OAK-1 AF camera

3. Set up ESP32 (optional) - see ESP32_INTEGRATION.md

4. Run the game:

    ```bash
    python oak_cam.py
    ```

## Music

The game includes background music (`music/bg_theme.mp3`) that:

-   Plays automatically when the game starts
-   Loops continuously during gameplay
-   Can be controlled with keyboard shortcuts
-   Runs at 30% volume by default for non-intrusive gameplay

## ESP32 Integration

See `ESP32_INTEGRATION.md` for details on setting up the ESP32 health management system.

## Testing

-   `test_music.py` - Test background music functionality
-   `test_esp32.py` - Test ESP32 API connectivity
-   `test_boss_advancement.py` - Test boss/campaign logic
-   `test_video.py` - Test cutscene video compatibility
-   `test_cutscene_manual.py` - Manual cutscene testing
