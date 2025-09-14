# ESP32 Health System Integration

## Overview

The boxing game now integrates with an ESP32 device on the mannequin to fetch real-time health values and synchronize game state.

## ESP32 Endpoints

### Status Endpoint

-   **URL**: `http://172.20.10.2/status`
-   **Method**: GET
-   **Response**: JSON with health and force data
-   **Example**: `{"currentNetForce":3.060,"maxForceScore":4.446,"bossHealth":96.583,"bossMaxHealth":100.000}`
-   **Fields**:
    -   `bossHealth`: Current health value
    -   `bossMaxHealth`: Maximum health for current boss
    -   `currentNetForce`: Current force being applied
    -   `maxForceScore`: Maximum force score achieved

### Set Health Endpoint

-   **URL**: `http://172.20.10.2/set_health?value=<health>`
-   **Method**: POST
-   **Parameter**: `value` - Health value to set (0-100)
-   **Example**: `http://172.20.10.2/set_health?value=100`

## Implementation Details

### Health Monitoring

-   Background thread continuously fetches health from ESP32 every 100ms
-   Thread-safe updates using locks
-   Graceful handling of connection failures
-   Low latency prioritized for responsive gameplay

### Game Integration

The boxing game integrates ESP32 health in several ways:

#### Health Monitoring

-   **Real-time polling**: ESP32 health is fetched every 100ms in a background thread
-   **Automatic sync**: Local game health stays synchronized with ESP32 values
-   **Change detection**: Hit zones automatically reposition when health changes from ANY source
-   **Thread Safety**: All ESP32 operations are thread-safe

#### Hit Zone Repositioning

The game now features intelligent hit zone management:

-   **Health-based repositioning**: Hit zones move to new random positions whenever health changes
-   **Multi-source detection**: Responds to health changes from both:
    -   In-game hand/punch hits detected by camera
    -   External ESP32 damage (e.g., physical sensors, network commands)
-   **Visual feedback**: Players see zones reposition immediately when health changes
-   **Performance optimized**: Only repositions when health actually changes, not on every frame

### Features

-   **Real-time Health**: Health values fetched every 100ms for low latency
-   **Connection Status**: Visual indicator in HUD shows ESP32 connection status
-   **Boss Synchronization**: ESP32 health updated when advancing to new bosses
-   **Error Handling**: Graceful degradation when ESP32 is offline
-   **Thread Safety**: All ESP32 operations are thread-safe

## Usage

### Running the Game

1. Ensure ESP32 is connected and accessible at `172.20.10.2`
2. Run the boxing game as normal
3. Health will be automatically synced with ESP32
4. Connection status shown in bottom-left corner

### Testing ESP32 Connection

Run the test script to verify ESP32 connectivity:

```bash
python test_esp32.py
```

This will test:

-   Status endpoint availability
-   Set health functionality
-   Network latency (important for gameplay)

### Game Controls

-   **'r'**: Reset current boss (syncs ESP32 health)
-   **'c'**: Reset entire campaign (syncs ESP32 to first boss health)
-   **'q'**: Quit game (stops ESP32 monitoring)

## Network Requirements

-   ESP32 must be accessible on local network at `172.20.10.2`
-   Low latency network connection recommended
-   Game continues functioning if ESP32 goes offline (uses last known health)

## Technical Notes

-   Health monitoring runs in separate daemon thread
-   500ms timeout for health fetching (prioritizes responsiveness)
-   1000ms timeout for health setting
-   Automatic reconnection handling
-   Thread cleanup on game exit
