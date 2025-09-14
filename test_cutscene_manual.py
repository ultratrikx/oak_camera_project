#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(__file__))

from oak_cam import BoxingGameDetector, Boss
import cv2
import numpy as np
import time

def test_cutscene_system():
    """Test the cutscene system with manual trigger"""
    print("🎬 Testing cutscene system")
    
    # Create detector
    detector = BoxingGameDetector()
    
    # Create a test boss with existing video
    test_boss = Boss(
        name="TEST BOSS",
        max_health=100,
        cutscene="cutscenes/Evil_Bobby.mp4"
    )
    
    # Create a test frame
    test_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    
    # Manually trigger cutscene
    print("🎯 Triggering cutscene...")
    detector._play_cutscene(test_boss)
    
    # Create window
    cv2.namedWindow("Cutscene Test", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Cutscene Test", 1920, 1080)
    
    print("🎬 Starting cutscene playback...")
    print("Press 'q' to quit, 's' to skip")
    
    frame_count = 0
    while detector.cutscene_overlay['active']:
        # Get cutscene frame
        cutscene_frame = detector._render_fullscreen_cutscene(test_frame)
        
        # Add frame counter
        cv2.putText(cutscene_frame, f"Frame: {frame_count}", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
        # Show frame
        cv2.imshow("Cutscene Test", cutscene_frame)
        
        # Handle input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            detector.cutscene_overlay['active'] = False
        
        frame_count += 1
        time.sleep(0.033)  # ~30 FPS
    
    cv2.destroyAllWindows()
    detector._cleanup_cutscene()
    print("✅ Test completed")

if __name__ == "__main__":
    test_cutscene_system()
