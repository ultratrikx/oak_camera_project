#!/usr/bin/env python3

import cv2
import numpy as np
import time
import os
from oak_cam import BoxingGameDetector

def test_cutscene():
    """Test cutscene playback"""
    print("🎬 Testing cutscene system")
    
    # Create a simple test frame
    test_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    cv2.putText(test_frame, "Testing Cutscene System", (200, 500), 
               cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 4)
    
    # Initialize the boxing game detector
    detector = BoxingGameDetector()
    
    # List available video files
    cutscene_dir = "cutscenes"
    if os.path.exists(cutscene_dir):
        video_files = [f for f in os.listdir(cutscene_dir) if f.endswith('.mp4')]
        print(f"📁 Available video files: {video_files}")
        
        if video_files:
            # Test with the first available video
            test_video = os.path.join(cutscene_dir, video_files[0])
            print(f"🎞️ Testing with: {test_video}")
            
            # Manually trigger cutscene
            detector.cutscene_overlay['active'] = True
            detector.cutscene_overlay['boss_name'] = "TEST BOSS"
            detector.cutscene_overlay['start_time'] = time.time()
            detector.cutscene_overlay['video_path'] = test_video
            detector.cutscene_overlay['duration'] = 10.0  # 10 second test
            detector.cutscene_playing = True
            
            print("🎬 Cutscene triggered! Press 'q' to quit, 's' to skip cutscene")
            
            # Create window
            cv2.namedWindow("Cutscene Test", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Cutscene Test", 1920, 1080)
            
            start_time = time.time()
            while True:
                current_time = time.time()
                
                # Process cutscene frame
                if detector.cutscene_overlay['active']:
                    result_frame = detector._render_fullscreen_cutscene(test_frame)
                    cv2.putText(result_frame, f"Elapsed: {current_time - start_time:.1f}s", 
                               (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                else:
                    result_frame = test_frame.copy()
                    cv2.putText(result_frame, "CUTSCENE FINISHED", (200, 500), 
                               cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 4)
                
                cv2.imshow("Cutscene Test", result_frame)
                
                # Handle input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    print("⏭️ Skipping cutscene")
                    detector.cutscene_overlay['active'] = False
                    detector.cutscene_playing = False
                
                # Auto-end after cutscene finishes
                if not detector.cutscene_overlay['active'] and detector.cutscene_playing == False:
                    time.sleep(2)  # Show "finished" message for 2 seconds
                    break
            
            cv2.destroyAllWindows()
            detector._cleanup_cutscene()
            print("✅ Cutscene test completed")
        else:
            print("❌ No video files found in cutscenes directory")
    else:
        print("❌ Cutscenes directory not found")

if __name__ == "__main__":
    test_cutscene()
