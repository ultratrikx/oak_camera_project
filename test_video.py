#!/usr/bin/env python3

import cv2
import os

def test_video_files():
    """Test if video files can be loaded and played"""
    cutscene_dir = "cutscenes"
    
    if not os.path.exists(cutscene_dir):
        print("❌ Cutscenes directory not found")
        return
    
    video_files = [f for f in os.listdir(cutscene_dir) if f.endswith('.mp4')]
    print(f"📁 Found {len(video_files)} video files: {video_files}")
    
    for video_file in video_files:
        video_path = os.path.join(cutscene_dir, video_file)
        print(f"\n🎬 Testing: {video_path}")
        
        # Test video loading
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Failed to open {video_file}")
            continue
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"✅ Video loaded successfully")
        print(f"   📐 Resolution: {width}x{height}")
        print(f"   🎞️ FPS: {fps:.1f}")
        print(f"   📊 Frames: {frame_count}")
        print(f"   ⏱️ Duration: {frame_count/fps:.1f}s")
        
        # Test reading first frame
        ret, frame = cap.read()
        if ret:
            print(f"   ✅ First frame read successfully")
        else:
            print(f"   ❌ Failed to read first frame")
        
        cap.release()

if __name__ == "__main__":
    test_video_files()
