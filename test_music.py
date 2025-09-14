#!/usr/bin/env python3
"""
Test script for background music functionality
"""

import pygame
import os
import time

def test_music():
    """Test background music loading and playback"""
    print("🎵 Testing background music system...")
    
    music_file = os.path.join(os.path.dirname(__file__), "music", "bg_theme.mp3")
    
    try:
        # Initialize pygame mixer
        pygame.mixer.pre_init(frequency=22050, size=-16, channels=2, buffer=512)
        pygame.mixer.init()
        print("✅ Pygame mixer initialized")
        
        # Check if music file exists
        if os.path.exists(music_file):
            print(f"✅ Found music file: {music_file}")
            
            # Load and play music
            pygame.mixer.music.load(music_file)
            pygame.mixer.music.set_volume(0.3)  # 30% volume
            pygame.mixer.music.play(-1)  # Loop indefinitely
            print("🎵 Music started (looping)")
            
            # Test for a few seconds
            print("Playing for 5 seconds...")
            time.sleep(5)
            
            # Test volume control
            print("🔊 Testing volume control...")
            pygame.mixer.music.set_volume(0.1)  # Lower volume
            print("Volume lowered to 10%")
            time.sleep(2)
            
            pygame.mixer.music.set_volume(0.5)  # Higher volume
            print("Volume raised to 50%")
            time.sleep(2)
            
            # Stop music
            pygame.mixer.music.stop()
            print("🎵 Music stopped")
            
            pygame.mixer.quit()
            print("✅ Test completed successfully!")
            
        else:
            print(f"❌ Music file not found: {music_file}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing music: {e}")
        return False
    
    return True

if __name__ == "__main__":
    test_music()
