#!/usr/bin/env python3
"""
Test script to verify boss defeat and advancement logic
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from oak_cam import BoxingGameDetector
import time

def test_boss_advancement():
    """Test boss advancement logic without camera"""
    print("🧪 Testing Boss Advancement Logic")
    print("=" * 40)
    
    # Create game instance
    game = BoxingGameDetector()
    
    # Wait for initialization
    time.sleep(2.0)
    
    print(f"Initial boss: {game.bosses[game.current_boss_index].name}")
    print(f"Initial health: {game.mannequin_health}/{game.max_health}")
    
    # Simulate health reaching 0
    print("\n🥊 Simulating health reaching 0...")
    game.mannequin_health = 0
    
    # Manually trigger game state update
    print("📊 Updating game state...")
    game._update_game_state()
    
    # Check if cutscene started
    if game.cutscene_playing:
        print("✅ Cutscene started successfully")
        print(f"🎬 Cutscene active: {game.cutscene_overlay['active']}")
        print(f"⏱️ Cutscene duration: {game.cutscene_overlay['duration']}s")
    else:
        print("❌ Cutscene did not start")
    
    # Wait for cutscene to finish
    if game.cutscene_playing:
        print("\n⏳ Waiting for cutscene to finish...")
        start_time = time.time()
        while game.cutscene_playing and (time.time() - start_time) < 10:
            # Simulate frame processing
            elapsed = time.time() - game.cutscene_overlay['start_time']
            if elapsed >= game.cutscene_overlay['duration']:
                print(f"🎬 Cutscene finished after {elapsed:.1f}s")
                game.cutscene_overlay['active'] = False
                game.cutscene_playing = False
                game._advance_to_next_boss()
                break
            time.sleep(0.1)
    
    # Check final state
    print(f"\n📊 Final State:")
    print(f"Current boss index: {game.current_boss_index}")
    if game.current_boss_index < len(game.bosses):
        print(f"Current boss: {game.bosses[game.current_boss_index].name}")
        print(f"Current health: {game.mannequin_health}/{game.max_health}")
    else:
        print("🏁 Campaign completed!")
    
    print(f"Campaign active: {game.campaign_active}")
    print(f"Cutscene playing: {game.cutscene_playing}")
    
    # Clean up
    game.stop_esp32_monitoring()
    
    return True

if __name__ == "__main__":
    test_boss_advancement()
