#!/usr/bin/env python3
"""
ESP32 Damage Simulation Script
Simulates receiving damage on the ESP32 side for testing.
This would typically be implemented on the ESP32 itself.
"""

import time
import random

class ESP32Simulator:
    """Simulates ESP32 health system for testing"""
    
    def __init__(self):
        self.health = 100
        self.max_health = 100
        self.last_damage_time = 0
        
    def get_health(self):
        """Return current health"""
        return {"health": self.health}
    
    def set_health(self, value):
        """Set health to specific value"""
        self.health = max(0, min(self.max_health, int(value)))
        return {"status": "ok", "health": self.health}
    
    def apply_damage(self, damage):
        """Apply damage to health"""
        self.health = max(0, self.health - damage)
        self.last_damage_time = time.time()
        print(f"💥 Damage applied: {damage}, Health now: {self.health}")
        return {"status": "damaged", "health": self.health, "damage": damage}
    
    def simulate_random_damage(self):
        """Simulate random damage for testing"""
        if time.time() - self.last_damage_time > 2.0:  # Every 2 seconds
            damage = random.randint(5, 25)
            return self.apply_damage(damage)
        return None
    
    def reset_health(self, max_health=None):
        """Reset health to maximum"""
        if max_health:
            self.max_health = max_health
        self.health = self.max_health
        return {"status": "reset", "health": self.health}

# Example usage for testing the boxing game integration
if __name__ == "__main__":
    simulator = ESP32Simulator()
    
    print("🤖 ESP32 Health Simulator")
    print("This simulates what the ESP32 would do")
    print("=" * 40)
    
    # Simulate a boxing session
    print(f"Initial health: {simulator.get_health()}")
    
    # Simulate some punches
    test_damages = [15, 20, 12, 18, 25, 10]
    for i, damage in enumerate(test_damages):
        print(f"\nPunch {i+1}:")
        result = simulator.apply_damage(damage)
        print(f"Result: {result}")
        
        if simulator.health <= 0:
            print("💀 Knockout!")
            break
    
    # Reset for next boss
    print(f"\n🔄 Resetting for next boss...")
    simulator.reset_health(150)  # Next boss has more health
    print(f"New health: {simulator.get_health()}")
