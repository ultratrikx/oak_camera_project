#!/usr/bin/env python3
"""
ESP32 Damage Application Script
This simulates how the ESP32 should handle damage when punches are detected.
Also includes the original fetch_damage function for compatibility.
"""

import requests
import time

class ESP32DamageHandler:
    """Simulates how ESP32 handles damage from boxing punches"""
    
    def __init__(self):
        self.boss_health = 100.0
        self.boss_max_health = 100.0
        self.current_net_force = 0.0
        self.max_force_score = 0.0
        
    def get_status(self):
        """Return current status in ESP32 API format"""
        return {
            "currentNetForce": round(self.current_net_force, 3),
            "maxForceScore": round(self.max_force_score, 3),
            "bossHealth": round(self.boss_health, 3),
            "bossMaxHealth": round(self.boss_max_health, 3)
        }
    
    def set_health(self, value):
        """Set boss health to specific value (used for level transitions)"""
        self.boss_health = max(0.0, min(float(value), self.boss_max_health))
        print(f"🎯 Health set to: {self.boss_health}")
        return {"status": "ok", "bossHealth": self.boss_health}
    
    def set_max_health(self, value):
        """Set maximum health for new boss"""
        self.boss_max_health = float(value)
        self.boss_health = self.boss_max_health
        print(f"👑 New boss with max health: {self.boss_max_health}")
        return {"status": "ok", "bossMaxHealth": self.boss_max_health, "bossHealth": self.boss_health}
    
    def apply_force_damage(self, force_value):
        """Apply damage based on detected force/punch"""
        # Update current force reading
        self.current_net_force = force_value
        
        # Update max force if this is a new record
        if force_value > self.max_force_score:
            self.max_force_score = force_value
        
        # Calculate damage based on force (you can adjust this formula)
        # Example: force of 4.0 = 20 damage, force of 2.0 = 10 damage
        damage = max(0, force_value * 5.0)  # Scale factor of 5
        
        # Apply damage to health
        previous_health = self.boss_health
        self.boss_health = max(0.0, self.boss_health - damage)
        
        print(f"💥 Force: {force_value:.2f} → Damage: {damage:.1f}")
        print(f"   Health: {previous_health:.1f} → {self.boss_health:.1f}")
        
        # Check for knockout
        if self.boss_health <= 0 and previous_health > 0:
            print("💀 KNOCKOUT! Boss defeated!")
        
        return {
            "force": force_value,
            "damage": damage,
            "bossHealth": self.boss_health,
            "knockout": self.boss_health <= 0
        }
    
    def reset_force_readings(self):
        """Reset force readings (call periodically)"""
        self.current_net_force = 0.0

# Original function for compatibility
def fetch_damage_values(url="http://172.20.10.2/"):
    """
    Fetches damage/force values from the specified server URL.
    Returns a list of numbers found in the response text.
    """
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        # Extract all numbers (integers or floats) from the text
        import re
        numbers = re.findall(r"[-+]?[0-9]*\.?[0-9]+", response.text)
        # Convert to float or int as appropriate
        parsed = [float(n) if '.' in n else int(n) for n in numbers]
        return parsed
    except Exception as e:
        print(f"Error fetching damage values: {e}")
        return []

if __name__ == "__main__":
    # Test the ESP32 damage handler simulation
    handler = ESP32DamageHandler()
    
    print("🤖 ESP32 Damage Handler Simulation")
    print("=" * 40)
    
    # Initial status
    print("Initial status:")
    print(handler.get_status())
    
    # Simulate some punches with different forces
    test_forces = [2.5, 4.2, 1.8, 5.1, 3.7, 6.0, 2.2]
    
    print(f"\n🥊 Simulating {len(test_forces)} punches:")
    for i, force in enumerate(test_forces):
        print(f"\nPunch {i+1}:")
        result = handler.apply_force_damage(force)
        
        if result["knockout"]:
            break
        
        # Simulate force decay
        time.sleep(0.1)
        handler.reset_force_readings()
    
    print(f"\n📊 Final status:")
    print(handler.get_status())
    
    # Simulate boss change
    print(f"\n🔄 Advancing to next boss...")
    handler.set_max_health(150)  # Next boss has more health
    print(handler.get_status())
    
    print("\n" + "=" * 40)
    print("Testing original fetch function:")
    values = fetch_damage_values()
    print("Fetched damage/force values:", values)
