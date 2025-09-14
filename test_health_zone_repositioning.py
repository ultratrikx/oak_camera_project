#!/usr/bin/env python3

"""
Test script to verify that hit zones reposition only when health changes,
regardless of the source (hand hits or ESP32 external changes).
"""

import time
import requests
import threading
from threading import Lock

# ESP32 Simulator URL (can be changed if needed)
ESP32_URL = "http://172.20.10.2"

def test_health_changes():
    """Test different health change scenarios"""
    
    # Test values
    initial_health = 100
    reduced_health = 80
    very_low_health = 20
    reset_health = 100
    
    print("🧪 Testing Health-Based Zone Repositioning")
    print("=" * 50)
    
    print(f"1. Setting initial health to {initial_health}")
    result = set_esp32_health(initial_health)
    print(f"   Result: {'✅ Success' if result else '❌ Failed'}")
    time.sleep(2)
    
    print(f"2. Reducing health to {reduced_health} (simulating external damage)")
    result = set_esp32_health(reduced_health)
    print(f"   Result: {'✅ Success' if result else '❌ Failed'}")
    print("   🎯 Hit zones should reposition now")
    time.sleep(3)
    
    print(f"3. Further reducing health to {very_low_health}")
    result = set_esp32_health(very_low_health)
    print(f"   Result: {'✅ Success' if result else '❌ Failed'}")
    print("   🎯 Hit zones should reposition again")
    time.sleep(3)
    
    print(f"4. Resetting health to {reset_health} (simulating boss change)")
    result = set_esp32_health(reset_health)
    print(f"   Result: {'✅ Success' if result else '❌ Failed'}")
    print("   🎯 Hit zones should reposition for new boss")
    time.sleep(2)
    
    print(f"5. Setting same health value again ({reset_health}) - no change expected")
    result = set_esp32_health(reset_health)
    print(f"   Result: {'✅ Success' if result else '❌ Failed'}")
    print("   ⭕ Hit zones should NOT reposition (no health change)")
    time.sleep(2)
    
    print("\n🏁 Test complete! Check game output for zone repositioning messages.")
    print("Expected: Zones should reposition 3 times (steps 2, 3, 4) but not in step 5")

def set_esp32_health(health_value):
    """Set ESP32 health via API"""
    try:
        url = f"{ESP32_URL}/set_health"
        response = requests.post(url, params={'value': health_value}, timeout=2.0)
        if response.status_code == 200:
            return True
        else:
            print(f"   ⚠️ HTTP {response.status_code}: {response.text}")
            return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def get_esp32_health():
    """Get current ESP32 health"""
    try:
        response = requests.get(f"{ESP32_URL}/status", timeout=2.0)
        if response.status_code == 200:
            data = response.json()
            health = data.get('bossHealth', data.get('health', None))
            return health
        return None
    except Exception as e:
        print(f"Error fetching health: {e}")
        return None

def monitor_health_changes():
    """Monitor and display health changes in real-time"""
    previous_health = None
    
    print("\n📊 Real-time Health Monitor (Press Ctrl+C to stop)")
    print("-" * 50)
    
    try:
        while True:
            current_health = get_esp32_health()
            if current_health is not None:
                if previous_health is None:
                    print(f"Initial health: {current_health}")
                elif current_health != previous_health:
                    print(f"Health changed: {previous_health} → {current_health}")
                previous_health = current_health
            
            time.sleep(0.5)  # Check every 500ms
    except KeyboardInterrupt:
        print("\n🛑 Health monitoring stopped")

if __name__ == "__main__":
    print("Health-Based Zone Repositioning Test")
    print("====================================")
    print("\nThis test will:")
    print("1. Change ESP32 health values programmatically")
    print("2. Verify that hit zones reposition only when health changes")
    print("3. Monitor real-time health changes")
    
    choice = input("\nChoose test mode:\n1. Automated health changes\n2. Real-time health monitor\n\nEnter choice (1 or 2): ").strip()
    
    if choice == "1":
        print("\n🚀 Starting automated health change test...")
        print("👀 Watch the game window for zone repositioning messages")
        time.sleep(2)
        test_health_changes()
    elif choice == "2":
        monitor_health_changes()
    else:
        print("Invalid choice. Exiting.")
