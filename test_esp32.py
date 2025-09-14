#!/usr/bin/env python3
"""
Test script to verify ESP32 connectivity and health system
"""

import requests
import time

ESP32_IP = "172.20.10.2"

def test_esp32_health():
    """Test ESP32 health endpoints"""
    print("🔗 Testing ESP32 Health System")
    print(f"ESP32 IP: {ESP32_IP}")
    print("=" * 40)
    
    # Test status endpoint
    try:
        print("📡 Testing /status endpoint...")
        response = requests.get(f"http://{ESP32_IP}/status", timeout=2.0)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Status OK: {data}")
            # New API format: {"currentNetForce":3.060,"maxForceScore":4.446,"bossHealth":96.583,"bossMaxHealth":100.000}
            current_health = data.get('bossHealth', data.get('health', 'unknown'))
            max_health = data.get('bossMaxHealth', 'unknown')
            net_force = data.get('currentNetForce', 'unknown')
            max_force = data.get('maxForceScore', 'unknown')
            print(f"Boss Health: {current_health}/{max_health}")
            print(f"Force: {net_force} (Max: {max_force})")
        else:
            print(f"❌ Status failed: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Status endpoint error: {e}")
        return False
    
    # Test set_health endpoint
    try:
        print("\n📡 Testing /set_health endpoint...")
        test_health = 75
        response = requests.post(f"http://{ESP32_IP}/set_health?value={test_health}", timeout=2.0)
        if response.status_code == 200:
            print(f"✅ Set health to {test_health} OK")
            
            # Verify the change
            time.sleep(0.5)
            response = requests.get(f"http://{ESP32_IP}/status", timeout=2.0)
            if response.status_code == 200:
                data = response.json()
                new_health = data.get('bossHealth', data.get('health', 'unknown'))
                if isinstance(new_health, (int, float)) and abs(new_health - test_health) < 1:
                    print(f"✅ Health verified: {new_health}")
                else:
                    print(f"⚠️ Health mismatch: expected {test_health}, got {new_health}")
            else:
                print("❌ Failed to verify health change")
        else:
            print(f"❌ Set health failed: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Set health endpoint error: {e}")
        return False
    
    # Test latency
    print("\n⏱️ Testing latency (10 requests)...")
    latencies = []
    for i in range(10):
        try:
            start_time = time.time()
            response = requests.get(f"http://{ESP32_IP}/status", timeout=1.0)
            end_time = time.time()
            if response.status_code == 200:
                latency = (end_time - start_time) * 1000  # Convert to ms
                latencies.append(latency)
                print(f"Request {i+1}: {latency:.1f}ms")
            else:
                print(f"Request {i+1}: FAILED")
        except Exception as e:
            print(f"Request {i+1}: ERROR - {e}")
    
    if latencies:
        avg_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        print(f"\n📊 Latency Stats:")
        print(f"Average: {avg_latency:.1f}ms")
        print(f"Min: {min_latency:.1f}ms")
        print(f"Max: {max_latency:.1f}ms")
        
        if avg_latency < 100:
            print("✅ Latency is good for gaming!")
        else:
            print("⚠️ Latency might affect gameplay")
    
    print("\n🎯 ESP32 Health System Test Complete!")
    return True

if __name__ == "__main__":
    test_esp32_health()
