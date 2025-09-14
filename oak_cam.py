#!/usr/bin/env python3

import cv2
import numpy as np
import mediapipe as mp
import time
import math
import depthai as dai
import random
from typing import Optional, Tuple, List
from dataclasses import dataclass
import os
import subprocess
import requests
import threading
from threading import Lock
import json
import pygame

@dataclass
class HitZone:
    """Represents a target zone on the mannequin"""
    name: str
    center: Tuple[int, int]
    size: int  # Changed from radius to size for square boxes
    color: Tuple[int, int, int]
    active: bool = True
    hit_damage: int = 10
    last_hit_time: float = 0
    hit_cooldown: float = 1.0  # seconds


@dataclass
class Boss:
    """Represents a campaign boss"""
    name: str
    max_health: int
    portrait: Optional[str] = None  # path to image to show, optional
    cutscene: Optional[str] = None  # path to video file to play on defeat
    defeated: bool = False

class BoxingGameDetector:
    """Boxing game with pose detection and hit zones"""
    
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize pose and hand detection with very relaxed settings
        self.pose = self.mp_pose.Pose(
            model_complexity=0,
            min_detection_confidence=0.4,  # Even lower
            min_tracking_confidence=0.2,   # Even lower
            enable_segmentation=False,
            smooth_landmarks=True
        )
        
        self.hands = self.mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.4,  # Lower for easier detection
            min_tracking_confidence=0.2,   # Lower for easier tracking
            max_num_hands=2
        )
        
        # Game state
        self.mannequin_health = 100
        self.max_health = 100
        self.score = 0
        self.combo_count = 0
        self.last_hit_time = 0
        self.combo_timeout = 2.0  # seconds
        
        # Hit zones (will be dynamically positioned based on detected pose)
        self.hit_zones = []
        self.hand_positions = []
        self.last_hand_positions = []
        
        # Visual effects
        self.hit_effects = []  # Store hit effect animations
        self.punch_speed_threshold = 50  # pixels per frame for punch detection
        
        # Game-style hit zones
        self.base_zones = []  # Store base zone positions
        self.zones_need_repositioning = True  # Flag to trigger initial setup
        
        # FPS tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        # Campaign / bosses
        self.bosses = []
        self.current_boss_index = 0
        self._init_campaign()
        self.campaign_active = True
        self.cutscene_playing = False
        # Smoothing state for pose and hands to reduce flicker
        self.landmark_smooth = {}  # keyed by landmark enum name -> (x_px, y_px)
        self.smooth_alpha = 0.45  # EMA alpha (0-1), higher = more responsive, lower = smoother
        self.hand_smooth_positions = []  # list of smoothed hand positions by index
        self.hand_prev_positions = []  # previous raw positions used to compute approach
        self.hand_velocities = []  # per-hand velocity vector (dx, dy)
        # Tunable thresholds
        # Make hits easier: lower base speed and relax approach requirement
        self.base_punch_speed = 30  # lowered base pixels/frame threshold (more forgiving)
        self.approach_dot_threshold = 0.2  # relaxed approach requirement
        self.torso_size = 200  # baseline torso size (px) to scale thresholds
        # Overlay state for in-app cutscene playback
        self.cutscene_overlay = {
            'active': False,
            'boss_name': None,
            'start_time': 0,
            'duration': 3.0,
            'portrait': None,
            'video_cap': None,
            'audio_proc': None,
            'fullscreen_mode': False,
            'frame_buffer': None,
            'video_fps': 30.0,
            'last_frame_time': 0,
            'end_delay': 1.0  # Extra delay at end to ensure full playback
        }
        
        # Background Music System
        self.music_initialized = False
        self.music_volume = 0.3  # Soft volume (30%)
        self.music_file = os.path.join(os.path.dirname(__file__), "music", "bg_theme.mp3")
        self.music_playing = False
        
        # Initialize background music
        self._init_background_music()
        
        # ESP32 Health System
        self.esp32_ip = "172.20.10.2"
        self.esp32_health = 100  # Local cache of ESP32 health
        self.esp32_lock = Lock()  # Thread safety for health updates
        self.esp32_last_fetch = 0
        self.esp32_fetch_interval = 0.1  # Fetch every 100ms for low latency
        self.esp32_connected = False
        self.esp32_thread = None
        self.esp32_running = True
        self.previous_health = 100  # Track previous health to detect changes
        
        # Start ESP32 health monitoring thread
        self._start_esp32_monitoring()
        
        # Initialize ESP32 health after a short delay to allow monitoring to start
        threading.Timer(0.5, self.initialize_esp32_health).start()
        
    def detect_and_process(self, frame: np.ndarray) -> np.ndarray:
        """Main detection and game logic - with frame validation"""
        # Validate input frame
        if frame is None or frame.size == 0:
            print("⚠️ Invalid frame received")
            return np.zeros((360, 480, 3), dtype=np.uint8)  # Return black frame
        
        # Create game frame first
        game_frame = frame.copy()
        
        # If cutscene is playing, only show cutscene (disable all game processing)
        if self.cutscene_overlay['active']:
            return self._render_fullscreen_cutscene(game_frame)
        
        try:
            # Only process every 3rd frame to reduce load
            self.fps_counter += 1
            if self.fps_counter % 3 != 0:
                # Skip heavy processing, just draw cached elements
                self._draw_hit_zones(game_frame)
                self._draw_hit_effects(game_frame)
                self._draw_hud(game_frame)
                return game_frame
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Try pose detection first
            pose_results = None
            try:
                pose_results = self.pose.process(rgb_frame)
            except Exception as e:
                print(f"Pose detection error: {e}")
            
            # Process mannequin if pose detected
            if pose_results and pose_results.pose_landmarks:
                try:
                    self._setup_hit_zones(game_frame, pose_results.pose_landmarks)
                    self._draw_mannequin(game_frame, pose_results.pose_landmarks)
                except Exception as e:
                    print(f"Pose processing error: {e}")
            
            # Try hand detection only if we're not overloaded
            hand_results = None
            try:
                hand_results = self.hands.process(rgb_frame)
            except Exception as e:
                print(f"Hand detection error: {e}")
            
            # Process hands if detected
            if hand_results and hand_results.multi_hand_landmarks:
                try:
                    self._process_hands(game_frame, hand_results.multi_hand_landmarks)
                    self._check_hits()
                except Exception as e:
                    print(f"Hand processing error: {e}")
            
        except Exception as e:
            print(f"⚠️ Detection error: {e}")
            # Continue with basic display
        
        # Always draw game elements
        try:
            self._draw_hit_zones(game_frame)
            self._draw_hit_effects(game_frame)
            self._draw_hud(game_frame)
        except Exception as e:
            print(f"Drawing error: {e}")
        
        # Update game state
        self._update_game_state()
        self._update_fps()
        
        return game_frame
    
    def _setup_hit_zones(self, frame, landmarks):
        """Setup hit zones with game-style variation"""
        h, w = frame.shape[:2]
        current_time = time.time()

        # Helper to smooth a landmark to pixel coordinates using EMA
        def smooth_landmark(name, lm):
            x_px = int(lm.x * w)
            y_px = int(lm.y * h)
            if name in self.landmark_smooth:
                prev_x, prev_y = self.landmark_smooth[name]
                x_px = int(prev_x * (1 - self.smooth_alpha) + x_px * self.smooth_alpha)
                y_px = int(prev_y * (1 - self.smooth_alpha) + y_px * self.smooth_alpha)
            self.landmark_smooth[name] = (x_px, y_px)
            return x_px, y_px

        # Get key landmarks with smoothing for base positioning
        nose = landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
        left_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]

        head_x, head_y = smooth_landmark('NOSE', nose)
        ls_x, ls_y = smooth_landmark('L_SHOULDER', left_shoulder)
        rs_x, rs_y = smooth_landmark('R_SHOULDER', right_shoulder)

        # Estimate torso width/size to scale zones and thresholds
        torso_width = max(20, int(abs(ls_x - rs_x)))
        self.torso_size = int(torso_width * 1.6)

        # Store base positions ONLY if not set or if zones need repositioning (after hit)
        if not self.base_zones or self.zones_need_repositioning:
            chest_x = int((ls_x + rs_x) / 2)
            chest_y = int((ls_y + rs_y) / 2 + self.torso_size * 0.12)
            
            # Add random variation ONLY when repositioning
            head_offset_x = random.randint(-40, 40)
            head_offset_y = random.randint(-20, 20)
            chest_offset_x = random.randint(-40, 40)
            chest_offset_y = random.randint(-20, 20)
            
            self.base_zones = [
                {
                    'name': 'HEAD', 
                    'base_pos': (head_x + head_offset_x, head_y + head_offset_y), 
                    'damage': 20, 
                    'color': (255, 100, 100)
                },
                {
                    'name': 'CHEST', 
                    'base_pos': (chest_x + chest_offset_x, chest_y + chest_offset_y), 
                    'damage': 15, 
                    'color': (100, 255, 255)
                },
            ]
            self.zones_need_repositioning = False
            print(f"🎯 Repositioned hit zones - HEAD: ({head_x + head_offset_x}, {head_y + head_offset_y}), CHEST: ({chest_x + chest_offset_x}, {chest_y + chest_offset_y})")

        # Clear existing zones and create new ones using FIXED positions
        self.hit_zones.clear()

        for zone_info in self.base_zones:
            # Use the stored fixed position - NO random variation here
            final_x, final_y = zone_info['base_pos']
            
            # Ensure zones stay within frame bounds
            final_x = max(50, min(w-50, final_x))
            final_y = max(50, min(h-50, final_y))
            
            # Square hit zones - size based on torso
            zone_size = max(60, int(self.torso_size * 0.4))
            
            self.hit_zones.append(HitZone(
                name=zone_info['name'],
                center=(final_x, final_y),
                size=zone_size,
                color=zone_info['color'],
                hit_damage=zone_info['damage'],
                last_hit_time=getattr(self, f"{zone_info['name'].lower()}_last_hit", 0)
            ))
    
    def _draw_mannequin(self, frame, landmarks):
        """Draw mannequin pose - simplified for game aesthetic"""
        # Don't draw pose landmarks for cleaner game look
        pass
    
    def _process_hands(self, frame, hand_landmarks):
        """Process player hand positions and movements"""
        # Smooth hand positions and compute velocities
        self.hand_prev_positions = self.hand_positions.copy()
        self.hand_positions.clear()

        h, w = frame.shape[:2]
        for idx, hand_landmark in enumerate(hand_landmarks):
            wrist = hand_landmark.landmark[self.mp_hands.HandLandmark.WRIST]
            hand_x = int(wrist.x * w)
            hand_y = int(wrist.y * h)

            # Smooth position with EMA per index
            if idx < len(self.hand_smooth_positions):
                prev_x, prev_y = self.hand_smooth_positions[idx]
                sx = int(prev_x * (1 - self.smooth_alpha) + hand_x * self.smooth_alpha)
                sy = int(prev_y * (1 - self.smooth_alpha) + hand_y * self.smooth_alpha)
                self.hand_smooth_positions[idx] = (sx, sy)
            else:
                # initialize
                sx, sy = hand_x, hand_y
                if idx >= len(self.hand_smooth_positions):
                    self.hand_smooth_positions.append((sx, sy))

            self.hand_positions.append((sx, sy))

            # Velocity (dx, dy) from raw previous to current
            if idx < len(self.hand_prev_positions):
                prev_raw = self.hand_prev_positions[idx]
                dx = sx - prev_raw[0]
                dy = sy - prev_raw[1]
            else:
                dx, dy = 0, 0

            if idx < len(self.hand_velocities):
                self.hand_velocities[idx] = (dx, dy)
            else:
                self.hand_velocities.append((dx, dy))

            # Don't draw hand tracking for cleaner game look

    
    def _check_hits(self):
        """Check if hands hit any target zones"""
        current_time = time.time()
        for i, hand_pos in enumerate(self.hand_positions):
            # Use smoothed velocity if available
            dx, dy = (0, 0)
            if i < len(self.hand_velocities):
                dx, dy = self.hand_velocities[i]

            hand_speed = math.sqrt(dx * dx + dy * dy)

            # Scale threshold to torso size so users at different distances work
            adaptive_threshold = max(8, int(self.base_punch_speed * (self.torso_size / 200.0)))

            # Only consider a punch if the speed exceeds adaptive threshold
            if hand_speed < adaptive_threshold:
                continue

            # Check collision with zones and require approach towards zone center
            for zone in self.hit_zones:
                if not zone.active or (current_time - zone.last_hit_time) < zone.hit_cooldown:
                    continue

                zx, zy = zone.center
                vx = zx - hand_pos[0]
                vy = zy - hand_pos[1]
                dist = math.hypot(vx, vy)

                # Allow a small margin so near-misses still count
                margin = max(8, int(zone.size * 0.12))
                if dist > (zone.size//2 + margin):
                    continue

                # Normalize approach vector and velocity to compute dot product
                vlen = hand_speed
                if dist > 1:
                    ax, ay = vx / dist, vy / dist
                    if vlen > 0:
                        vxn, vyn = dx / vlen, dy / vlen
                        approach_dot = ax * vxn + ay * vyn
                    else:
                        approach_dot = 0
                else:
                    approach_dot = 1.0

                # Require the hand to be moving roughly toward the center (dot product positive above threshold)
                # Relaxed approach requirement: allow borderline angles if speed is high
                speed_bonus = 1.0 if vlen > adaptive_threshold * 1.5 else 0.0
                if approach_dot >= self.approach_dot_threshold or speed_bonus:
                    # Register hit with measured speed
                    self._register_hit(zone, hand_pos, hand_speed)
                    break
    
    def _register_hit(self, zone: HitZone, hit_pos: Tuple[int, int], speed: float):
        """Register a successful hit"""
        current_time = time.time()
        
        # Update zone
        zone.last_hit_time = current_time
        if zone.name == "HEAD":
            self.head_last_hit = current_time
        elif zone.name == "CHEST":
            self.chest_last_hit = current_time
        # Calculate damage based on speed and distance scaling
        # Allow up to 3x multiplier for strong hits and slightly boost by torso scale
        speed_multiplier = min(speed / 100, 3.0)  # Max 3x damage
        torso_factor = max(0.9, min(1.3, self.torso_size / 200.0))
        damage = int(zone.hit_damage * speed_multiplier * torso_factor)

        # NOTE: Health is now managed by ESP32, not directly modified here
        # The ESP32 will handle damage application and health reduction
        # We just track score and combo locally
        self.score += damage * 10
        
        # Update combo
        if (current_time - self.last_hit_time) < self.combo_timeout:
            self.combo_count += 1
        else:
            self.combo_count = 1
        
        self.last_hit_time = current_time
        
        # Add hit effect
        self.hit_effects.append({
            'pos': hit_pos,
            'damage': damage,
            'time': current_time,
            'zone': zone.name,
            'duration': 1.0
        })
        
        print(f"🥊 {zone.name} HIT! Damage: {damage}, Speed: {speed:.1f}, Combo: {self.combo_count}")

        # Note: Zone repositioning will be triggered automatically when ESP32 health changes
        # No need to set zones_need_repositioning here since health changes are detected in ESP32 monitor

        # If boss health reaches zero, handle defeat
        if self.mannequin_health <= 0 and self.campaign_active and not self.cutscene_playing:
            # Mark boss defeated
            current_boss = self.bosses[self.current_boss_index]
            current_boss.defeated = True
            print(f"🎉 Boss defeated: {current_boss.name}")
            # Play cutscene (non-blocking start)
            try:
                self.cutscene_playing = True
                self._play_cutscene(current_boss)
            except Exception as e:
                print(f"Cutscene playback error: {e}")
                # Immediately advance if cutscene failed
                self._advance_to_next_boss()
    
    def _draw_hit_zones(self, frame):
        """Draw target zones as game-style squares"""
        current_time = time.time()
        
        for zone in self.hit_zones:
            if not zone.active:
                continue
            
            # Calculate zone appearance based on cooldown
            time_since_hit = current_time - zone.last_hit_time
            alpha = min(time_since_hit / zone.hit_cooldown, 1.0)
            
            # Zone color with pulsing effect for active zones
            base_color = zone.color
            if alpha >= 1.0:
                # Add pulsing effect when zone is ready
                pulse = (math.sin(current_time * 4) + 1) * 0.3 + 0.4  # 0.4 to 1.0
                color = tuple(int(c * pulse) for c in base_color)
            else:
                # Dim when on cooldown
                color = tuple(int(c * alpha * 0.5) for c in base_color)
            
            # Draw square zone
            half_size = zone.size // 2
            top_left = (zone.center[0] - half_size, zone.center[1] - half_size)
            bottom_right = (zone.center[0] + half_size, zone.center[1] + half_size)
            
            # Outer glow effect
            if alpha >= 1.0:
                glow_size = half_size + 8
                glow_color = tuple(int(c * 0.3) for c in color)
                cv2.rectangle(frame, 
                            (zone.center[0] - glow_size, zone.center[1] - glow_size),
                            (zone.center[0] + glow_size, zone.center[1] + glow_size),
                            glow_color, 4)
            
            # Main zone rectangle
            cv2.rectangle(frame, top_left, bottom_right, color, 4)
            
            # Inner targeting reticle
            if alpha >= 1.0:
                reticle_size = half_size // 3
                # Horizontal line
                cv2.line(frame,
                        (zone.center[0] - reticle_size, zone.center[1]),
                        (zone.center[0] + reticle_size, zone.center[1]),
                        color, 2)
                # Vertical line
                cv2.line(frame,
                        (zone.center[0], zone.center[1] - reticle_size),
                        (zone.center[0], zone.center[1] + reticle_size),
                        color, 2)
            
            # Zone damage indicator
            damage_text = f"{zone.hit_damage}"
            text_size = cv2.getTextSize(damage_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            text_pos = (zone.center[0] - text_size[0]//2, zone.center[1] + text_size[1]//2)
            
            if alpha >= 1.0:
                cv2.putText(frame, damage_text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 
                           0.8, (255, 255, 255), 2, cv2.LINE_AA)
    
    def _draw_hit_effects(self, frame):
        """Draw hit effect animations with game-style effects"""
        current_time = time.time()
        effects_to_remove = []
        
        for i, effect in enumerate(self.hit_effects):
            time_elapsed = current_time - effect['time']
            if time_elapsed > effect['duration']:
                effects_to_remove.append(i)
                continue
            
            # Animation progress (0 to 1)
            progress = time_elapsed / effect['duration']
            
            # Multiple effect layers for game feel
            alpha = 1.0 - progress
            
            # Expanding square effect
            size = int(20 + progress * 60)
            half_size = size // 2
            
            # Color based on zone with intensity
            if effect['zone'] == 'HEAD':
                color = (int(255 * alpha), int(100 * alpha), int(100 * alpha))
            else:
                color = (int(100 * alpha), int(255 * alpha), int(255 * alpha))
            
            # Draw expanding square
            top_left = (effect['pos'][0] - half_size, effect['pos'][1] - half_size)
            bottom_right = (effect['pos'][0] + half_size, effect['pos'][1] + half_size)
            cv2.rectangle(frame, top_left, bottom_right, color, 3)
            
            # Inner flash effect
            if progress < 0.3:  # Flash for first 30% of animation
                flash_alpha = (0.3 - progress) / 0.3
                flash_color = (int(255 * flash_alpha), int(255 * flash_alpha), int(255 * flash_alpha))
                inner_size = int(size * 0.6)
                inner_half = inner_size // 2
                cv2.rectangle(frame, 
                            (effect['pos'][0] - inner_half, effect['pos'][1] - inner_half),
                            (effect['pos'][0] + inner_half, effect['pos'][1] + inner_half),
                            flash_color, -1)
            
            # Damage text with outline
            damage_text = f"-{effect['damage']}"
            text_y = effect['pos'][1] - size - 20
            text_size = cv2.getTextSize(damage_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
            text_pos = (effect['pos'][0] - text_size[0]//2, text_y)
            
            # Text outline
            cv2.putText(frame, damage_text, text_pos, 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 5, cv2.LINE_AA)
            # Text fill
            cv2.putText(frame, damage_text, text_pos, 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv2.LINE_AA)
        
        # Remove expired effects
        for i in reversed(effects_to_remove):
            del self.hit_effects[i]
    
    def _draw_hud(self, frame):
        """Draw minimal game-style HUD"""
        h, w = frame.shape[:2]
        
        # Health bar - top right
        bar_width = 250
        bar_height = 20
        bar_x = w - bar_width - 30
        bar_y = 30
        
        # Health bar background with border
        cv2.rectangle(frame, (bar_x-2, bar_y-2), (bar_x + bar_width+2, bar_y + bar_height+2), 
                     (255, 255, 255), 1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                     (40, 40, 40), -1)
        
        # Health bar fill with color coding
        if self.mannequin_health > 0:
            health_width = int((self.mannequin_health / self.max_health) * bar_width)
            if self.mannequin_health > 60:
                health_color = (0, 255, 0)
            elif self.mannequin_health > 30:
                health_color = (0, 255, 255)
            else:
                health_color = (0, 100, 255)
            
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + health_width, bar_y + bar_height), 
                         health_color, -1)
        
        # Health text
        health_text = f"{self.mannequin_health}/{self.max_health} HP"
        cv2.putText(frame, health_text, (bar_x, bar_y - 8), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Score - top left
        score_text = f"SCORE: {self.score:,}"
        cv2.putText(frame, score_text, (30, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
        
        # Combo - below score
        if self.combo_count > 1:
            combo_color = (0, 255, 255) if self.combo_count < 5 else (255, 100, 255)
            combo_text = f"COMBO x{self.combo_count}!"
            cv2.putText(frame, combo_text, (30, 85), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, combo_color, 2, cv2.LINE_AA)
        
        # Boss info - center top
        if self.campaign_active and self.bosses:
            boss = self.bosses[self.current_boss_index]
            boss_text = f"{boss.name}"
            text_size = cv2.getTextSize(boss_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
            text_x = (w - text_size[0]) // 2
            
            # Boss name with outline
            cv2.putText(frame, boss_text, (text_x, 45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(frame, boss_text, (text_x, 45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 255), 2, cv2.LINE_AA)
        
        # ESP32 status - bottom left corner
        status_color = (0, 255, 0) if self.esp32_connected else (0, 0, 255)
        status_text = "●" if self.esp32_connected else "●"
        cv2.putText(frame, status_text, (30, h - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2, cv2.LINE_AA)
        cv2.putText(frame, "ESP32", (55, h - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Victory message
        if self.mannequin_health <= 0:
            if self.campaign_active:
                boss = self.bosses[self.current_boss_index]
                msg = f"{boss.name} DEFEATED!"
            else:
                msg = "KNOCKOUT!"

            victory_text = f"🏆 {msg} 🏆"
            text_size = cv2.getTextSize(victory_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 4)[0]
            text_x = (w - text_size[0]) // 2
            text_y = h // 2

            # Victory background
            padding = 30
            cv2.rectangle(frame, (text_x - padding, text_y - 40), 
                         (text_x + text_size[0] + padding, text_y + 20), (0, 0, 0), -1)
            cv2.rectangle(frame, (text_x - padding, text_y - 40), 
                         (text_x + text_size[0] + padding, text_y + 20), (0, 255, 0), 3)

            # Victory text
            cv2.putText(frame, victory_text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4, cv2.LINE_AA)

    def _draw_cutscene_overlay(self, frame):
        """Draw a full-screen overlay for in-app cutscenes (3s)."""
        h, w = frame.shape[:2]
        elapsed = time.time() - self.cutscene_overlay['start_time']
        alpha = 0.95

        # Darken background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # If integrated video is available, draw its current frame
        cap = self.cutscene_overlay.get('video_cap')
        if cap is not None:
            try:
                ret, vframe = cap.read()
                if not ret:
                    # Video finished
                    cap.release()
                    self.cutscene_overlay['video_cap'] = None
                    # stop audio proc if any
                    ap = self.cutscene_overlay.get('audio_proc')
                    if ap:
                        try:
                            ap.terminate()
                        except Exception:
                            pass
                        self.cutscene_overlay['audio_proc'] = None
                    # mark overlay finished
                    self.cutscene_overlay['active'] = False
                    self.cutscene_playing = False
                    self._advance_to_next_boss()
                else:
                    # Fit video frame into central rectangle (letterbox)
                    ph = int(h * 0.8)
                    pw = int(ph * (vframe.shape[1] / max(1, vframe.shape[0])))
                    if pw > w * 0.95:
                        pw = int(w * 0.95)
                        ph = int(pw * (vframe.shape[0] / max(1, vframe.shape[1])))
                    img = cv2.resize(vframe, (pw, ph))
                    x = (w - pw) // 2
                    y = (h - ph) // 2 - 10
                    frame[y:y+ph, x:x+pw] = img
            except Exception:
                # Something went wrong with playback: fallback to portrait/card and finish
                try:
                    cap.release()
                except Exception:
                    pass
                self.cutscene_overlay['video_cap'] = None
                self.cutscene_overlay['active'] = False
                self.cutscene_playing = False
                self._advance_to_next_boss()
            return

        # Draw portrait if available
        portrait = self.cutscene_overlay.get('portrait')
        if portrait and os.path.isfile(portrait):
            try:
                img = cv2.imread(portrait)
                if img is not None:
                    # Fit portrait into central box
                    ph = int(h * 0.6)
                    pw = int(ph * (img.shape[1] / max(1, img.shape[0])))
                    img = cv2.resize(img, (pw, ph))
                    x = (w - pw) // 2
                    y = (h - ph) // 2 - 30
                    frame[y:y+ph, x:x+pw] = img
            except Exception:
                pass
        else:
            # Stylized boss card
            title = self.cutscene_overlay.get('boss_name', 'BOSS')
            title_text = f"K.O. {title}!"
            text_size = cv2.getTextSize(title_text, cv2.FONT_HERSHEY_SIMPLEX, 2.2, 4)[0]
            tx = (w - text_size[0]) // 2
            ty = h // 2
            cv2.putText(frame, title_text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 220, 0), 4, cv2.LINE_AA)

        # Small timer/progress bar
        bar_w = int(w * 0.6)
        bar_h = 14
        bx = (w - bar_w) // 2
        by = int(h * 0.85)
        cv2.rectangle(frame, (bx, by), (bx + bar_w, by + bar_h), (80, 80, 80), -1)
        pct = min(1.0, max(0.0, elapsed / self.cutscene_overlay['duration']))
        cv2.rectangle(frame, (bx, by), (bx + int(bar_w * (1 - pct)), by + bar_h), (0, 180, 0), -1)
    
    def _render_fullscreen_cutscene(self, frame: np.ndarray) -> np.ndarray:
        """Render cutscene in fullscreen mode with no interruptions"""
        h, w = frame.shape[:2]
        
        # Create a black fullscreen canvas
        cutscene_frame = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Check if cutscene should end
        elapsed = time.time() - self.cutscene_overlay['start_time']
        if elapsed >= self.cutscene_overlay['duration']:
            print(f"🎬 Cutscene finished after {elapsed:.1f}s, advancing to next boss")
            self.cutscene_overlay['active'] = False
            self.cutscene_playing = False
            self._advance_to_next_boss()
            return cutscene_frame
        
        # Try to load and display video
        cutscene_path = self.cutscene_overlay.get('video_path')
        
        if cutscene_path and os.path.exists(cutscene_path):
            try:
                # Initialize video capture ONCE at the start of cutscene
                if not hasattr(self, '_cutscene_cap') or self._cutscene_cap is None:
                    print(f"🎬 Initializing video capture for: {cutscene_path}")
                    self._cutscene_cap = cv2.VideoCapture(cutscene_path)
                    
                    if not self._cutscene_cap.isOpened():
                        print(f"❌ Failed to open video: {cutscene_path}")
                        self._cutscene_cap = None
                        # Fall through to text display
                    else:
                        print(f"✅ Video capture initialized successfully")
                
                # Read next frame sequentially (no seeking)
                if self._cutscene_cap and self._cutscene_cap.isOpened():
                    ret, video_frame = self._cutscene_cap.read()
                    
                    if ret and video_frame is not None:
                        # Resize video to fill screen while maintaining aspect ratio
                        video_h, video_w = video_frame.shape[:2]
                        
                        # Calculate scaling to fit screen
                        scale_w = w / video_w
                        scale_h = h / video_h
                        scale = min(scale_w, scale_h)
                        
                        new_w = int(video_w * scale)
                        new_h = int(video_h * scale)
                        
                        # Resize video frame
                        resized_video = cv2.resize(video_frame, (new_w, new_h))
                        
                        # Center the video on the screen
                        start_x = (w - new_w) // 2
                        start_y = (h - new_h) // 2
                        end_x = start_x + new_w
                        end_y = start_y + new_h
                        
                        # Place video on the cutscene frame
                        cutscene_frame[start_y:end_y, start_x:end_x] = resized_video
                        
                        # Add progress bar at bottom
                        progress = elapsed / self.cutscene_overlay['duration']
                        bar_width = int(w * 0.8 * progress)
                        bar_y = h - 30
                        cv2.rectangle(cutscene_frame, (w//10, bar_y), 
                                     (w//10 + int(w * 0.8), bar_y + 10), (60, 60, 60), -1)
                        cv2.rectangle(cutscene_frame, (w//10, bar_y), 
                                     (w//10 + bar_width, bar_y + 10), (0, 255, 0), -1)
                        
                        return cutscene_frame
                        
                    else:
                        # Video ended or failed to read, show text for remaining time
                        if self._cutscene_cap:
                            self._cutscene_cap.release()
                            self._cutscene_cap = None
                        # Fall through to text display
                        
            except Exception as e:
                print(f"❌ Video playback error: {e}")
                if hasattr(self, '_cutscene_cap') and self._cutscene_cap:
                    self._cutscene_cap.release()
                    self._cutscene_cap = None
                # Fall through to text display
        
        # Text fallback display
        boss_name = self.cutscene_overlay.get('boss_name', 'BOSS')
        text = f"{boss_name} Defeated!"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)[0]
        text_x = (w - text_size[0]) // 2
        text_y = h // 2
        cv2.putText(cutscene_frame, text, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        
        # Add some visual flair with animated background
        progress = elapsed / self.cutscene_overlay['duration']
        
        # Animated circles
        for i in range(5):
            radius = int(50 + i * 30 + progress * 100)
            alpha = max(0, 1.0 - progress - i * 0.1)
            color_intensity = int(255 * alpha * 0.3)
            if color_intensity > 0:
                cv2.circle(cutscene_frame, (w//2, h//2), radius, 
                          (0, color_intensity, 0), 2)
        
        # Progress bar
        bar_width = int(w * 0.8 * progress)
        bar_y = h//2 + 100
        cv2.rectangle(cutscene_frame, (w//10, bar_y), 
                     (w//10 + int(w * 0.8), bar_y + 20), (60, 60, 60), -1)
        cv2.rectangle(cutscene_frame, (w//10, bar_y), 
                     (w//10 + bar_width, bar_y + 20), (0, 255, 0), -1)
        
        return cutscene_frame

    def _update_game_state(self):
        """Update game state"""
        current_time = time.time()
        
        # Reset combo if timeout exceeded
        if (current_time - self.last_hit_time) > self.combo_timeout:
            self.combo_count = 0
        
        # Check for boss defeat (health reached 0) even if not from direct hit
        if (self.mannequin_health <= 0 and self.campaign_active and 
            not self.cutscene_playing and 
            not self.bosses[self.current_boss_index].defeated):
            
            # Mark boss defeated
            current_boss = self.bosses[self.current_boss_index]
            current_boss.defeated = True
            print(f"🎉 Boss defeated: {current_boss.name} (Health: {self.mannequin_health})")
            
            # Play cutscene (non-blocking start)
            try:
                self.cutscene_playing = True
                self._play_cutscene(current_boss)
                print(f"🎬 Starting cutscene for {current_boss.name}")
            except Exception as e:
                print(f"Cutscene playback error: {e}")
                # Immediately advance if cutscene failed
                print("⚠️ Cutscene failed, advancing to next boss")
                self._advance_to_next_boss()
        
        # Regeneration removed for campaign mode — mannequin does not auto-heal
        # (Previously: slow auto-heal after 5s with +1 HP). Kept intentionally off.

        # If cutscene finished, advance boss
        if self.cutscene_playing:
            # cutscene_playing is managed by _play_cutscene; when false, advance
            pass
    
    def _update_fps(self):
        """Update FPS calculation"""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def reset_game(self, reset_campaign: bool = False):
        """Reset game state. If reset_campaign True, reset campaign progress as well."""
        # Reset current boss state
        current_health = self.max_health
        if reset_campaign:
            # Reset campaign to first boss
            for b in self.bosses:
                b.defeated = False
            self.current_boss_index = 0
            if self.bosses:
                self.max_health = self.bosses[0].max_health
                current_health = self.max_health
            self.campaign_active = True
            print("🔁 Campaign Reset to Boss 1")
        
        # Set ESP32 health to reset value
        print(f"🔄 Resetting ESP32 health to: {current_health}")
        if self._set_esp32_health(current_health):
            self.mannequin_health = current_health
            print(f"✅ ESP32 health reset successful")
        else:
            print("⚠️ Failed to reset ESP32 health, using local value")
            self.mannequin_health = current_health
        
        self.score = 0
        self.combo_count = 0
        self.hit_effects.clear()
        # Reset zone positioning for fresh start
        self.zones_need_repositioning = True
        self.base_zones = []
        print("🔄 Game Reset!")

    def initialize_esp32_health(self):
        """Initialize ESP32 health to starting boss health"""
        if self.bosses:
            initial_health = self.bosses[0].max_health
            print(f"🚀 Initializing game - Setting ESP32 health to {initial_health}")
            
            # Try up to 3 times to set initial health
            for attempt in range(3):
                success = self._set_esp32_health(initial_health)
                if success:
                    print("✅ ESP32 health initialization successful")
                    with self.esp32_lock:
                        self.mannequin_health = initial_health
                        self.previous_health = initial_health  # Update previous health
                    return True
                else:
                    print(f"⚠️ ESP32 health initialization attempt {attempt + 1} failed")
                    if attempt < 2:  # Don't sleep on last attempt
                        time.sleep(1.0)
            
            print("❌ ESP32 health initialization failed after 3 attempts")
            # Set local health anyway
            with self.esp32_lock:
                self.mannequin_health = initial_health
                self.previous_health = initial_health  # Update previous health
            return False
        return False
        
    # ---------------- Campaign helpers ----------------
    def _init_campaign(self):
        """Initialize campaign bosses. Try loading 'campaign.json' in workspace; otherwise use built-in list."""
        cfg_file = os.path.join(os.path.dirname(__file__), 'campaign.json')

        loaded = False
        if os.path.isfile(cfg_file):
            try:
                import json
                with open(cfg_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Validate and create bosses
                for i, entry in enumerate(data):
                    name = entry.get('name')
                    max_hp = int(entry.get('max_health', 100))
                    portrait = entry.get('portrait') if 'portrait' in entry else None
                    cutscene = entry.get('cutscene') if 'cutscene' in entry else f"cutscenes/boss{i+1}.mp4"

                    if not name:
                        continue

                    boss = Boss(name=name, max_health=max_hp, portrait=portrait, cutscene=cutscene, defeated=False)
                    self.bosses.append(boss)

                if self.bosses:
                    loaded = True
            except Exception as e:
                print(f"Failed to load campaign.json: {e}")

        if not loaded:
            # Fallback built-in list with correct video file paths
            custom_bosses = [
                ("COWBOY BOB", "cutscenes/Money_Bob.mp4"),
                ("SIR ROYAL BOB", "cutscenes/Sir_Royal_Bob.mp4"), 
                ("EVIL BOBBY", "cutscenes/Evil_Bobby.mp4"),
                ("FANCY BOB", "cutscenes/Fancy_Bob.mp4"),
                ("RICH BOB", "cutscenes/Money_Bob.mp4"),  # Reuse Money_Bob for Rich Bob
            ]

            for i, (name, video_file) in enumerate(custom_bosses):
                boss = Boss(
                    name=name,
                    max_health=100 + i * 50,
                    portrait=None,
                    cutscene=video_file,
                    defeated=False
                )
                self.bosses.append(boss)

        # Set current boss health
        if self.bosses:
            self.current_boss_index = 0
            self.max_health = self.bosses[0].max_health
            self.mannequin_health = self.max_health
            # Don't set ESP32 health here - it will be done by initialize_esp32_health()

    def _play_cutscene(self, boss: Boss):
        """In-app cutscene overlay: play the video's frames inside the main window.
        If ffplay is available, attempt to play audio in background (no external video window).
        """
        print(f"🎬 Playing cutscene for {boss.name}")
        file = boss.cutscene
        portrait = boss.portrait if boss.portrait and os.path.isfile(boss.portrait) else None

        # Clean up any previous video capture
        if hasattr(self, '_cutscene_cap') and self._cutscene_cap is not None:
            self._cutscene_cap.release()
            self._cutscene_cap = None

        # Initialize overlay state
        self.cutscene_overlay['active'] = True
        self.cutscene_overlay['boss_name'] = boss.name
        self.cutscene_overlay['start_time'] = time.time()
        self.cutscene_overlay['portrait'] = portrait
        self.cutscene_overlay['video_path'] = file
        self.cutscene_overlay['audio_proc'] = None
        self.cutscene_overlay['duration'] = 4.0  # Default duration with buffer

        # Try opening video file for integrated playback
        if file and os.path.isfile(file):
            print(f"📁 Found cutscene file: {file}")
            try:
                # Set video path for fullscreen renderer
                self.cutscene_overlay['video_path'] = file
                print(f"🎬 Set video path: {file}")
                
                # Get video duration for proper timing
                temp_cap = cv2.VideoCapture(file)
                if temp_cap.isOpened():
                    try:
                        frames = int(temp_cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                        fps = temp_cap.get(cv2.CAP_PROP_FPS) or 30.0
                        if frames > 0 and fps > 0:
                            video_duration = frames / fps
                            # Add 1 second buffer for smooth transitions
                            self.cutscene_overlay['duration'] = max(4.0, video_duration + 1.0)
                            print(f"🎞️ Video duration: {video_duration:.1f}s (with buffer: {self.cutscene_overlay['duration']:.1f}s)")
                    except Exception as e:
                        print(f"Error getting video duration: {e}")
                    temp_cap.release()

                    # Try to play audio in background using ffplay (no display)
                    try:
                        proc = subprocess.Popen(["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", file], 
                                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        self.cutscene_overlay['audio_proc'] = proc
                        print("🔊 Audio playback started")
                    except Exception:
                        # audio not available; ignore
                        self.cutscene_overlay['audio_proc'] = None
                        print("🔇 No audio playback available")
                else:
                    print("⚠️ Failed to open video file for inspection")
            except Exception as e:
                print(f"⚠️ Video processing error: {e}")
                self.cutscene_overlay['video_path'] = None
        else:
            print(f"📁 No cutscene file found at: {file}")
            self.cutscene_overlay['video_path'] = None

        # If no video and no portrait, use fallback duration
        if not self.cutscene_overlay['video_path'] and not portrait:
            print("🎭 Using default 4s boss card cutscene")
            self.cutscene_overlay['duration'] = 4.0

        self.cutscene_playing = True
        print(f"🎬 Cutscene started, duration: {self.cutscene_overlay['duration']:.1f}s")

    def _advance_to_next_boss(self):
        """Move campaign to the next boss or finish campaign"""
        # Clean up cutscene resources
        self._cleanup_cutscene()
        
        # Advance index
        self.current_boss_index += 1
        if self.current_boss_index >= len(self.bosses):
            # Campaign complete
            print("🏁 Campaign complete! All bosses defeated.")
            self.campaign_active = False
            # Show final victory state
            self.mannequin_health = 0
            return

        # Setup next boss
        next_boss = self.bosses[self.current_boss_index]
        self.max_health = next_boss.max_health
        
        # Reset zone positioning for new boss
        self.zones_need_repositioning = True
        self.base_zones = []
        
        # Set ESP32 health to new boss health and wait for confirmation
        print(f"🎯 Setting ESP32 health for next boss: {next_boss.max_health}")
        if self._set_esp32_health(next_boss.max_health):
            self.mannequin_health = next_boss.max_health
            self.previous_health = next_boss.max_health  # Update previous health
            print(f"✅ ESP32 health set to {next_boss.max_health} for {next_boss.name}")
        else:
            print("⚠️ Failed to set ESP32 health, using local value")
            self.mannequin_health = next_boss.max_health
            self.previous_health = next_boss.max_health  # Update previous health
        
        self.hit_effects.clear()
        self.combo_count = 0
        self.last_hit_time = 0
        print(f"➡️ Next boss: {next_boss.name} (HP: {next_boss.max_health})")

    def _start_esp32_monitoring(self):
        """Start background thread to monitor ESP32 health"""
        self.esp32_running = True
        self.esp32_thread = threading.Thread(target=self._esp32_health_monitor, daemon=True)
        self.esp32_thread.start()
        print("🔗 Started ESP32 health monitoring thread")
    
    def _esp32_health_monitor(self):
        """Background thread to continuously fetch health from ESP32"""
        while self.esp32_running:
            try:
                current_time = time.time()
                if current_time - self.esp32_last_fetch >= self.esp32_fetch_interval:
                    health = self._fetch_esp32_health()
                    if health is not None:
                        with self.esp32_lock:
                            # Check if health has changed to trigger zone repositioning
                            if health != self.previous_health:
                                print(f"🔄 Health changed from {self.previous_health} to {health} - repositioning hit zones")
                                self.zones_need_repositioning = True
                                self.previous_health = health
                            
                            self.esp32_health = health
                            self.mannequin_health = health  # Sync local cache
                            if not self.esp32_connected:
                                self.esp32_connected = True
                                print("✅ ESP32 connected successfully")
                    else:
                        if self.esp32_connected:
                            self.esp32_connected = False
                            print("❌ ESP32 connection lost")
                    
                    self.esp32_last_fetch = current_time
                
                time.sleep(0.05)  # Small sleep to prevent CPU overload
            except Exception as e:
                print(f"ESP32 monitoring error: {e}")
                time.sleep(1.0)  # Longer sleep on error
    
    def _fetch_esp32_health(self):
        """Fetch current health from ESP32"""
        try:
            response = requests.get(f"http://{self.esp32_ip}/status", timeout=0.5)
            if response.status_code == 200:
                data = response.json()
                # New API format: {"currentNetForce":3.060,"maxForceScore":4.446,"bossHealth":96.583,"bossMaxHealth":100.000}
                boss_health = data.get('bossHealth', None)
                if boss_health is not None:
                    return int(boss_health)  # Convert to integer for game display
                # Fallback to old format
                return data.get('health', None)
        except Exception as e:
            # Silently handle connection errors for low latency
            pass
        return None
    
    def _set_esp32_health(self, health_value):
        """Set health value on ESP32"""
        try:
            # Use POST method for setting health
            response = requests.post(f"http://{self.esp32_ip}/set_health?value={health_value}", timeout=1.0)
            if response.status_code == 200:
                print(f"🎯 ESP32 health set to: {health_value}")
                return True
            else:
                print(f"❌ ESP32 set health failed: HTTP {response.status_code}")
                return False
        except Exception as e:
            print(f"Failed to set ESP32 health: {e}")
        return False
    
    def stop_esp32_monitoring(self):
        """Stop ESP32 monitoring thread and cleanup resources"""
        self.esp32_running = False
        if self.esp32_thread and self.esp32_thread.is_alive():
            self.esp32_thread.join(timeout=2.0)
        print("🔗 ESP32 monitoring stopped")
        
        # Stop background music
        self.stop_background_music()
        
        # Cleanup pygame
        if self.music_initialized:
            try:
                pygame.mixer.quit()
                print("🎵 Music system shutdown")
            except Exception as e:
                print(f"⚠️ Error shutting down music system: {e}")
        
    def _cleanup_cutscene(self):
        """Clean up cutscene resources"""
        if hasattr(self, '_cutscene_cap') and self._cutscene_cap is not None:
            self._cutscene_cap.release()
            self._cutscene_cap = None
        
        # Clean up audio process if running
        if self.cutscene_overlay.get('audio_proc'):
            try:
                self.cutscene_overlay['audio_proc'].terminate()
            except:
                pass
            self.cutscene_overlay['audio_proc'] = None

    def _init_background_music(self):
        """Initialize pygame mixer and load background music"""
        try:
            # Initialize pygame mixer for audio
            pygame.mixer.pre_init(frequency=22050, size=-16, channels=2, buffer=512)
            pygame.mixer.init()
            self.music_initialized = True
            print("🎵 Music system initialized")
            
            # Check if music file exists
            if os.path.exists(self.music_file):
                print(f"🎵 Found background music: {os.path.basename(self.music_file)}")
            else:
                print(f"⚠️ Background music file not found: {self.music_file}")
                self.music_initialized = False
                
        except Exception as e:
            print(f"⚠️ Failed to initialize music system: {e}")
            self.music_initialized = False
    
    def start_background_music(self):
        """Start playing background music in a loop"""
        if not self.music_initialized:
            print("⚠️ Music system not initialized")
            return
            
        try:
            if os.path.exists(self.music_file):
                pygame.mixer.music.load(self.music_file)
                pygame.mixer.music.set_volume(self.music_volume)
                pygame.mixer.music.play(-1)  # -1 means loop indefinitely
                self.music_playing = True
                print(f"🎵 Started background music (Volume: {int(self.music_volume * 100)}%)")
            else:
                print(f"⚠️ Cannot start music: file not found: {self.music_file}")
        except Exception as e:
            print(f"⚠️ Failed to start background music: {e}")
    
    def stop_background_music(self):
        """Stop background music"""
        if self.music_initialized and self.music_playing:
            try:
                pygame.mixer.music.stop()
                self.music_playing = False
                print("🎵 Stopped background music")
            except Exception as e:
                print(f"⚠️ Failed to stop background music: {e}")
    
    def set_music_volume(self, volume: float):
        """Set background music volume (0.0 to 1.0)"""
        if self.music_initialized and self.music_playing:
            try:
                self.music_volume = max(0.0, min(1.0, volume))
                pygame.mixer.music.set_volume(self.music_volume)
                print(f"🔊 Music volume set to {int(self.music_volume * 100)}%")
            except Exception as e:
                print(f"⚠️ Failed to set music volume: {e}")

def create_boxing_pipeline():
    """Create simplified DepthAI pipeline for boxing game"""
    pipeline = dai.Pipeline()
    
    # Create camera with minimal settings
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    xout = pipeline.create(dai.node.XLinkOut)
    
    xout.setStreamName("rgb")
    
    # Very conservative camera settings
    # Set FOV to 78 degrees (THE_1080_P)
    cam_rgb.setPreviewSize(1920, 1080)
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setInterleaved(False)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam_rgb.setFps(15)
    # Flip the camera image vertically (rotate 180 degrees)
    cam_rgb.setImageOrientation(dai.CameraImageOrientation.ROTATE_180_DEG)

    cam_rgb.preview.link(xout.input)

    return pipeline


def main():
    """Main boxing game function"""
    print("🥊 OAK BOXING TRAINER 🥊")
    print("=" * 40)
    print("First-person boxing game!")
    print("Position a person/mannequin in view")
    print("Punch the highlighted zones to score!")
    print()
    
    # Check for OAK device
    devices = dai.Device.getAllAvailableDevices()
    if not devices:
        print("❌ No OAK devices found!")
        return False

    print(f"✅ Found OAK device: {devices[0].name}")

    # Initialize game
    boxing_game = BoxingGameDetector()
    
    # Start background music
    boxing_game.start_background_music()
    
    # Wait a moment for ESP32 initialization to complete
    print("🔄 Initializing ESP32 connection...")
    time.sleep(1.5)

    try:
        pipeline = create_boxing_pipeline()

        print("🔌 Connecting to OAK camera...")
        device = dai.Device(pipeline, devices[0])

        print("✅ Connected successfully!")
        print(f"🚀 USB: {device.getUsbSpeed().name}")

        # Get output queue with very conservative settings
        q = device.getOutputQueue(name="rgb", maxSize=1, blocking=False)  # Minimal queue
        print("📡 Camera ready")

        print("\n🥊 BOXING GAME CONTROLS:")
        print("  'q' = quit")
        print("  'r' = reset current boss")
        print("  'c' = reset entire campaign")
        print("  'f' = toggle fullscreen")
        print("  '+' = increase music volume")
        print("  '-' = decrease music volume")
        print("  'm' = toggle music on/off")
        print("🎯 Position mannequin in view and start punching!\n")

        window_name = "🥊 OAK Boxing Trainer"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1920, 1080)  # Match 78 degree FOV resolution

        fullscreen_mode = False
        frame_count = 0
        last_frame_time = time.time()
        consecutive_failures = 0
        
        while True:
            try:
                # Get frame with timeout handling
                inRgb = q.tryGet()
                
                if inRgb is not None:
                    consecutive_failures = 0
                    
                    # Process frame
                    oak_frame = inRgb.getCvFrame()

                    if oak_frame is not None and oak_frame.size > 0:
                        # Resize frame to consistent size for processing
                        if oak_frame.shape != (1080, 1920, 3):
                            oak_frame = cv2.resize(oak_frame, (1920, 1080))

                        # Rely on camera hardware flip (we configured the camera to rotate 180deg)
                        # Removed software flip to avoid double-flipping.

                        # Run boxing game detection
                        game_frame = boxing_game.detect_and_process(oak_frame)

                        # Display
                        cv2.imshow(window_name, game_frame)

                        if fullscreen_mode:
                            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

                        frame_count += 1
                        last_frame_time = time.time()
                    else:
                        print("⚠️ Received empty frame")
                        consecutive_failures += 1
                else:
                    consecutive_failures += 1
                    if consecutive_failures > 30:  # 30 consecutive failures
                        print("⚠️ Too many consecutive frame failures, restarting...")
                        break
                
                # Check for overall timeout
                if time.time() - last_frame_time > 10.0:
                    print("⚠️ No valid frames for 10 seconds, stopping...")
                    break
                
                # Handle input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("👋 Thanks for playing!")
                    break
                elif key == ord('r'):
                    boxing_game.reset_game()
                elif key == ord('c'):
                    boxing_game.reset_game(reset_campaign=True)
                elif key == ord('f'):
                    fullscreen_mode = not fullscreen_mode
                    if fullscreen_mode:
                        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                        print("🖥️ Fullscreen: ON")
                    else:
                        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                        print("🖥️ Fullscreen: OFF")
                elif key == ord('+') or key == ord('='):  # Handle both + and = keys
                    new_volume = min(1.0, boxing_game.music_volume + 0.1)
                    boxing_game.set_music_volume(new_volume)
                elif key == ord('-'):
                    new_volume = max(0.0, boxing_game.music_volume - 0.1)
                    boxing_game.set_music_volume(new_volume)
                elif key == ord('m'):
                    if boxing_game.music_playing:
                        boxing_game.stop_background_music()
                    else:
                        boxing_game.start_background_music()
                elif key == 27:  # ESC
                    if fullscreen_mode:
                        fullscreen_mode = False
                        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                        print("🖥️ Fullscreen: OFF")
                
            except Exception as e:
                print(f"⚠️ Frame processing error: {e}")
                continue
        
        device.close()
        boxing_game.stop_esp32_monitoring()
        return True
        
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False
    
    finally:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    print("🥊 OAK-1 AF Boxing Trainer")
    print("🎯 First-person boxing game with pose detection")
    print("👊 Punch the zones to score points!")
    print()
    
    success = main()
    
    if success:
        print("\n✅ Boxing session completed!")
    else:
        print("\n❌ Failed to start boxing trainer")
        print("💡 Try: Different USB port, restart computer, check connections")