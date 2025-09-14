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

@dataclass
class HitZone:
    """Represents a target zone on the mannequin"""
    name: str
    center: Tuple[int, int]
    radius: int
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
            'audio_proc': None
        }
        
    def detect_and_process(self, frame: np.ndarray) -> np.ndarray:
        """Main detection and game logic - with frame validation"""
        # Validate input frame
        if frame is None or frame.size == 0:
            print("⚠️ Invalid frame received")
            return np.zeros((360, 480, 3), dtype=np.uint8)  # Return black frame
        
        # Create game frame first
        game_frame = frame.copy()
        
        try:
            # Only process every 3rd frame to reduce load
            self.fps_counter += 1
            if self.fps_counter % 3 != 0:
                # Skip heavy processing, just draw cached elements
                self._draw_hit_zones(game_frame)
                self._draw_hit_effects(game_frame)
                self._draw_hud(game_frame)
                # Always draw overlay if active
                if self.cutscene_overlay['active']:
                    self._draw_cutscene_overlay(game_frame)
                    # Check overlay expiry
                    if time.time() - self.cutscene_overlay['start_time'] >= self.cutscene_overlay['duration']:
                        self.cutscene_overlay['active'] = False
                        self.cutscene_playing = False
                        self._advance_to_next_boss()
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
            # Draw cutscene overlay if active
            if self.cutscene_overlay['active']:
                self._draw_cutscene_overlay(game_frame)
                if time.time() - self.cutscene_overlay['start_time'] >= self.cutscene_overlay['duration']:
                    self.cutscene_overlay['active'] = False
                    self.cutscene_playing = False
                    self._advance_to_next_boss()
        except Exception as e:
            print(f"Drawing error: {e}")
        
        # Update game state
        self._update_game_state()
        self._update_fps()
        
        return game_frame
    
    def _setup_hit_zones(self, frame, landmarks):
        """Setup hit zones based on detected mannequin pose"""
        h, w = frame.shape[:2]

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

        # Get key landmarks with smoothing
        nose = landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
        left_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]

        head_x, head_y = smooth_landmark('NOSE', nose)
        ls_x, ls_y = smooth_landmark('L_SHOULDER', left_shoulder)
        rs_x, rs_y = smooth_landmark('R_SHOULDER', right_shoulder)

        # Estimate torso width/size to scale zones and thresholds
        torso_width = max(20, int(abs(ls_x - rs_x)))
        self.torso_size = int(torso_width * 1.6)

        # Chest position slightly below shoulder midpoint
        chest_x = int((ls_x + rs_x) / 2)
        chest_y = int((ls_y + rs_y) / 2 + self.torso_size * 0.12)

        # Clear existing zones and create scaled ones
        self.hit_zones.clear()
        current_time = time.time()

        # Enlarge zones slightly to make hits easier
        head_radius = max(36, int(self.torso_size * 0.32))
        chest_radius = max(60, int(self.torso_size * 0.48))

        self.hit_zones.append(HitZone(
            name="HEAD",
            center=(head_x, head_y),
            radius=head_radius,
            color=(0, 0, 255),
            hit_damage=20,
            last_hit_time=getattr(self, 'head_last_hit', 0)
        ))

        self.hit_zones.append(HitZone(
            name="CHEST",
            center=(chest_x, chest_y),
            radius=chest_radius,
            color=(0, 255, 255),
            hit_damage=15,
            last_hit_time=getattr(self, 'chest_last_hit', 0)
        ))
    
    def _draw_mannequin(self, frame, landmarks):
        """Draw mannequin pose"""
        # Draw pose landmarks in a subtle way
        self.mp_drawing.draw_landmarks(
            frame,
            landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                color=(100, 100, 100), thickness=1, circle_radius=2),
            connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                color=(150, 150, 150), thickness=2)
        )
    
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

            # Draw hand tracking (smoothed)
            cv2.circle(frame, (sx, sy), 15, (0, 255, 0), 3)

    
    def _check_hits(self):
        """Check if hands hit any target zones"""
        current_time = time.time()
        for i, hand_pos in enumerate(self.hand_positions):
            # Use smoothed velocity if available
            dx, dy = (0, 0)
            if i < len(self.hand_velocities):
                dx, dy = self.hand_velocities[i]

            hand_speed = math.sqrt(dx * dx + dy * dy)

            # Scale threshold by torso size so users at different distances work
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
                margin = max(8, int(zone.radius * 0.12))
                if dist > (zone.radius + margin):
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

        # Update game state
        self.mannequin_health = max(0, self.mannequin_health - damage)
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
        """Draw target zones"""
        current_time = time.time()
        
        for zone in self.hit_zones:
            if not zone.active:
                continue
            
            # Calculate zone appearance based on cooldown
            time_since_hit = current_time - zone.last_hit_time
            alpha = min(time_since_hit / zone.hit_cooldown, 1.0)
            
            # Zone color (darker when on cooldown)
            color = tuple(int(c * alpha) for c in zone.color)
            
            # Draw zone circle
            cv2.circle(frame, zone.center, zone.radius, color, 3)
            
            # Draw zone label
            label_pos = (zone.center[0] - 30, zone.center[1] - zone.radius - 10)
            cv2.putText(frame, zone.name, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, color, 2)
            
            # Draw crosshair in center
            cross_size = 10
            cv2.line(frame, 
                    (zone.center[0] - cross_size, zone.center[1]),
                    (zone.center[0] + cross_size, zone.center[1]), 
                    color, 2)
            cv2.line(frame, 
                    (zone.center[0], zone.center[1] - cross_size),
                    (zone.center[0], zone.center[1] + cross_size), 
                    color, 2)
    
    def _draw_hit_effects(self, frame):
        """Draw hit effect animations"""
        current_time = time.time()
        effects_to_remove = []
        
        for i, effect in enumerate(self.hit_effects):
            time_elapsed = current_time - effect['time']
            if time_elapsed > effect['duration']:
                effects_to_remove.append(i)
                continue
            
            # Animation progress (0 to 1)
            progress = time_elapsed / effect['duration']
            
            # Growing circle effect
            radius = int(20 + progress * 30)
            alpha = 1.0 - progress
            
            # Color based on zone
            if effect['zone'] == 'HEAD':
                color = (0, 0, int(255 * alpha))
            else:
                color = (0, int(255 * alpha), int(255 * alpha))
            
            # Draw effect
            cv2.circle(frame, effect['pos'], radius, color, 3)
            
            # Draw damage text
            text_pos = (effect['pos'][0] - 20, effect['pos'][1] - radius - 10)
            cv2.putText(frame, f"-{effect['damage']}", text_pos, 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Remove expired effects
        for i in reversed(effects_to_remove):
            del self.hit_effects[i]
    
    def _draw_hud(self, frame):
        """Draw heads-up display"""
        h, w = frame.shape[:2]
        
        # Health bar
        bar_width = 300
        bar_height = 30
        bar_x = w - bar_width - 20
        bar_y = 20
        
        # Health bar background
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                     (50, 50, 50), -1)
        
        # Health bar fill
        health_width = int((self.mannequin_health / self.max_health) * bar_width)
        health_color = (0, 255, 0) if self.mannequin_health > 50 else (0, 165, 255) if self.mannequin_health > 25 else (0, 0, 255)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + health_width, bar_y + bar_height), 
                     health_color, -1)
        
        # Health text
        health_text = f"MANNEQUIN HP: {self.mannequin_health}/{self.max_health}"
        cv2.putText(frame, health_text, (bar_x, bar_y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Score and combo
        cv2.putText(frame, f"SCORE: {self.score}", (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        if self.combo_count > 1:
            combo_color = (0, 255, 255) if self.combo_count < 5 else (255, 0, 255)
            cv2.putText(frame, f"COMBO x{self.combo_count}!", (20, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, combo_color, 2)
        
        # Instructions
        instructions = [
            "🥊 BOXING TRAINER",
            "Punch the highlighted zones!",
            "HEAD: 20 dmg | CHEST: 15 dmg",
            f"FPS: {self.current_fps:.1f}"
        ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(frame, instruction, (20, h - 100 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Victory message
        # Show knockout / boss defeated and campaign progression
        if self.mannequin_health <= 0:
            if self.campaign_active:
                boss = self.bosses[self.current_boss_index]
                msg = f"KNOCKOUT! {boss.name} defeated"
            else:
                msg = "KNOCKOUT!"

            victory_text = f"🏆 {msg} 🏆"
            text_size = cv2.getTextSize(victory_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
            text_x = (w - text_size[0]) // 2
            text_y = h // 2

            # Background
            cv2.rectangle(frame, (text_x - 20, text_y - 30), 
                         (text_x + text_size[0] + 20, text_y + 10), (0, 0, 0), -1)

            # Text
            cv2.putText(frame, victory_text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        # Draw current boss info
        if self.campaign_active and self.bosses:
            boss = self.bosses[self.current_boss_index]
            boss_text = f"BOSS {self.current_boss_index+1}/ {len(self.bosses)}: {boss.name}"
            cv2.putText(frame, boss_text, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 255), 2)

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
    
    def _update_game_state(self):
        """Update game state"""
        current_time = time.time()
        
        # Reset combo if timeout exceeded
        if (current_time - self.last_hit_time) > self.combo_timeout:
            self.combo_count = 0
        
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
        self.mannequin_health = self.max_health
        self.score = 0
        self.combo_count = 0
        self.hit_effects.clear()
        print("🔄 Game Reset!")

        if reset_campaign:
            # Reset campaign to first boss
            for b in self.bosses:
                b.defeated = False
            self.current_boss_index = 0
            if self.bosses:
                self.max_health = self.bosses[0].max_health
                self.mannequin_health = self.max_health
            self.campaign_active = True
            print("🔁 Campaign Reset to Boss 1")

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
            # Fallback built-in list
            custom_names = [
                "COWBOY BOB",
                "SIR ROYAL BOB",
                "EVIL BOBBY",
                "FANCY BOB",
                "RICH BOB",
            ]

            for i, name in enumerate(custom_names):
                boss = Boss(
                    name=name,
                    max_health=100 + i * 50,
                    portrait=None,
                    cutscene=f"cutscenes/boss{i+1}.mp4",
                    defeated=False
                )
                self.bosses.append(boss)

        # Set current boss health
        if self.bosses:
            self.current_boss_index = 0
            self.max_health = self.bosses[0].max_health
            self.mannequin_health = self.max_health

    def _play_cutscene(self, boss: Boss):
        """In-app cutscene overlay: play the video's frames inside the main window.
        If ffplay is available, attempt to play audio in background (no external video window).
        """
        file = boss.cutscene
        portrait = boss.portrait if boss.portrait and os.path.isfile(boss.portrait) else None

        # Initialize overlay state
        self.cutscene_overlay['active'] = True
        self.cutscene_overlay['boss_name'] = boss.name
        self.cutscene_overlay['start_time'] = time.time()
        self.cutscene_overlay['portrait'] = portrait
        self.cutscene_overlay['video_cap'] = None
        self.cutscene_overlay['audio_proc'] = None

        # Try opening video file for integrated playback
        if file and os.path.isfile(file):
            try:
                cap = cv2.VideoCapture(file)
                if cap.isOpened():
                    self.cutscene_overlay['video_cap'] = cap
                    # set duration to video length if available
                    try:
                        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                        if frames > 0 and fps > 0:
                            self.cutscene_overlay['duration'] = max(3.0, frames / fps)
                    except Exception:
                        pass

                    # Try to play audio in background using ffplay (no display)
                    try:
                        proc = subprocess.Popen(["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", file], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        self.cutscene_overlay['audio_proc'] = proc
                    except Exception:
                        # audio not available; ignore
                        self.cutscene_overlay['audio_proc'] = None
                else:
                    cap.release()
            except Exception:
                self.cutscene_overlay['video_cap'] = None

        # If no video cap and no portrait, fallback to 3s card
        if not self.cutscene_overlay['video_cap'] and not portrait:
            self.cutscene_overlay['duration'] = 3.0

        self.cutscene_playing = True

    def _advance_to_next_boss(self):
        """Move campaign to the next boss or finish campaign"""
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
        self.mannequin_health = self.max_health
        self.hit_effects.clear()
        self.combo_count = 0
        self.last_hit_time = 0
        print(f"➡️ Next boss: {next_boss.name} (HP: {next_boss.max_health})")


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
                elif key == 27:  # ESC
                    if fullscreen_mode:
                        fullscreen_mode = False
                        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                        print("🖥️ Fullscreen: OFF")
                
            except Exception as e:
                print(f"⚠️ Frame processing error: {e}")
                continue
        
        device.close()
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