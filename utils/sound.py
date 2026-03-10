"""
Sound Alert Module
==================
Handles audio feedback for safety alerts using Pygame.
This allows for non-blocking audio playback so the video feed doesn't freeze.
"""

import pygame
import os
import time

class SoundManager:
    """
    Manages audio playback for alerts.
    """
    
    def __init__(self) -> None:
        """
        Initialize the pygame mixer for audio.
        """
        self.initialized = False
        try:
            pygame.mixer.init()
            self.initialized = True
            print("✓ Audio system initialized")
        except Exception as e:
            print(f"✗ Failed to initialize audio: {e}")
            print("  (Ensure pygame is installed: pip install pygame)")

    def play_alarm(self, file_path: str, loop: bool = False) -> None:
        """
        Play a sound file.
        
        Args:
            file_path: Path to sound file (.wav or .mp3)
            loop: Whether to loop the sound (True for critical alarms)
        """
        if not self.initialized:
            return

        if not os.path.exists(file_path):
            # Only print warning once to avoid spamming console
            return

        # Check if audio is already playing to avoid restarting it every frame
        if pygame.mixer.music.get_busy():
            return

        try:
            pygame.mixer.music.load(file_path)
            # loops=-1 means infinite loop, loops=0 means play once
            loops = -1 if loop else 0
            pygame.mixer.music.play(loops=loops)
        except Exception as e:
            print(f"Error playing sound: {e}")

    def stop_alarm(self) -> None:
        """
        Stop any currently playing sound.
        """
        if self.initialized:
            if pygame.mixer.music.get_busy():
                pygame.mixer.music.stop()