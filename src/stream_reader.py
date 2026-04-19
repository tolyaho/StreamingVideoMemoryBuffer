"""reads a video file and yields fixed-duration RawWindows at a target fps."""
from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from typing import Iterator, List

import numpy as np


@dataclass
class RawWindow:
    """raw video window before visual encoding."""

    window_id: str
    start_time: float
    end_time: float
    frames: List[np.ndarray]   # list of RGB frames, shape (H, W, 3)

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def representative_frame(self) -> np.ndarray | None:
        """middle frame of the window."""
        if not self.frames:
            return None
        return self.frames[len(self.frames) // 2].copy()


class StreamReader:
    """converts a video file into a lazy sequence of RawWindows.

    Args:
        fps: frames per second to sample from the video.
        window_duration: duration of each output window in seconds.
    """

    def __init__(
        self,
        fps: float = 1.0,
        window_duration: float = 5.0,
    ):
        self.fps = fps
        self.window_duration = window_duration
        self._frames_per_window = max(1, int(fps * window_duration))

    def read_windows(self, video_path: str) -> Iterator[RawWindow]:
        """lazily yield RawWindow objects from a video file."""
        import cv2

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_step = max(1, int(round(video_fps / self.fps)))

        frame_idx = 0
        window_frames: List[np.ndarray] = []
        window_start_time = 0.0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_step == 0:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    window_frames.append(rgb)

                frame_idx += 1
                current_time = frame_idx / video_fps

                if len(window_frames) >= self._frames_per_window:
                    yield self._make_window(window_frames, window_start_time, current_time)
                    window_frames = []
                    window_start_time = current_time

            if window_frames:
                end_time = frame_idx / video_fps
                yield self._make_window(window_frames, window_start_time, end_time)
        finally:
            cap.release()

    def _make_window(
        self,
        frames: List[np.ndarray],
        start_time: float,
        end_time: float,
    ) -> RawWindow:
        return RawWindow(
            window_id=uuid.uuid4().hex[:8],
            start_time=round(start_time, 2),
            end_time=round(end_time, 2),
            frames=frames,
        )

    @staticmethod
    def synthetic_stream(
        n_windows: int = 60,
        window_duration: float = 5.0,
        frame_size: tuple = (224, 224),
        n_scenes: int = 4,
        frames_per_window: int = 5,
        seed: int = 42,
    ) -> Iterator[RawWindow]:
        """generate a synthetic video stream for demos without a real video file."""
        rng = np.random.default_rng(seed)
        scene_colors = [
            [180, 60, 60],
            [60, 120, 180],
            [60, 160, 80],
            [200, 160, 60],
        ]

        windows_per_scene = max(1, n_windows // n_scenes)

        for i in range(n_windows):
            scene_idx = min(i // windows_per_scene, n_scenes - 1)
            base_color = np.array(scene_colors[scene_idx], dtype=np.float32)

            frames = []
            for _ in range(frames_per_window):
                noise = rng.integers(-30, 30, size=(frame_size[0], frame_size[1], 3))
                frame = np.clip(
                    base_color[None, None, :] + noise, 0, 255
                ).astype(np.uint8)
                frames.append(frame)

            start_t = i * window_duration
            end_t = start_t + window_duration

            yield RawWindow(
                window_id=uuid.uuid4().hex[:8],
                start_time=round(start_t, 2),
                end_time=round(end_t, 2),
                frames=frames,
            )
