"""SQLite sidecar (via peewee) for the memory buffer.
Lets me inspect long runs or resume after a crash."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import numpy as np
from peewee import (
    BlobField,
    BooleanField,
    CharField,
    CompositeKey,
    DateTimeField,
    FloatField,
    ForeignKeyField,
    IntegerField,
    Model,
    Proxy,
    SqliteDatabase,
    TextField,
)

from .data_structures import EpisodeEntry, EventEntry, WindowEntry

database_proxy = Proxy()


class BaseModel(Model):
    class Meta:
        database = database_proxy


class Window(BaseModel):
    entry_id = CharField(primary_key=True)
    start_time = FloatField(index=True)
    end_time = FloatField()
    tier = CharField(default="recent")
    summary_text = TextField(null=True)
    visual_embedding = BlobField()
    summary_embedding = BlobField(null=True)
    embedding_dim = IntegerField()
    frame_jpeg = BlobField(null=True)
    created_at = DateTimeField(default=datetime.utcnow)

    class Meta:
        table_name = "window"


class Episode(BaseModel):
    entry_id = CharField(primary_key=True)
    start_time = FloatField(index=True)
    end_time = FloatField()
    summary_text = TextField(null=True)
    visual_embedding = BlobField()
    summary_embedding = BlobField(null=True)
    embedding_dim = IntegerField()
    created_at = DateTimeField(default=datetime.utcnow)

    class Meta:
        table_name = "episode"


class EpisodeWindow(BaseModel):
    episode = ForeignKeyField(Episode, backref="members", on_delete="CASCADE")
    window = ForeignKeyField(Window, backref="episodes", on_delete="RESTRICT")
    position = IntegerField()
    is_representative = BooleanField(default=False)

    class Meta:
        table_name = "episode_window"
        primary_key = CompositeKey("episode", "window")


class Event(BaseModel):
    entry_id = CharField(primary_key=True)
    start_time = FloatField(index=True)
    end_time = FloatField()
    summary_text = TextField(null=True)
    visual_embedding = BlobField()
    summary_embedding = BlobField(null=True)
    embedding_dim = IntegerField()
    created_at = DateTimeField(default=datetime.utcnow)

    class Meta:
        table_name = "event"


class EventEpisode(BaseModel):
    event = ForeignKeyField(Event, backref="members", on_delete="CASCADE")
    episode = ForeignKeyField(Episode, backref="events", on_delete="RESTRICT")
    position = IntegerField()

    class Meta:
        table_name = "event_episode"
        primary_key = CompositeKey("event", "episode")


class EventRepWindow(BaseModel):
    event = ForeignKeyField(Event, backref="rep_windows", on_delete="CASCADE")
    window = ForeignKeyField(Window, backref="rep_for_events", on_delete="RESTRICT")
    position = IntegerField()

    class Meta:
        table_name = "event_rep_window"
        primary_key = CompositeKey("event", "window")


ALL_MODELS = (
    Window,
    Episode,
    EpisodeWindow,
    Event,
    EventEpisode,
    EventRepWindow,
)


def _emb_to_bytes(arr: Optional[np.ndarray]) -> Optional[bytes]:
    if arr is None:
        return None
    return np.ascontiguousarray(arr, dtype=np.float32).tobytes()


def emb_from_bytes(buf: Optional[bytes], dim: int) -> Optional[np.ndarray]:
    if buf is None:
        return None
    out = np.frombuffer(buf, dtype=np.float32)
    if out.size != dim:
        raise ValueError(f"embedding dim mismatch: expected {dim}, got {out.size}")
    return out


def _frame_to_jpeg(frame: Optional[np.ndarray], quality: int = 90) -> Optional[bytes]:
    if frame is None:
        return None
    try:
        import cv2
    except ImportError:
        return None
    # stored frames are RGB numpy arrays; cv2.imencode wants BGR
    bgr = frame[:, :, ::-1] if frame.ndim == 3 and frame.shape[2] == 3 else frame
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return None
    return buf.tobytes()


class MemoryStore:
    """facade the HierarchicalMemoryWriter writes through"""

    def __init__(
        self,
        path: Union[str, Path],
        store_frames: bool = True,
        jpeg_quality: int = 90,
    ):
        self.path = str(path)
        self.store_frames = store_frames
        self.jpeg_quality = int(jpeg_quality)

        self.db = SqliteDatabase(
            self.path,
            pragmas={
                "journal_mode": "wal",
                "foreign_keys": 1,
                "synchronous": "normal",
                "cache_size": -64 * 1024,
            },
        )
        database_proxy.initialize(self.db)
        self.db.connect(reuse_if_open=True)
        self.db.create_tables(ALL_MODELS)

    def close(self) -> None:
        if not self.db.is_closed():
            self.db.close()

    def __enter__(self) -> "MemoryStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def save_window(self, window: WindowEntry) -> None:
        frame_bytes = (
            _frame_to_jpeg(window.frame, self.jpeg_quality)
            if self.store_frames
            else None
        )
        data = {
            "entry_id": window.entry_id,
            "start_time": float(window.start_time),
            "end_time": float(window.end_time),
            "tier": window.tier,
            "summary_text": window.summary_text,
            "visual_embedding": _emb_to_bytes(window.visual_embedding),
            "summary_embedding": _emb_to_bytes(window.summary_embedding),
            "embedding_dim": int(window.visual_embedding.shape[-1]),
            "frame_jpeg": frame_bytes,
        }
        with self.db.atomic():
            (
                Window
                .insert(**data)
                .on_conflict(
                    conflict_target=[Window.entry_id],
                    preserve=[
                        Window.tier,
                        Window.summary_text,
                        Window.summary_embedding,
                        Window.frame_jpeg,
                    ],
                )
                .execute()
            )

    def save_episode(self, episode: EpisodeEntry) -> None:
        rep_ids = set(episode.representative_window_ids or [])
        with self.db.atomic():
            Episode.replace(
                entry_id=episode.entry_id,
                start_time=float(episode.start_time),
                end_time=float(episode.end_time),
                summary_text=episode.summary_text,
                visual_embedding=_emb_to_bytes(episode.visual_embedding),
                summary_embedding=_emb_to_bytes(episode.summary_embedding),
                embedding_dim=int(episode.visual_embedding.shape[-1]),
            ).execute()

            # wipe and rewrite members — cheap and keeps rep flags consistent
            EpisodeWindow.delete().where(EpisodeWindow.episode == episode.entry_id).execute()
            rows = [
                {
                    "episode": episode.entry_id,
                    "window": wid,
                    "position": pos,
                    "is_representative": wid in rep_ids,
                }
                for pos, wid in enumerate(episode.member_window_ids)
            ]
            if rows:
                EpisodeWindow.insert_many(rows).execute()

    def save_event(self, event: EventEntry) -> None:
        with self.db.atomic():
            Event.replace(
                entry_id=event.entry_id,
                start_time=float(event.start_time),
                end_time=float(event.end_time),
                summary_text=event.summary_text,
                visual_embedding=_emb_to_bytes(event.visual_embedding),
                summary_embedding=_emb_to_bytes(event.summary_embedding),
                embedding_dim=int(event.visual_embedding.shape[-1]),
            ).execute()

            EventEpisode.delete().where(EventEpisode.event == event.entry_id).execute()
            ep_rows = [
                {"event": event.entry_id, "episode": epid, "position": pos}
                for pos, epid in enumerate(event.member_episode_ids)
            ]
            if ep_rows:
                EventEpisode.insert_many(ep_rows).execute()

            EventRepWindow.delete().where(EventRepWindow.event == event.entry_id).execute()
            seen: set = set()
            rw_rows = []
            for pos, wid in enumerate(event.representative_window_ids or []):
                if wid in seen:
                    continue
                seen.add(wid)
                rw_rows.append({"event": event.entry_id, "window": wid, "position": pos})
            if rw_rows:
                EventRepWindow.insert_many(rw_rows).execute()

    def counts(self) -> dict:
        return {
            "windows": Window.select().count(),
            "episodes": Episode.select().count(),
            "events": Event.select().count(),
        }
