import asyncio
import os
from pathlib import Path
from typing import List
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from .chunker import chunk_file

import logging
logger = logging.getLogger(__name__)

class _EventHandler(FileSystemEventHandler):
    def __init__(self, server, repo_path: Path):
        self.server = server
        self.repo_path = repo_path
        super().__init__()

    def on_created(self, event):
        if not event.is_directory:
            asyncio.get_event_loop().create_task(self._process(event.src_path))

    def on_modified(self, event):
        if not event.is_directory:
            asyncio.get_event_loop().create_task(self._process(event.src_path))

    async def _process(self, path: str):
        rel_path = os.path.relpath(path, self.repo_path)
        try:
            chunks = chunk_file(path)
            meta = [{"path": rel_path, "chunk_id": i} for i in range(len(chunks))]
            await self.server.add_to_index(chunks, meta)
            logger.info(f"🔄 Indexed updated file {rel_path}")
        except Exception as e:
            logger.warning(f"Watcher error for {rel_path}: {e}")

class RepositoryWatcher:
    def __init__(self, server, repo_path: str):
        self.server = server
        self.repo_path = Path(repo_path)
        self.observer = Observer()

    def start(self):
        handler = _EventHandler(self.server, self.repo_path)
        self.observer.schedule(handler, str(self.repo_path), recursive=True)
        self.observer.start()
        logger.info(f"👀 Watching repository {self.repo_path}")

    async def stop(self):
        self.observer.stop()
        self.observer.join()
