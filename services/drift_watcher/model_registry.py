"""Lightweight, filesystem-based model versioning for the Drift Watcher's
promotion workflow: a metadata.json per version plus a `current` pointer
file. Enough to version, promote, and roll back without standing up
separate registry infrastructure a laptop-scale project doesn't need.
"""

from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class ModelVersion:
    version_id: str
    component: str
    metrics: dict
    created_at: str
    promoted: bool = False


class ModelRegistry:
    def __init__(self, root: str | Path = "models/registry") -> None:
        self._root = Path(root)

    def _component_dir(self, component: str) -> Path:
        d = self._root / component
        d.mkdir(parents=True, exist_ok=True)
        return d

    def register(self, component: str, artifact_dir: str | Path, metrics: dict) -> ModelVersion:
        """Copies the trained artifact directory (e.g. what
        BouncerModel.save()/DetectiveModel.save() produced) into a new,
        versioned slot under the registry.
        """
        version_id = uuid.uuid4().hex[:12]
        version = ModelVersion(
            version_id=version_id, component=component, metrics=metrics,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        dest = self._component_dir(component) / version_id
        shutil.copytree(artifact_dir, dest)
        (dest / "version.json").write_text(json.dumps(asdict(version), indent=2))
        return version

    def promote(self, component: str, version_id: str) -> None:
        """Only if a candidate beats the current model does the caller
        (retrainer.py) call this — the registry itself doesn't compare,
        it just records the decision, matching FR (Drift Watcher):
        "promotes a new model only if it beats the current one."
        """
        version_dir = self._component_dir(component) / version_id
        if not version_dir.exists():
            raise FileNotFoundError(f"no such version: {component}/{version_id}")
        pointer = self._component_dir(component) / "current.json"
        pointer.write_text(json.dumps({"version_id": version_id}))

    def get_current(self, component: str) -> ModelVersion | None:
        pointer = self._component_dir(component) / "current.json"
        if not pointer.exists():
            return None
        version_id = json.loads(pointer.read_text())["version_id"]
        return self.get_version(component, version_id)

    def get_version(self, component: str, version_id: str) -> ModelVersion | None:
        meta_path = self._component_dir(component) / version_id / "version.json"
        if not meta_path.exists():
            return None
        return ModelVersion(**json.loads(meta_path.read_text()))

    def current_artifact_dir(self, component: str) -> Path | None:
        current = self.get_current(component)
        if current is None:
            return None
        return self._component_dir(component) / current.version_id
