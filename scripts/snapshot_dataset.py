"""Versionado simple de datasets en data/versions/ (alternativa ligera a DVC)."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import DATA_VERSIONS_DIR, DATASET_VERSION, RAW_CSV_PATH


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot_dataset(
    version: str | None = None,
    source: Path | str | None = None,
    note: str = "",
) -> Path:
    version = version or DATASET_VERSION
    src = Path(source) if source else RAW_CSV_PATH
    if not src.exists():
        raise FileNotFoundError(f"Dataset fuente no encontrado: {src}")

    dest_dir = DATA_VERSIONS_DIR / version
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_csv = dest_dir / "journal_entries.csv"
    shutil.copy2(src, dest_csv)

    # También sincroniza data/raw como "current working copy"
    RAW_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() != RAW_CSV_PATH.resolve():
        shutil.copy2(src, RAW_CSV_PATH)

    lines = dest_csv.read_text(encoding="utf-8").splitlines()
    n_rows = max(0, len(lines) - 1)
    manifest = {
        "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": str(src),
        "path": str(dest_csv.relative_to(PROJECT_ROOT)),
        "sha256": _sha256(dest_csv),
        "n_rows": n_rows,
        "note": note
        or "Snapshot del dataset de diarios Productivo/Procrastinación",
    }
    manifest_path = dest_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    current = {
        "current": version,
        "updated_at": manifest["created_at"],
        "manifest": str(manifest_path.relative_to(PROJECT_ROOT)),
    }
    (DATA_VERSIONS_DIR / "current.json").write_text(
        json.dumps(current, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    readme = DATA_VERSIONS_DIR / "README.md"
    if not readme.exists():
        readme.write_text(
            """# Dataset versions

Cada carpeta `vX.Y.Z/` contiene:

- `journal_entries.csv` — snapshot inmutable del corpus
- `manifest.json` — metadata (sha256, filas, fecha)

`current.json` apunta a la versión activa.

```bash
python scripts/snapshot_dataset.py --version v1.2.0 --note "más ejemplos de procrastinación"
```
""",
            encoding="utf-8",
        )
    return dest_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Snapshot versionado del dataset FocusAI")
    parser.add_argument("--version", default=DATASET_VERSION)
    parser.add_argument("--source", default=str(RAW_CSV_PATH))
    parser.add_argument("--note", default="")
    args = parser.parse_args()
    dest = snapshot_dataset(version=args.version, source=args.source, note=args.note)
    print(f"Snapshot OK: {dest}")


if __name__ == "__main__":
    main()
