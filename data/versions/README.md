# Dataset versions

Cada carpeta `vX.Y.Z/` contiene:

- `journal_entries.csv` — snapshot inmutable del corpus
- `manifest.json` — metadata (sha256, filas, fecha)

`current.json` apunta a la versión activa.

```bash
python scripts/snapshot_dataset.py --version v1.2.0 --note "más ejemplos de procrastinación"
```
