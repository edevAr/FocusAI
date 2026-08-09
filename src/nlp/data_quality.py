"""Validaciones de calidad de datos para el pipeline NLP."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from config.settings import (
    DATA_QUALITY_REPORT_PATH,
    LABEL_PROCRASTINATION,
    LABEL_PRODUCTIVE,
    MAX_CLASS_IMBALANCE_RATIO,
    MIN_SAMPLES_PER_CLASS,
)


def _normalize_for_dedupe(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def assess_data_quality(
    df: pd.DataFrame,
    *,
    text_col: str = "texto",
    label_col: str = "etiqueta",
) -> dict[str, Any]:
    """Calcula métricas de calidad sin mutar el DataFrame."""
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"Se requieren columnas '{text_col}' y '{label_col}'")

    working = df.copy()
    working["_texto_str"] = working[text_col].astype(str)
    working["_is_null_text"] = df[text_col].isna()
    working["_is_blank"] = working["_texto_str"].str.strip().eq("") | working["_is_null_text"]
    working["_norm"] = working["_texto_str"].map(_normalize_for_dedupe)

    n_total = int(len(working))
    n_blank = int(working["_is_blank"].sum())
    n_exact_dupes = int(working.duplicated(subset=[text_col], keep="first").sum())
    n_norm_dupes = int(working.duplicated(subset=["_norm"], keep="first").sum())

    # Conteos por clase (soporta etiqueta int 0/1 o string)
    labels = working[label_col].copy()
    label_counts: dict[str, int] = {}
    for value, count in labels.value_counts(dropna=False).items():
        key = str(value)
        if value in (1, "1", LABEL_PRODUCTIVE, "productivo", "Productivo"):
            key = LABEL_PRODUCTIVE
        elif value in (0, "0", LABEL_PROCRASTINATION, "procrastinación", "procrastinacion", "Procrastinación"):
            key = LABEL_PROCRASTINATION
        label_counts[key] = label_counts.get(key, 0) + int(count)

    n_prod = int(label_counts.get(LABEL_PRODUCTIVE, 0))
    n_proc = int(label_counts.get(LABEL_PROCRASTINATION, 0))
    minority = min(n_prod, n_proc) if (n_prod and n_proc) else 0
    majority = max(n_prod, n_proc)
    imbalance_ratio = float(majority / minority) if minority else float("inf")

    issues: list[str] = []
    if n_blank:
        issues.append(f"{n_blank} textos vacíos o nulos")
    if n_norm_dupes:
        issues.append(f"{n_norm_dupes} duplicados (texto normalizado)")
    if n_prod < MIN_SAMPLES_PER_CLASS or n_proc < MIN_SAMPLES_PER_CLASS:
        issues.append(
            f"clase con menos de {MIN_SAMPLES_PER_CLASS} muestras "
            f"(Productivo={n_prod}, Procrastinación={n_proc})"
        )
    if imbalance_ratio > MAX_CLASS_IMBALANCE_RATIO:
        issues.append(
            f"desbalance de clases ratio={imbalance_ratio:.2f} "
            f"(umbral={MAX_CLASS_IMBALANCE_RATIO})"
        )

    return {
        "n_total": n_total,
        "n_blank_or_null": n_blank,
        "n_exact_duplicates": n_exact_dupes,
        "n_normalized_duplicates": n_norm_dupes,
        "class_counts": {
            LABEL_PRODUCTIVE: n_prod,
            LABEL_PROCRASTINATION: n_proc,
        },
        "class_balance_ratio": None if imbalance_ratio == float("inf") else round(imbalance_ratio, 4),
        "issues": issues,
        "ok": len(issues) == 0,
    }


def clean_quality_issues(
    df: pd.DataFrame,
    *,
    text_col: str = "texto",
    label_col: str = "etiqueta",
    drop_duplicates: bool = True,
    drop_blank: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Elimina vacíos/duplicados y devuelve (df_limpio, reporte)."""
    report_before = assess_data_quality(df, text_col=text_col, label_col=label_col)
    working = df.copy()
    removed = {"blank": 0, "duplicates": 0}

    if drop_blank:
        mask_blank = working[text_col].isna() | working[text_col].astype(str).str.strip().eq("")
        removed["blank"] = int(mask_blank.sum())
        working = working.loc[~mask_blank].copy()

    if drop_duplicates:
        norm = working[text_col].astype(str).map(_normalize_for_dedupe)
        dup_mask = norm.duplicated(keep="first")
        removed["duplicates"] = int(dup_mask.sum())
        working = working.loc[~dup_mask].copy()

    working = working.reset_index(drop=True)
    report_after = assess_data_quality(working, text_col=text_col, label_col=label_col)
    report = {
        "before": report_before,
        "removed": removed,
        "after": report_after,
        "ok": report_after["ok"],
        "issues": report_after["issues"],
    }
    return working, report


def save_quality_report(report: dict[str, Any], path: Path | str | None = None) -> Path:
    out = Path(path) if path else DATA_QUALITY_REPORT_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    return out
