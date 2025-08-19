# utils.py
import os
import uuid
import logging
import json
import time
import pathlib

import numpy as np
import pandas as pd
from werkzeug.utils import secure_filename

from sklearn.impute import KNNImputer
try:
    from ydata_profiling import ProfileReport
except Exception:
    ProfileReport = None

logger = logging.getLogger("autoeda.utils")


def save_uploaded_file(file_obj, upload_folder: str) -> str:
    os.makedirs(upload_folder, exist_ok=True)
    original = secure_filename(file_obj.filename)
    unique_name = f"{uuid.uuid4().hex[:8]}_{original}"
    path = os.path.join(upload_folder, unique_name)
    file_obj.save(path)
    logger.info("Saved uploaded file to %s", path)
    return path


def read_dataset(filepath: str) -> pd.DataFrame:
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".csv":
        return pd.read_csv(filepath)
    elif ext in (".xls", ".xlsx"):
        return pd.read_excel(filepath)
    else:
        raise ValueError("Unsupported file type")


def apply_schema(df: pd.DataFrame, schema: dict):
    logs = []
    if not schema:
        return df, logs
    mappings = schema.get("mappings", {})
    if mappings:
        df = df.rename(columns=mappings)
        logs.append({"step": "mapping", "mappings": mappings})
    types = schema.get("types", {})
    for col, t in types.items():
        if col in df.columns:
            try:
                if t in ("int", "float"):
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                elif t == "str":
                    df[col] = df[col].astype(str)
                logs.append({"step": "coerce", "column": col, "to": t})
            except Exception as e:
                logs.append({"step": "coerce_error", "column": col, "error": str(e)})
    keep = schema.get("keep")
    if keep:
        keep_existing = [c for c in keep if c in df.columns]
        df = df[keep_existing].copy()
        logs.append({"step": "keep", "kept_columns": keep_existing})
    if schema.get("weight_col"):
        logs.append({"step": "weight_col", "weight_col": schema.get("weight_col")})
    return df, logs


def impute_missing(df: pd.DataFrame, strategy="mean", columns=None, knn_k: int = 5):
    logs = []
    cols = columns if columns else df.columns.tolist()
    if strategy in ("mean", "median"):
        for c in cols:
            if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
                if strategy == "mean":
                    v = df[c].mean()
                else:
                    v = df[c].median()
                df[c] = df[c].fillna(v)
                logs.append({"step": "impute", "column": c, "strategy": strategy, "value": v})
    elif strategy == "knn":
        num_cols = [c for c in cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
        if num_cols:
            imputer = KNNImputer(n_neighbors=knn_k)
            df[num_cols] = imputer.fit_transform(df[num_cols])
            logs.append({"step": "impute", "strategy": "knn", "columns": num_cols, "k": knn_k})
    else:
        logs.append({"step": "impute", "error": f"unknown strategy {strategy}"})
    return df, logs


def detect_outliers(df: pd.DataFrame, method="iqr", columns=None, z_thresh=3.0, iqr_multiplier=1.5):
    cols = columns if columns else df.select_dtypes(include=[np.number]).columns.tolist()
    flags = {}
    for c in cols:
        if c not in df.columns:
            continue
        series = df[c]
        if series.dropna().empty:
            flags[c] = pd.Series([0] * len(df), index=df.index)
            continue
        if method == "iqr":
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            low = q1 - iqr_multiplier * iqr
            high = q3 + iqr_multiplier * iqr
            flag = (~series.between(low, high)).astype(int)
            flags[c] = flag
        elif method == "zscore":
            z = (series - series.mean()) / (series.std(ddof=0) if series.std(ddof=0) != 0 else 1)
            flag = (z.abs() > z_thresh).astype(int)
            flags[c] = flag
        else:
            flags[c] = pd.Series([0] * len(df), index=df.index)
    return flags


def apply_winsorization(df: pd.DataFrame, columns=None, limits=(0.01, 0.01)):
    cols = columns if columns else df.select_dtypes(include=[np.number]).columns.tolist()
    lower_p, upper_p = limits
    for c in cols:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            low_val = df[c].quantile(lower_p)
            high_val = df[c].quantile(1 - upper_p)
            df[c] = df[c].clip(lower=low_val, upper=high_val)
    return df


def apply_rules(df: pd.DataFrame, rules: list):
    violations = []
    for r in rules:
        expr = r.get("expr")
        desc = r.get("desc", "")
        try:
            mask = ~df.eval(expr)
            count = int(mask.sum())
            examples = df[mask].head(5).to_dict(orient="records")
            if count > 0:
                violations.append({"rule": r, "count": count, "examples": examples})
        except Exception as e:
            violations.append({"rule": r, "error": str(e)})
    return violations


def compute_weighted_mean_se(values, weights):
    x = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    mask = ~np.isnan(x) & ~np.isnan(w)
    x = x[mask]; w = w[mask]
    if x.size == 0 or w.sum() == 0:
        return (float("nan"), float("nan"))
    sw = w.sum()
    wm = (w * x).sum() / sw
    v = (w * (x - wm) ** 2).sum() / sw
    denom = (w**2).sum()
    n_eff = (sw**2 / denom) if denom > 0 else x.size
    se = (v / n_eff) ** 0.5
    return (float(wm), float(se))


def generate_eda_report(df: pd.DataFrame = None, output_folder: str = ".", base_name: str = "report"):
    if ProfileReport is None:
        raise ImportError("ydata_profiling not available. Install ydata-profiling and use Python <=3.12.")
    os.makedirs(output_folder, exist_ok=True)
    profile = ProfileReport(df, title=f"{base_name} - AutoEDA", explorative=True)
    report_filename = f"{base_name}.html"
    report_path = os.path.join(output_folder, report_filename)
    profile.to_file(report_path)
    logger.info("Wrote profile to %s", report_path)
    return report_path


# ----------------- PDF conversion helper -----------------
def html_to_pdf(html_path: str, pdf_path: str, playwright_wait: float = 0.8) -> str:
    """
    Convert an HTML file to PDF.
    Priority:
      1) Playwright (headless Chromium) - runs JS and yields faithful PDF
      2) WeasyPrint - static renderer (no JS)
      3) pdfkit/wkhtmltopdf - fallback (may or may not execute JS well)
    Returns the created pdf_path on success, otherwise raises RuntimeError.
    """
    logger = logging.getLogger("autoeda.utils.html_to_pdf")
    html_abspath = str(pathlib.Path(html_path).absolute())
    pdf_abspath = str(pathlib.Path(pdf_path).absolute())

    # ---------- 1) Playwright ----------
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": 1200, "height": 900})
            page.goto(pathlib.Path(html_abspath).as_uri(), wait_until="networkidle")
            time.sleep(playwright_wait)

            # hide interactive parts and add print rules
            page.add_style_tag(content="""
                .navbar, .topbar, .profile-header, .pg-controls, .btn, .dropdown { display: none !important; }
                .content, .container, .report-container { max-width: 100% !important; width: 100% !important; }
                .card, .section, .variable { page-break-inside: avoid; }
                * { -webkit-print-color-adjust: exact; color-adjust: exact; }
            """)
            page.add_style_tag(content="@page { size: A4; margin: 12mm; }")

            page.pdf(path=pdf_abspath, format="A4", print_background=True)
            browser.close()
        logger.info("Playwright PDF created: %s", pdf_abspath)
        return pdf_abspath
    except Exception as e:
        logger.info("Playwright conversion not available/failed: %s", str(e))

    # ---------- 2) WeasyPrint ----------
    try:
        from weasyprint import HTML
        HTML(filename=html_abspath).write_pdf(pdf_abspath)
        logger.info("WeasyPrint PDF created: %s", pdf_abspath)
        return pdf_abspath
    except Exception as e:
        logger.info("WeasyPrint not available/failed: %s", str(e))

    # ---------- 3) pdfkit / wkhtmltopdf ----------
    try:
        import pdfkit
        pdfkit.from_file(html_abspath, pdf_abspath)
        logger.info("pdfkit PDF created: %s", pdf_abspath)
        return pdf_abspath
    except Exception as e:
        logger.exception("pdfkit not available or failed: %s", str(e))

    raise RuntimeError("No HTML->PDF converter succeeded. Install Playwright (recommended), or WeasyPrint, or wkhtmltopdf.")
