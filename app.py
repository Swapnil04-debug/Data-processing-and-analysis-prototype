# app.py
import os
import json
import logging
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash

from utils import (
    save_uploaded_file,
    read_dataset,
    apply_schema,
    impute_missing,
    detect_outliers,
    apply_winsorization,
    apply_rules,
    compute_weighted_mean_se,
    generate_eda_report,
    html_to_pdf
)
from reporting import report_exists, save_workflow_log, load_workflow_log

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("autoeda")

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
REPORT_FOLDER = os.path.join(BASE_DIR, "reports")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {".csv", ".xls", ".xlsx"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["REPORT_FOLDER"] = REPORT_FOLDER
app.secret_key = "dev-secret"


def allowed_file(filename: str) -> bool:
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET"])
def index():
    return render_template("upload.html")

@app.route('/about')
def about():
    """Renders the about page."""
    return render_template('about.html')

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        flash("No file uploaded")
        return redirect(url_for("index"))    

    file = request.files["file"]
    if file.filename == "":
        flash("No file selected")
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        flash("Unsupported file type")
        return redirect(url_for("index"))

    schema_text = request.form.get("schema_json", "").strip()
    schema = None
    if schema_text:
        try:
            schema = json.loads(schema_text)
        except Exception as e:
            flash(f"Invalid JSON schema: {e}")
            return redirect(url_for("index"))

    try:
        saved_path = save_uploaded_file(file, app.config["UPLOAD_FOLDER"])
        logger.info("Saved file to %s", saved_path)

        df_raw = read_dataset(saved_path)
        raw_report_path = generate_eda_report(
            df=df_raw,
            output_folder=app.config["REPORT_FOLDER"],
            base_name=os.path.splitext(os.path.basename(saved_path))[0] + "_raw"
        )

        # try to create PDF for raw report (best effort)
        raw_pdf = ""
        try:
            raw_pdf_path = os.path.splitext(raw_report_path)[0] + ".pdf"
            html_to_pdf(raw_report_path, raw_pdf_path)
            raw_pdf = os.path.basename(raw_pdf_path)
        except Exception as e:
            logger.info("Raw PDF generation skipped/failed: %s", str(e))

        workflow_log = []
        workflow_log.append({"step": "upload", "file": saved_path, "rows": int(df_raw.shape[0]), "cols": int(df_raw.shape[1])})

        if schema:
            df_clean = df_raw.copy()
            df_clean, log_schema = apply_schema(df_clean, schema)
            workflow_log.extend(log_schema)

            impute_cfg = schema.get("impute", {})
            if impute_cfg:
                strategy = impute_cfg.get("strategy", "mean")
                cols = impute_cfg.get("columns", None)
                knn_k = impute_cfg.get("knn_k", 5)
                df_clean, log_impute = impute_missing(df_clean, strategy=strategy, columns=cols, knn_k=knn_k)
                workflow_log.extend(log_impute)

            out_cfg = schema.get("outlier", {})
            if out_cfg:
                method = out_cfg.get("method", "iqr")
                cols = out_cfg.get("columns", None)
                flags = detect_outliers(df_clean, method=method, columns=cols,
                                         z_thresh=out_cfg.get("z_thresh", 3.0),
                                         iqr_multiplier=out_cfg.get("iqr_multiplier", 1.5))
                workflow_log.append({"step": "outlier_detection", "method": method, "flags_summary": {c: int(flags[c].sum()) for c in flags}})
                if schema.get("winsorize"):
                    limits = tuple(schema.get("winsorize", {}).get("limits", (0.01, 0.01)))
                    df_clean = apply_winsorization(df_clean, columns=cols, limits=limits)
                    workflow_log.append({"step": "winsorization", "limits": limits})
            else:
                flags = {}

            rules = schema.get("rules", [])
            if rules:
                violations = apply_rules(df_clean, rules)
                workflow_log.append({"step": "rules_validation", "violations": violations})

            estimates = {}
            for est in schema.get("estimates", []):
                if isinstance(est, str):
                    var = est; by = None
                else:
                    var = est.get("variable"); by = est.get("by")
                wt_col = schema.get("weight_col")
                if by and by in df_clean.columns:
                    groups = df_clean.groupby(by)
                    estimates[var] = {}
                    for gname, gdf in groups:
                        if wt_col and wt_col in gdf.columns:
                            mean, se = compute_weighted_mean_se(gdf[var].dropna().values, gdf[wt_col].fillna(0).values)
                        else:
                            mean = gdf[var].mean()
                            se = gdf[var].std(ddof=1) / (len(gdf) ** 0.5) if len(gdf) > 1 else None
                        estimates[var][str(gname)] = {"mean": mean, "se": se, "n": int(len(gdf))}
                else:
                    if wt_col and wt_col in df_clean.columns:
                        mean, se = compute_weighted_mean_se(df_clean[var].dropna().values, df_clean[wt_col].fillna(0).values)
                    else:
                        mean = df_clean[var].mean()
                        se = df_clean[var].std(ddof=1) / (len(df_clean[var].dropna()) ** 0.5) if df_clean[var].dropna().shape[0] > 1 else None
                    estimates[var] = {"mean": mean, "se": se, "n": int(df_clean[var].dropna().shape[0])}

            workflow_log.append({"step": "estimates", "estimates": estimates})

            base_clean = os.path.splitext(os.path.basename(saved_path))[0] + "_cleaned"
            workflow_log_path = os.path.join(app.config["REPORT_FOLDER"], base_clean + "_workflow.json")
            save_workflow_log(workflow_log_path, workflow_log)

            cleaned_report_path = generate_eda_report(df=df_clean, output_folder=app.config["REPORT_FOLDER"], base_name=base_clean)

            # try to create PDF for cleaned report (best effort)
            clean_pdf = ""
            try:
                clean_pdf_path = os.path.splitext(cleaned_report_path)[0] + ".pdf"
                html_to_pdf(cleaned_report_path, clean_pdf_path)
                clean_pdf = os.path.basename(clean_pdf_path)
            except Exception as e:
                logger.info("Clean PDF generation skipped/failed: %s", str(e))

            raw_name = os.path.basename(raw_report_path)
            clean_name = os.path.basename(cleaned_report_path)
            log_name = os.path.basename(workflow_log_path)
            return redirect(url_for("view_report", raw=raw_name, clean=clean_name, log=log_name, raw_pdf=raw_pdf, clean_pdf=clean_pdf))

        else:
            raw_name = os.path.basename(raw_report_path)
            return redirect(url_for("view_report", raw=raw_name, clean="", log="", raw_pdf=raw_pdf, clean_pdf=""))

    except Exception as e:
        logger.exception("processing failed")
        flash(f"Processing failed: {e}")
        return redirect(url_for("index"))


@app.route("/view")
def view_report():
    raw = request.args.get("raw", "")
    clean = request.args.get("clean", "")
    log = request.args.get("log", "")
    raw_pdf = request.args.get("raw_pdf", "")
    clean_pdf = request.args.get("clean_pdf", "")

    workflow = None
    if log:
        workflow = load_workflow_log(os.path.join(app.config["REPORT_FOLDER"], log))

    return render_template("view_report.html",
                           raw_report=raw,
                           cleaned_report=clean,
                           workflow_log=workflow,
                           log_file=log,
                           raw_pdf=raw_pdf,
                           clean_pdf=clean_pdf)


@app.route("/reports/<path:filename>")
def serve_report(filename):
    return send_from_directory(app.config["REPORT_FOLDER"], filename)


@app.route("/download/<path:filename>")
def download_report(filename):
    return send_from_directory(app.config["REPORT_FOLDER"], filename, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
