"""
AutoMark - teacher web app.

Run:  python app.py      then open  http://127.0.0.1:5000

Flow: New test -> upload marking scheme -> check scheme -> upload student scripts
      -> own model reads handwriting -> photo quality check (re-upload if unreadable) -> teacher fixes text
      -> Sentence-BERT (or keywords) marks -> results / CSV.   Everything runs locally, no API.
"""
import csv
import io
import json
import os
import re
import uuid

from flask import (Flask, Response, abort, flash, redirect, render_template, request,
                   send_from_directory, url_for)
from werkzeug.utils import secure_filename

from grader import db, marker
from grader.reader import read_script

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "automark-local")
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024   # 100 MB per upload

SCRIPT_EXT = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
SCHEME_EXT = SCRIPT_EXT | {".txt", ".docx"}


# ---------------------------------------------------------------- helpers
def _save_upload(f, allowed):
    name = secure_filename(f.filename or "")
    ext = os.path.splitext(name)[1].lower()
    if ext not in allowed:
        raise ValueError(f"File type {ext or '(none)'} not allowed: {f.filename}")
    os.makedirs(db.UPLOAD_DIR, exist_ok=True)
    stored = f"{uuid.uuid4().hex[:10]}_{name}"
    path = os.path.join(db.UPLOAD_DIR, stored)
    f.save(path)
    return path


def _get_test(con, test_id):
    t = db.row_to_dict(con.execute("SELECT * FROM tests WHERE id=?", (test_id,)).fetchone())
    if not t:
        abort(404)
    return t


def _get_script(con, script_id):
    s = db.row_to_dict(con.execute("SELECT * FROM scripts WHERE id=?", (script_id,)).fetchone())
    if not s:
        abort(404)
    return s


def _grade(scheme, text, engine):
    if engine == "similarity":
        return marker.grade_similarity(scheme, text)
    return marker.grade_keywords(scheme, text)


def _add_scheme_words(con, script_id):
    """Give the dictionary correction the subject words of this test's marking scheme."""
    from grader import lexicon
    row = con.execute("SELECT t.scheme_json FROM scripts s JOIN tests t ON t.id = s.test_id WHERE s.id=?",
                      (script_id,)).fetchone()
    if row:
        scheme = json.loads(row[0])
        text = " ".join(q.get("question", "") + " " + " ".join(p["text"] for p in q.get("points", []))
                        for q in scheme.get("questions", []))
        lexicon.add_words(re.findall(r"[A-Za-z]+(?:['-][A-Za-z]+)*", text))


def _read_into(con, script_id, paths):
    """Read a script with our own model, run the photo quality check, store both."""
    _add_scheme_words(con, script_id)
    prefix = os.path.join(db.UPLOAD_DIR, f"preview_{script_id}_{uuid.uuid4().hex[:6]}")
    r = read_script(paths, preview_prefix=prefix)
    previews = [os.path.basename(p["preview"]) for p in r["pages"] if p["preview"]]
    con.execute("UPDATE scripts SET text=?, words=?, low_conf=?, previews_json=?, quality=?, "
                "quality_reasons_json=?, status='read', result_json=NULL WHERE id=?",
                (r["text"], r["words"], r["low_conf"], json.dumps(previews), r["quality"],
                 json.dumps(r["reasons"]), script_id))
    return r


QUALITY_LABEL = {"ok": "OK", "warn": "Check", "bad": "Re-upload"}


@app.context_processor
def _inject():
    from grader.reader import RETAKE_TIPS
    return {"settings": db.load_settings(), "quality_label": QUALITY_LABEL, "retake_tips": RETAKE_TIPS}


# ---------------------------------------------------------------- pages
@app.route("/")
def home():
    con = db.connect()
    tests = con.execute("""SELECT t.*, COUNT(s.id) AS n_scripts,
                           SUM(CASE WHEN s.status='marked' THEN 1 ELSE 0 END) AS n_marked
                           FROM tests t LEFT JOIN scripts s ON s.test_id=t.id
                           GROUP BY t.id ORDER BY t.created DESC""").fetchall()
    return render_template("home.html", tests=tests)


@app.route("/settings", methods=["GET", "POST"])
def settings():
    if request.method == "POST":
        db.save_settings(request.form.get("engine", "similarity"))
        flash("Settings saved.")
        return redirect(url_for("home"))
    from grader import training_data
    return render_template("settings.html", sbert_model=marker.SBERT_MODEL, training=training_data.stats())


@app.route("/training_data.zip")
def training_zip():
    from grader import training_data
    return Response(training_data.zip_bytes(), mimetype="application/zip",
                    headers={"Content-Disposition": "attachment; filename=training_data.zip"})


@app.route("/test/new", methods=["GET", "POST"])
def new_test():
    if request.method == "GET":
        return render_template("new_test.html")
    name = request.form.get("name", "").strip() or "Untitled test"
    pasted = request.form.get("scheme_text", "").strip()
    f = request.files.get("scheme_file")
    try:
        if f and f.filename:
            path = _save_upload(f, SCHEME_EXT)
            text = marker.extract_file_text(path)
            if not text.strip():
                raise ValueError("Only typed PDF / Word / TXT marking schemes can be read. "
                                 "For a photo or scan, paste the scheme as text.")
            scheme = marker.parse_scheme_simple(text)
        elif pasted:
            scheme = marker.parse_scheme_simple(pasted)
        else:
            raise ValueError("Upload a marking scheme file or paste it as text.")
    except Exception as e:  # noqa: BLE001 - show any failure to the teacher
        flash(f"Could not read the marking scheme: {e}", "error")
        return render_template("new_test.html", name=name, scheme_text=pasted)
    con = db.connect()
    cur = con.execute("INSERT INTO tests(name, created, scheme_json) VALUES (?,?,?)",
                      (name, db.now(), json.dumps(scheme)))
    con.commit()
    flash("Marking scheme read. Please check it before uploading answer scripts.")
    return redirect(url_for("edit_scheme", test_id=cur.lastrowid))


@app.route("/test/<int:test_id>/scheme", methods=["GET", "POST"])
def edit_scheme(test_id):
    con = db.connect()
    t = _get_test(con, test_id)
    if request.method == "POST":
        scheme = marker.parse_scheme_simple(request.form.get("scheme_text", ""))
        if not scheme["questions"]:
            flash("No questions found. Use the format shown below the box.", "error")
            return render_template("scheme.html", t=t, text=request.form.get("scheme_text", ""))
        con.execute("UPDATE tests SET scheme_json=?, name=? WHERE id=?",
                    (json.dumps(scheme), request.form.get("name", t["name"]).strip() or t["name"], test_id))
        con.commit()
        flash("Marking scheme saved.")
        return redirect(url_for("test_page", test_id=test_id))
    return render_template("scheme.html", t=t, text=marker.scheme_to_text(t["scheme"]))


@app.route("/test/<int:test_id>")
def test_page(test_id):
    con = db.connect()
    t = _get_test(con, test_id)
    scripts = [db.row_to_dict(r) for r in
               con.execute("SELECT * FROM scripts WHERE test_id=? ORDER BY student COLLATE NOCASE", (test_id,))]
    max_total = sum(q["max_marks"] for q in t["scheme"]["questions"])
    return render_template("test.html", t=t, scripts=scripts, max_total=max_total)


@app.route("/test/<int:test_id>/upload", methods=["POST"])
def upload_scripts(test_id):
    con = db.connect()
    _get_test(con, test_id)
    mode = request.form.get("mode", "one")
    files = [f for f in request.files.getlist("files") if f and f.filename]
    if not files:
        flash("Choose at least one file.", "error")
        return redirect(url_for("test_page", test_id=test_id))
    try:
        if mode == "one":
            # one student, files are that student's pages in order
            student = request.form.get("student", "").strip() or os.path.splitext(files[0].filename)[0]
            groups = [(student, request.form.get("roll", "").strip(),
                       [_save_upload(f, SCRIPT_EXT) for f in sorted(files, key=lambda f: f.filename)])]
        else:
            # many students: each file is one student's whole script, file name = student name
            groups = [(os.path.splitext(f.filename)[0], "", [_save_upload(f, SCRIPT_EXT)]) for f in files]
    except ValueError as e:
        flash(str(e), "error")
        return redirect(url_for("test_page", test_id=test_id))

    done, failed, needs = 0, [], []
    for student, roll, paths in groups:
        cur = con.execute("INSERT INTO scripts(test_id, student, roll, files_json, created) VALUES (?,?,?,?,?)",
                          (test_id, student, roll, json.dumps([os.path.basename(p) for p in paths]), db.now()))
        try:
            r = _read_into(con, cur.lastrowid, paths)
            done += 1
            if r["quality"] == "bad":
                needs.append(student)
        except Exception as e:  # noqa: BLE001
            con.execute("UPDATE scripts SET status='error', text=? WHERE id=?", (f"Reading failed: {e}", cur.lastrowid))
            failed.append(student)
        con.commit()
    flash(f"Read {done} script(s)." + (f" Failed: {', '.join(failed)}" if failed else ""),
          "error" if failed else "info")
    if needs:
        flash(f"{len(needs)} photo(s) could not be read reliably and should be re-uploaded: {', '.join(needs)}.",
              "error")
    if mode == "one" and done == 1:
        return redirect(url_for("script_page", script_id=cur.lastrowid))
    return redirect(url_for("test_page", test_id=test_id))


@app.route("/script/<int:script_id>", methods=["GET", "POST"])
def script_page(script_id):
    con = db.connect()
    s = _get_script(con, script_id)
    t = _get_test(con, s["test_id"])
    if request.method == "POST":
        action = request.form.get("action")
        force = action == "mark_force"            # "Mark anyway" (teacher confirmed in the browser)
        if force:
            action = "mark"
        if action in ("save_text", "mark"):
            text = request.form.get("text", "")
            con.execute("UPDATE scripts SET text=?, student=?, roll=? WHERE id=?",
                        (text, request.form.get("student", s["student"]), request.form.get("roll", s["roll"]),
                         script_id))
            con.commit()
            if action == "mark" and s["quality"] == "bad" and not force:
                flash("This photo could not be read reliably. Re-upload a clearer photo, "
                      "or use 'Mark anyway'.", "error")
            elif action == "mark":
                st = db.load_settings()
                engine = request.form.get("engine") or st["engine"]
                try:
                    result = _grade(t["scheme"], text, engine)
                    con.execute("UPDATE scripts SET result_json=?, status='marked' WHERE id=?",
                                (json.dumps(result), script_id))
                    con.commit()
                    flash(f"Marked: {result['total']:g} / {result['max']:g}")
                except Exception as e:  # noqa: BLE001
                    flash(f"Marking failed: {e}", "error")
            else:
                flash("Text saved.")
        elif action == "override" and s["result"]:
            res = s["result"]
            for q in res["questions"]:
                v = request.form.get(f"q_{q['id']}")
                if v not in (None, ""):
                    q["awarded"] = max(0.0, min(float(v), q["max"]))
                    q["teacher_changed"] = True
            res["total"] = sum(q["awarded"] for q in res["questions"])
            con.execute("UPDATE scripts SET result_json=? WHERE id=?", (json.dumps(res), script_id))
            con.commit()
            flash(f"Marks updated: {res['total']:g} / {res['max']:g}")
        elif action == "save_training":
            # save the teacher's corrected text first, then store word images + correct spelling
            text = request.form.get("text", s["text"])
            con.execute("UPDATE scripts SET text=? WHERE id=?", (text, script_id))
            con.commit()
            from grader import training_data
            from grader.reader import get_model
            _add_scheme_words(con, script_id)
            saved, skipped = training_data.export_script(dict(s, text=text), get_model().chars)
            flash(f"Saved {saved} words as training data ({skipped} skipped: unchecked [?] words "
                  f"or lines that could not be matched). Total now: {training_data.stats()['words']} words.")
        elif action == "reread":
            paths = [os.path.join(db.UPLOAD_DIR, f) for f in s["files"]]
            try:
                _read_into(con, script_id, paths)
                con.commit()
                flash("Handwriting read again.")
            except Exception as e:  # noqa: BLE001
                flash(f"Reading failed: {e}", "error")
        return redirect(url_for("script_page", script_id=script_id))
    ids = [r["id"] for r in con.execute("SELECT id FROM scripts WHERE test_id=? ORDER BY student COLLATE NOCASE",
                                         (t["id"],))]
    pos = ids.index(script_id)
    nav = {"prev": ids[pos - 1] if pos > 0 else None, "next": ids[pos + 1] if pos + 1 < len(ids) else None,
           "pos": pos + 1, "count": len(ids)}
    return render_template("script.html", s=s, t=t, nav=nav)


@app.route("/script/<int:script_id>/reupload", methods=["POST"])
def reupload_script(script_id):
    """Replace all pages of one student's script with new files, then read them again."""
    con = db.connect()
    s = _get_script(con, script_id)
    files = [f for f in request.files.getlist("files") if f and f.filename]
    if not files:
        flash("Choose the new photo(s) or PDF first.", "error")
        return redirect(url_for("script_page", script_id=script_id))
    try:
        paths = [_save_upload(f, SCRIPT_EXT) for f in sorted(files, key=lambda f: f.filename)]
    except ValueError as e:
        flash(str(e), "error")
        return redirect(url_for("script_page", script_id=script_id))
    con.execute("UPDATE scripts SET files_json=? WHERE id=?",
                (json.dumps([os.path.basename(p) for p in paths]), script_id))
    try:
        r = _read_into(con, script_id, paths)
        con.commit()
        flash({"ok": "New photo read. Quality looks fine.",
               "warn": "New photo read. Please check the warnings below.",
               "bad": "The new photo still could not be read reliably."}[r["quality"]],
              "error" if r["quality"] == "bad" else "info")
    except Exception as e:  # noqa: BLE001
        con.execute("UPDATE scripts SET status='error', text=? WHERE id=?", (f"Reading failed: {e}", script_id))
        con.commit()
        flash(f"Reading failed: {e}", "error")
    return redirect(url_for("script_page", script_id=s["id"]))


@app.route("/test/<int:test_id>/mark_all", methods=["POST"])
def mark_all(test_id):
    con = db.connect()
    t = _get_test(con, test_id)
    st = db.load_settings()
    n, errors = 0, []
    skipped = [r["student"] for r in con.execute(
        "SELECT student FROM scripts WHERE test_id=? AND status='read' AND quality='bad'", (test_id,))]
    for r in con.execute("SELECT * FROM scripts WHERE test_id=? AND status='read' AND quality!='bad'",
                         (test_id,)).fetchall():
        try:
            res = _grade(t["scheme"], r["text"], st["engine"])
            con.execute("UPDATE scripts SET result_json=?, status='marked' WHERE id=?", (json.dumps(res), r["id"]))
            con.commit()
            n += 1
        except Exception as e:  # noqa: BLE001
            errors.append(f"{r['student']}: {e}")
    flash(f"Marked {n} script(s)." + (" Problems: " + "; ".join(errors[:3]) if errors else ""),
          "error" if errors else "info")
    if skipped:
        flash(f"Not marked, photo needs re-upload: {', '.join(skipped)}.", "error")
    return redirect(url_for("test_page", test_id=test_id))


@app.route("/script/<int:script_id>/delete", methods=["POST"])
def delete_script(script_id):
    con = db.connect()
    s = _get_script(con, script_id)
    con.execute("DELETE FROM scripts WHERE id=?", (script_id,))
    con.commit()
    flash(f"Removed {s['student']}.")
    return redirect(url_for("test_page", test_id=s["test_id"]))


@app.route("/test/<int:test_id>/delete", methods=["POST"])
def delete_test(test_id):
    con = db.connect()
    t = _get_test(con, test_id)
    con.execute("DELETE FROM tests WHERE id=?", (test_id,))
    con.commit()
    flash(f"Deleted test {t['name']}.")
    return redirect(url_for("home"))


@app.route("/test/<int:test_id>/results.csv")
def export_csv(test_id):
    con = db.connect()
    t = _get_test(con, test_id)
    qs = t["scheme"]["questions"]
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["Student", "Roll"] + [f"Q{q['id']} (/{q['max_marks']:g})" for q in qs] +
               ["Total", "Out of", "Percent", "Status"])
    for r in con.execute("SELECT * FROM scripts WHERE test_id=? ORDER BY student COLLATE NOCASE", (test_id,)):
        s = db.row_to_dict(r)
        res = s["result"] or {}
        by = {q["id"]: q["awarded"] for q in res.get("questions", [])}
        total, mx = res.get("total"), res.get("max") or sum(q["max_marks"] for q in qs)
        w.writerow([s["student"], s["roll"]] + [by.get(q["id"], "") for q in qs] +
                   [total if total is not None else "", mx,
                    f"{100 * total / mx:.1f}" if total is not None and mx else "", s["status"]])
    name = secure_filename(t["name"]) or "results"
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": f"attachment; filename={name}_results.csv"})


@app.route("/files/<path:name>")
def files(name):
    return send_from_directory(db.UPLOAD_DIR, name)


@app.template_filter("date")
def date(ts):
    import datetime
    return datetime.datetime.fromtimestamp(ts).strftime("%d %b %Y")


@app.template_filter("pct")
def pct(v, total):
    return f"{100 * v / total:.0f}%" if total else ""


if __name__ == "__main__":
    db.connect()
    print("\n  AutoMark is running. Open  http://127.0.0.1:5000  in your browser.\n")
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)), debug=False)
