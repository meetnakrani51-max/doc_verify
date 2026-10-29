import os, json, joblib
import numpy as np, pandas as pd, streamlit as st
from PIL import Image
import core

st.set_page_config(page_title="DocVerify", page_icon="🛡️", layout="centered", initial_sidebar_state="collapsed")
st.markdown("""<style>
:root{--ink:#0b1f3a;--muted:#5b6475;--drop:#f6f9ff;--drop2:#eef4ff;--card:#ffffff;--line:#e3e8f2}
@media (prefers-color-scheme:dark){:root{--ink:#f1f5fb;--muted:#a3adc2;--drop:#161d2b;--drop2:#1c2536;--card:#161d2b;--line:#2b3548}}
[data-testid="stFileUploaderDropzone"] *{color:var(--ink)!important}
[data-testid="stFileUploaderDropzone"] small{color:var(--muted)!important}
footer,#MainMenu,[data-testid="stSidebar"],[data-testid="stSidebarCollapsedControl"],header[data-testid="stHeader"]{display:none!important}
.block-container{padding-top:1.2rem;max-width:820px}
h1,h2,h3{color:var(--ink);font-weight:700;letter-spacing:-.02em}
.hero{text-align:center;margin:34px 0 8px}.hero h1{font-size:2.6rem;margin-bottom:6px}.hero p{color:var(--muted);font-size:1.05rem}
[data-testid="stFileUploaderDropzone"]{border:2px dashed #2563eb;background:var(--drop);border-radius:16px;padding:46px 20px}
[data-testid="stFileUploaderDropzone"]:hover{background:var(--drop2)}
.logo{font-weight:800;font-size:1.25rem;color:var(--ink)}
.trust{display:flex;justify-content:center;gap:22px;color:var(--muted);font-size:.85rem;margin:10px 0 18px;flex-wrap:wrap}
.res{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:30px 26px;text-align:center;box-shadow:0 6px 24px rgba(11,31,58,.06);margin-top:10px}
.ico{width:64px;height:64px;border-radius:50%;margin:0 auto 12px;display:flex;align-items:center;justify-content:center;font-size:2rem;color:#fff}
.res h2{margin:0 0 4px}.res .sub{color:var(--muted);margin-bottom:18px}
.row{display:flex;justify-content:space-between;align-items:center;padding:12px 4px;border-top:1px solid var(--line);text-align:left}
.pill{padding:3px 12px;border-radius:99px;font-size:.8rem;font-weight:600}
.ok{background:#e3f6ec;color:#12794a}.bad{background:#fde8e6;color:#b3261e}.mid{background:#fff3d6;color:#9a6200}.na{background:rgba(128,138,160,.18);color:var(--muted)}
.stButton>button{border-radius:10px}
</style>""", unsafe_allow_html=True)

if not os.path.exists("model.pkl"):
    st.error("Model not found. Run `python train.py` first."); st.stop()
core.seed_records()
bundle = joblib.load("model.pkl"); metrics = json.load(open("metrics.json"))
if "rev_t" not in st.session_state: st.session_state.update(rev_t=0.40, forged_t=0.65, boxes=True, sample=None)

lc, nc = st.columns([1, 2.2])
lc.markdown('<div class="logo">🛡️ DocVerify</div>', unsafe_allow_html=True)
NAV = ["Verify", "Performance", "History", "How it works"]
page = nc.segmented_control("nav", NAV, default="Verify", label_visibility="collapsed") or "Verify"
rev_t, forged_t, show_boxes = st.session_state.rev_t, st.session_state.forged_t, st.session_state.boxes


def verdict(p):
    return ("Likely forged", 2) if p >= forged_t else ("Needs review", 1) if p >= rev_t else ("Likely genuine", 0)


def row(label, text, cls):
    return f'<div class="row"><span>{label}</span><span class="pill {cls}">{text}</span></div>'


# ---------------- VERIFY ----------------
if page == "Verify":
    st.markdown('<div class="hero"><h1>Verify your document</h1><p>Check a certificate, marksheet or ID for signs of tampering — in seconds.</p></div>', unsafe_allow_html=True)
    up = st.file_uploader("Drop your document here", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    st.markdown('<div class="trust"><span>🔒 Analysed locally</span><span>🧠 AI image forensics</span><span>🔗 QR registry check</span><span>📄 PDF report</span></div>', unsafe_allow_html=True)
    samples = sorted(os.listdir("samples")) if os.path.isdir("samples") else []
    if samples:
        st.caption("No file handy? Try a sample:")
        sc = st.columns(len(samples))
        for col_, s in zip(sc, samples):
            if col_.button(s.split(".")[0][2:].replace("_", " ").capitalize(), key=s, width="stretch"): st.session_state.sample = s
    with st.expander("Optional: details printed on the document (for the QR registry match)"):
        d1, d2 = st.columns(2); name_in = d1.text_input("Name"); marks_in = d2.text_input("Marks")
        st.caption("Registry example: DV-1001 → Aarav Sharma, marks 86")

    sample = st.session_state.sample
    src = up if up else (open(f"samples/{sample}", "rb") if sample else None)
    if src:
        fname = up.name if up else sample
        with st.spinner("Analysing document…"):
            img = Image.open(src).convert("RGB")
            x, p = core.predict(img, bundle); vd, lvl = verdict(p)
            overlay, marked, boxes, _ = core.heatmap_and_regions(img)
            payload = core.read_qr(img); qr_status, det = core.qr_check(payload, name_in, marks_in)
            notes = core.explain(x, bundle, boxes, np.asarray(img).shape)
        qr_ok = "MATCH" in qr_status and "MISMATCH" not in qr_status
        qr_bad = bool(payload) and not qr_ok
        total = 2 if (lvl == 2 or qr_bad) else 1 if lvl == 1 else 0
        icon, color, title, sub = [("✓", "#16a34a", "No signs of tampering found", "The image analysis and registry checks found nothing suspicious."),
                                   ("!", "#d97706", "Manual review recommended", "Some irregularities were detected. Please inspect the flagged areas."),
                                   ("✕", "#dc2626", "Verification failed", "Signs of tampering or a registry problem were detected.")][total]
        st.markdown(f'<div class="res"><div class="ico" style="background:{color}">{icon}</div><h2>{title}</h2><div class="sub">{sub}</div>'
                    + row("Image integrity analysis", f"{vd} · {p*100:.0f}%", ["ok", "mid", "bad"][lvl])
                    + row("Suspicious regions", f"{len(boxes)} found", "ok" if not boxes else "mid")
                    + row("QR registry check", "Verified" if qr_ok else "Failed" if qr_bad else "No QR code", "ok" if qr_ok else "bad" if qr_bad else "na")
                    + row("File", fname, "na") + '</div>', unsafe_allow_html=True)

        st.write("")
        t1, t2, t3, t4 = st.tabs(["Evidence", "Explanation", "QR record", "Report"])
        with t1:
            view = st.radio("Show", ["All three", "Original", "Heatmap", "Flagged regions"], horizontal=True)
            flagged = marked if show_boxes else np.asarray(img)
            if view == "All three":
                v = st.columns(3)
                v[0].image(img, caption="Original", width="stretch"); v[1].image(overlay, caption="Heatmap", width="stretch"); v[2].image(flagged, caption="Flagged", width="stretch")
            else:
                st.image({"Original": np.asarray(img), "Heatmap": overlay, "Flagged regions": flagged}[view], width=640)
        with t2:
            for n in notes: st.write("• " + n)
            st.caption("Flagged regions are leads for manual checking, not proof of forgery.")
        with t3:
            st.write(f"**{qr_status}**")
            if payload: st.caption(f"QR content: {payload}")
            if det: st.dataframe(pd.DataFrame([{"Field": a_, "On document": b_, "Registered": c_, "Match": "✅" if m_ else "❌"} for a_, b_, c_, m_ in det]), hide_index=True, width="stretch")
            elif payload: st.caption("Enter name / marks in the optional details above to compare with the registry.")
        with t4:
            if st.button("Save & create PDF report", type="primary"):
                vid, ts = core.log(fname, p, vd, qr_status)
                st.success(f"Saved as {vid}")
                st.download_button("⬇️ Download PDF", core.pdf_report(vid, ts, fname, p, vd, qr_status, det, notes, marked, overlay), f"DocVerify_{vid}.pdf", "application/pdf")
    with st.expander("⚙️ Sensitivity settings"):
        st.session_state.rev_t = st.slider("Review above", 0.10, 0.90, st.session_state.rev_t, 0.05)
        st.session_state.forged_t = st.slider("Forged above", 0.20, 0.95, st.session_state.forged_t, 0.05)
        st.session_state.boxes = st.toggle("Draw flagged regions", st.session_state.boxes)

# ---------------- PERFORMANCE ----------------
elif page == "Performance":
    st.header("Model performance")
    st.caption(f"{metrics['n']} images from `{metrics['source']}` · 25% held out for testing · 5-fold cross-validation")
    r = metrics["report"]["forged"]
    k = st.columns(5)
    k[0].metric("Test accuracy", f"{metrics['accuracy']*100:.1f}%")
    k[1].metric("CV accuracy", f"{metrics.get('cv_mean', 0)*100:.1f}%")
    k[2].metric("Precision", f"{r['precision']:.2f}"); k[3].metric("Recall", f"{r['recall']:.2f}"); k[4].metric("F1-score", f"{r['f1-score']:.2f}")
    a, b = st.columns(2)
    with a:
        st.subheader("Confusion matrix")
        st.dataframe(pd.DataFrame(metrics["cm"], index=["Actual genuine", "Actual forged"], columns=["Predicted genuine", "Predicted forged"]), width="stretch")
    with b:
        st.subheader("Most important features")
        st.bar_chart(pd.Series(metrics["importance"]).sort_values(), horizontal=True)
    st.warning("Results on synthetic data are optimistic. Test on real documents before claiming real-world reliability.")

# ---------------- HISTORY ----------------
elif page == "History":
    st.header("Verification history")
    c = core.db(); df = pd.read_sql("SELECT * FROM history ORDER BY ts DESC", c); c.close()
    if len(df):
        st.dataframe(df.rename(columns={"vid": "ID", "ts": "Time", "filename": "File", "prob": "Probability", "verdict": "Verdict", "qr_status": "QR", "model": "Model"}),
                     hide_index=True, width="stretch", column_config={"Probability": st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=1)})
    else:
        st.info("Nothing saved yet. Verify a document and press “Save & create report”.")
    with st.expander("Trusted records registry"):
        c = core.db(); st.dataframe(pd.read_sql("SELECT * FROM records", c), hide_index=True, width="stretch"); c.close()

# ---------------- ABOUT ----------------
else:
    st.header("How it works")
    for i, (t, d) in enumerate([("Upload", "The document image is loaded."),
                                ("Error Level Analysis", "The image is re-saved as JPEG; edited areas compress differently."),
                                ("Noise & sharpness", "Each block is compared with the rest of the page."),
                                ("Classify", f"{len(core.FEATURES)} statistics go into an ensemble of three models."),
                                ("Highlight & explain", "Suspicious blocks are boxed and explained in plain words."),
                                ("QR check", "The QR code is looked up in a trusted database. This is a database lookup, not AI."),
                                ("Report", "A PDF report and a history entry are saved.")], 1):
        st.markdown(f"**{i}. {t}** — {d}")
    st.caption("DocVerify is a preliminary screening tool. Its output is never proof of forgery or authenticity.")