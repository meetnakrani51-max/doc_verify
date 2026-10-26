import os, json, joblib
import numpy as np, pandas as pd, streamlit as st
from PIL import Image
import core

st.set_page_config(page_title="DocVerify", page_icon="🛡️", layout="wide")
st.markdown("""<style>
.hero{background:linear-gradient(120deg,#1e3c78,#3b82f6);color:#fff;padding:22px 28px;border-radius:14px;margin-bottom:14px}
.hero h1{margin:0;font-size:2rem}.hero p{margin:4px 0 0;opacity:.9}
.badge{padding:14px;border-radius:12px;text-align:center;color:#fff;font-size:1.3rem;font-weight:700}
.green{background:#16a34a}.orange{background:#ea8a0c}.red{background:#dc2626}
</style>
<div class="hero"><h1>🛡️ DocVerify</h1><p>AI-based document authentication &amp; forgery detection — ELA + noise analysis + Random Forest + QR record verification</p></div>""",
            unsafe_allow_html=True)

if not os.path.exists("model.pkl"):
    st.error("Model not found. Run `python train.py` first."); st.stop()
core.seed_records()
bundle = joblib.load("model.pkl"); metrics = json.load(open("metrics.json"))

tab1, tab2, tab3, tab4 = st.tabs(["🔍 Verify Document", "📊 Model Performance", "🗂️ History", "ℹ️ How it works"])

with tab1:
    c1, c2 = st.columns([1, 1])
    with c1:
        up = st.file_uploader("Upload a document image (JPG/PNG)", type=["jpg", "jpeg", "png"])
        samples = sorted(os.listdir("samples")) if os.path.isdir("samples") else []
        demo = st.selectbox("…or pick a demo sample", ["—"] + samples)
    with c2:
        st.markdown("**Details printed on the document** (for QR record matching)")
        name_in = st.text_input("Name on document"); marks_in = st.text_input("Marks on document")
        st.caption("Demo: DV-1001 is registered as *Aarav Sharma*, marks *86*.")
    src = up if up else (open(f"samples/{demo}", "rb") if demo != "—" else None)
    if src:
        fname = up.name if up else demo
        img = Image.open(src).convert("RGB")
        x, p = core.predict(img, bundle); vd, col = core.verdict(p)
        overlay, marked, boxes, _ = core.heatmap_and_regions(img)
        payload = core.read_qr(img); qr_status, det = core.qr_check(payload, name_in, marks_in)
        notes = core.explain(x, bundle, boxes, np.asarray(img).shape)

        st.divider()
        m1, m2, m3 = st.columns([1.2, 1, 1.4])
        m1.markdown(f'<div class="badge {col}">{vd}</div>', unsafe_allow_html=True)
        m2.metric("Forgery probability", f"{p*100:.1f}%")
        m2.progress(min(p, 1.0))
        m3.markdown("**QR verification**"); m3.write(f"{'✅' if 'MATCH' in qr_status and 'MISMATCH' not in qr_status else '⚠️'} {qr_status}")
        if payload: m3.caption(f"QR payload: `{payload}`")

        i1, i2, i3 = st.columns(3)
        i1.image(img, caption="Original", use_container_width=True)
        i2.image(overlay, caption="ELA heatmap (red = unusual compression)", use_container_width=True)
        i3.image(marked, caption=f"Flagged regions ({len(boxes)})", use_container_width=True)

        if det:
            st.subheader("QR record comparison")
            st.dataframe(pd.DataFrame([{"Field": a, "On document": b, "Registered": c, "Match": "✅" if m else "❌"} for a, b, c, m in det]),
                         hide_index=True, use_container_width=True)
        st.subheader("Explanation")
        for n in notes: st.write("• " + n)
        st.caption("Flagged regions are leads for manual inspection, not proof of forgery.")

        if st.button("📄 Save to history & generate PDF report", type="primary"):
            vid, ts = core.log(fname, p, vd, qr_status)
            pdf = core.pdf_report(vid, ts, fname, p, vd, qr_status, det, notes, marked, overlay)
            st.success(f"Logged as {vid}")
            st.download_button("⬇️ Download PDF report", pdf, f"DocVerify_{vid}.pdf", "application/pdf")
    else:
        st.info("Upload a document or choose a demo sample to begin. Try the three samples in order for a quick demo.")

with tab2:
    st.caption(f"Trained on {metrics['n']} images from `{metrics['source']}` · model {core.MODEL_VERSION} · 25% held-out test set")
    a, b, c, d = st.columns(4); r = metrics["report"]
    a.metric("Test accuracy", f"{metrics['accuracy']*100:.1f}%", f"CV {metrics.get('cv_mean',0)*100:.1f}% ± {metrics.get('cv_std',0)*100:.1f}", delta_color="off"); b.metric("Precision (forged)", f"{r['forged']['precision']:.2f}")
    c.metric("Recall (forged)", f"{r['forged']['recall']:.2f}"); d.metric("F1 (forged)", f"{r['forged']['f1-score']:.2f}")
    l, rr = st.columns(2)
    cm = metrics["cm"]
    l.markdown("**Confusion matrix**"); l.dataframe(pd.DataFrame(cm, index=["True genuine", "True forged"], columns=["Pred genuine", "Pred forged"]))
    rr.markdown("**Top 12 features by importance**"); rr.bar_chart(pd.Series(metrics["importance"]).sort_values())
    st.warning("Performance on a synthetic dataset is optimistic. Re-test on diverse real documents before claiming real-world reliability.")

with tab3:
    c = core.db(); df = pd.read_sql("SELECT * FROM history ORDER BY ts DESC", c); c.close()
    st.dataframe(df, hide_index=True, use_container_width=True) if len(df) else st.info("No verifications saved yet.")
    with st.expander("Trusted records database"):
        c = core.db(); st.dataframe(pd.read_sql("SELECT * FROM records", c), hide_index=True); c.close()

with tab4:
    st.markdown("""
1. **Upload** → image is loaded and normalised.
2. **ELA** re-saves the image as JPEG (q=90) and measures the difference; edited areas often compress differently.
3. **Noise residual** (image − median filter) and **Laplacian sharpness** are compared block-by-block.
4. **11 statistical features** → **Random Forest** → forgery probability.
5. **Heatmap + boxes** show where the error is concentrated.
6. **Explanations** compare each feature with the genuine-document baseline learned at training time.
7. **QR check** looks the code up in a trusted SQLite database and compares name/marks (this is a database lookup, not an AI prediction).
8. **PDF report + history** give an audit trail.
""")