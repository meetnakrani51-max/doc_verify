"""DocVerify core: ELA, noise residual, features, region detection, QR, DB, PDF report."""
import io, sqlite3, uuid, datetime
import numpy as np, cv2
from PIL import Image
from scipy import ndimage, stats

DB = "docverify.db"
MODEL_VERSION = "Ensemble-v2"
BASE = ["ela_mean", "ela_std", "ela_p99", "ela_max", "ela_block_cv", "ela_block_peak",
            "noise_var", "noise_kurt", "noise_block_cv", "sharp_block_cv", "lap_var"]
EXTRA = [f"ela{q}_{s}" for q in (70, 80, 90, 95) for s in ("zmax", "z5", "outfrac", "cv", "p99med")] + \
        [f"{k}_{s}" for k in ("noise", "sharp") for s in ("zmax", "z5", "outfrac")] + ["grid_ratio", "loc_p995", "loc_max"]
FEATURES = BASE + EXTRA
FRIENDLY = {
    "ela_mean": "overall compression-error level", "ela_std": "spread of compression error",
    "ela_p99": "strongest compression-error pixels", "ela_max": "peak compression error",
    "ela_block_cv": "uneven compression error across the page",
    "ela_block_peak": "a local area whose compression error stands out from the rest",
    "noise_var": "overall noise level", "noise_kurt": "irregular (heavy-tailed) noise distribution",
    "noise_block_cv": "inconsistent noise between regions", "sharp_block_cv": "inconsistent sharpness between regions",
    "lap_var": "overall sharpness"}
for _q in (70, 80, 90, 95):
    FRIENDLY.update({f"ela{_q}_zmax": f"a local area with extreme compression error (q={_q})", f"ela{_q}_z5": f"several areas with unusual compression error (q={_q})",
                     f"ela{_q}_outfrac": f"share of outlier blocks in compression error (q={_q})", f"ela{_q}_cv": f"uneven compression error (q={_q})",
                     f"ela{_q}_p99med": f"brightest compression-error areas vs the page median (q={_q})"})
FRIENDLY.update(noise_zmax="a block with outlier noise", noise_z5="several blocks with outlier noise", noise_outfrac="share of blocks with outlier noise",
                sharp_zmax="a block with outlier sharpness", sharp_z5="several blocks with outlier sharpness", sharp_outfrac="share of blocks with outlier sharpness",
                grid_ratio="JPEG 8x8 block-grid irregularity", loc_p995="sharp local jumps in compression error", loc_max="largest local jump in compression error")
B = 16  # block size


def ela_map(img, q=90):
    buf = io.BytesIO(); img.convert("RGB").save(buf, "JPEG", quality=q); buf.seek(0)
    re = Image.open(buf).convert("RGB")
    d = np.abs(np.asarray(img.convert("RGB"), np.float32) - np.asarray(re, np.float32))
    return d.mean(axis=2)


def noise_residual(gray):
    return gray - cv2.medianBlur(gray.astype(np.uint8), 3).astype(np.float32)


def _z(b):
    m = np.median(b); return (b - m) / (1.4826 * np.median(np.abs(b - m)) + 1e-6)


def _blocks(a, fn):
    h, w = (a.shape[0] // B) * B, (a.shape[1] // B) * B
    t = a[:h, :w].reshape(h // B, B, w // B, B).swapaxes(1, 2).reshape(h // B, w // B, -1)
    return fn(t)


def base_features(img):
    e = ela_map(img); g = np.asarray(img.convert("L"), np.float32); n = noise_residual(g)
    eb = _blocks(e, lambda t: t.mean(-1)); nb = _blocks(n, lambda t: t.var(-1))
    lap = cv2.Laplacian(g, cv2.CV_32F); sb = _blocks(lap, lambda t: t.var(-1))
    cv = lambda x: float(x.std() / (x.mean() + 1e-6))
    f = dict(ela_mean=e.mean(), ela_std=e.std(), ela_p99=np.percentile(e, 99), ela_max=e.max(),
             ela_block_cv=cv(eb), ela_block_peak=float(eb.max() / (np.median(eb) + 1e-6)),
             noise_var=n.var(), noise_kurt=float(stats.kurtosis(n.ravel())), noise_block_cv=cv(nb),
             sharp_block_cv=cv(sb), lap_var=lap.var())
    return np.array([f[k] for k in BASE], np.float32)


def extra_features(img):
    g = np.asarray(img.convert("L"), np.float32); out = []
    for q in (70, 80, 90, 95):
        e = cv2.GaussianBlur(ela_map(img, q), (0, 0), 1.5); eb = _blocks(e, lambda t: t.mean(-1)); z = _z(eb)
        s = np.sort(z.ravel())[::-1]
        out += [s[0], s[:5].mean(), (z > 3).mean(), eb.std() / (eb.mean() + 1e-6), np.percentile(eb, 99) / (np.median(eb) + 1e-6)]
    nb = _blocks(noise_residual(g), lambda t: t.var(-1)); sb = _blocks(cv2.Laplacian(g, cv2.CV_32F), lambda t: t.var(-1))
    for b in (nb, sb):
        z = _z(np.log(b + 1e-3)); s = np.sort(z.ravel())[::-1]; out += [s[0], s[:5].mean(), (z > 3).mean()]
    dx = np.abs(np.diff(g, axis=1)); out.append(dx[:, 7::8].mean() / (dx.mean() + 1e-6))
    e = ela_map(img, 90); r = np.abs(cv2.blur(e, (8, 8)) - cv2.blur(e, (64, 64)))
    out += [np.percentile(r, 99.5), r.max()]
    return np.array(out, np.float32)


def features(img):
    return np.concatenate([base_features(img), extra_features(img)])


def heatmap_and_regions(img, top_pct=97):
    """Return heatmap overlay (RGB ndarray), boxes [(x,y,w,h)] in pixel coords."""
    e = ela_map(img); e = cv2.GaussianBlur(e, (0, 0), 2)
    eb = _blocks(e, lambda t: t.mean(-1))
    z = (eb - np.median(eb)) / (eb.std() + 1e-6)
    mask = (eb >= np.percentile(eb, top_pct)) & (z > 2.0)
    mask = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8))
    n, _, st, _ = cv2.connectedComponentsWithStats(mask)
    boxes = [(int(x * B), int(y * B), int(w * B), int(h * B)) for x, y, w, h, a in st[1:] if a >= 2]
    norm = np.clip(e / (np.percentile(e, 99.5) + 1e-6), 0, 1)
    heat = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)[:, :, ::-1]
    base = np.asarray(img.convert("RGB"))
    over = cv2.addWeighted(base, 0.45, heat, 0.55, 0)
    marked = base.copy()
    for x, y, w, h in boxes: cv2.rectangle(marked, (x, y), (x + w, y + h), (220, 30, 30), 2)
    return over, marked, boxes, (norm * 255).astype(np.uint8)


def explain(x, bundle, boxes, shape):
    """Explanations derived from measured values vs genuine-training baseline."""
    mu, sd, imp = bundle["mu"], bundle["sd"], bundle["imp"]
    z = (x - mu) / (sd + 1e-6); out = []
    for i in np.argsort(-(np.abs(z) * imp))[:4]:
        if abs(z[i]) > 1.5:
            d = "higher" if z[i] > 0 else "lower"
            out.append(f"{FRIENDLY[FEATURES[i]].capitalize()} is {d} than typical genuine documents "
                       f"({z[i]:+.1f} std devs).")
    H, W = shape[:2]
    for (bx, by, bw, bh) in boxes[:3]:
        v = "top" if by + bh / 2 < H / 3 else "bottom" if by + bh / 2 > 2 * H / 3 else "middle"
        hz = "left" if bx + bw / 2 < W / 3 else "right" if bx + bw / 2 > 2 * W / 3 else "centre"
        out.append(f"Region at ({bx},{by}) size {bw}x{bh} ({v}-{hz}) shows compression error that differs "
                   f"from its surroundings - inspect this area manually.")
    return out or ["No strong irregularities found in compression, noise or sharpness."]


def predict(img, bundle):
    x = features(img)
    p = float(bundle["model"].predict_proba([x])[0][1])
    return x, p


def verdict(p):
    return ("Likely Forged", "red") if p >= 0.65 else ("Suspicious - Review", "orange") if p >= 0.4 else ("Likely Genuine", "green")


# ---------- QR ----------
def read_qr(img):
    arr = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    data, _, _ = cv2.QRCodeDetector().detectAndDecode(arr)
    return data or None


def make_qr(text, size=110):
    import qrcode
    return qrcode.make(text).convert("RGB").resize((size, size))


# ---------- DB ----------
def db():
    c = sqlite3.connect(DB); c.row_factory = sqlite3.Row
    c.execute("CREATE TABLE IF NOT EXISTS records(doc_id TEXT PRIMARY KEY,name TEXT,course TEXT,marks TEXT)")
    c.execute("""CREATE TABLE IF NOT EXISTS history(vid TEXT,ts TEXT,filename TEXT,prob REAL,verdict TEXT,
                 qr_status TEXT,model TEXT)""")
    return c


def seed_records():
    c = db()
    rows = [("DV-1001", "Aarav Sharma", "B.Tech CSE", "86"), ("DV-1002", "Diya Patel", "B.Tech ECE", "91"),
            ("DV-1003", "Rohan Mehta", "B.Sc Physics", "78"), ("DV-1004", "Sara Khan", "BBA", "82")]
    c.executemany("INSERT OR IGNORE INTO records VALUES(?,?,?,?)", rows); c.commit(); c.close()


def qr_check(payload, name="", marks=""):
    if not payload: return "No QR code found", []
    c = db(); r = c.execute("SELECT * FROM records WHERE doc_id=?", (payload.strip(),)).fetchone(); c.close()
    if not r: return f"QR '{payload}' NOT registered in trusted database", []
    det = []
    if name: det.append(("Name", name, r["name"], name.strip().lower() == r["name"].lower()))
    if marks: det.append(("Marks", marks, r["marks"], marks.strip() == r["marks"]))
    ok = all(d[3] for d in det)
    return ("Record found - details MATCH" if ok else "Record found - MISMATCH in details"), det


def log(filename, prob, vd, qr):
    vid = "V-" + uuid.uuid4().hex[:8].upper(); ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c = db(); c.execute("INSERT INTO history VALUES(?,?,?,?,?,?,?)", (vid, ts, filename, prob, vd, qr, MODEL_VERSION))
    c.commit(); c.close(); return vid, ts


# ---------- PDF ----------
def pdf_report(vid, ts, filename, prob, vd, qr, det, notes, img_marked, img_heat):
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RI, Table
    from reportlab.lib.styles import getSampleStyleSheet
    s = getSampleStyleSheet(); buf = io.BytesIO()
    d = SimpleDocTemplate(buf, pagesize=A4, title="DocVerify Report")
    def pic(a):
        b = io.BytesIO(); Image.fromarray(a).save(b, "PNG"); b.seek(0)
        return RI(b, width=240, height=240 * a.shape[0] / a.shape[1])
    el = [Paragraph("DocVerify - Verification Report", s["Title"]),
          Table([["Verification ID", vid], ["Timestamp", ts], ["File", filename], ["Model", MODEL_VERSION],
                 ["Verdict", vd], ["Forgery probability", f"{prob*100:.1f}%"], ["QR verification", qr]]),
          Spacer(1, 10)]
    if det:
        el += [Paragraph("QR record comparison", s["Heading3"]),
               Table([["Field", "On document", "Registered", "Match"]] + [[a, b, c, "Yes" if m else "NO"] for a, b, c, m in det])]
    el += [Paragraph("Analysis observations", s["Heading3"])] + [Paragraph("- " + n, s["BodyText"]) for n in notes]
    el += [Spacer(1, 8), Table([[pic(img_marked), pic(img_heat)]]), Paragraph("Left: flagged regions. Right: ELA heatmap.", s["Italic"]),
           Spacer(1, 8), Paragraph("Recommendation: this is a preliminary automated screening, not proof of forgery or authenticity. "
                                   "Confirm flagged documents with the issuing institution.", s["BodyText"])]
    d.build(el); return buf.getvalue()
