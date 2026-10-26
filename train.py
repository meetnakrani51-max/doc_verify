"""Train the model. Uses dataset/genuine + dataset/forged if present, else builds a synthetic dataset.
Run:  python train.py"""
import os, glob, random, json, joblib
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import core

NAMES = ["Aarav Sharma", "Diya Patel", "Rohan Mehta", "Sara Khan", "Ishaan Rao", "Meera Nair", "Kabir Singh", "Anaya Iyer"]
COURSES = ["B.Tech CSE", "B.Tech ECE", "B.Sc Physics", "BBA", "B.Com", "MCA"]


def font(sz):
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "DejaVuSans-Bold.ttf", "arial.ttf"]:
        try: return ImageFont.truetype(p, sz)
        except Exception: pass
    return ImageFont.load_default()


def render(name, course, marks, doc_id, rng):
    im = Image.new("RGB", (640, 440), (250, 248, 240)); d = ImageDraw.Draw(im)
    d.rectangle([12, 12, 628, 428], outline=(30, 60, 120), width=4)
    d.text((320, 50), "STATE UNIVERSITY", font=font(30), fill=(30, 60, 120), anchor="mm")
    d.text((320, 90), "Certificate of Achievement", font=font(18), fill=(90, 90, 90), anchor="mm")
    for i, (k, v) in enumerate([("Name", name), ("Course", course), ("Marks", marks), ("Doc ID", doc_id)]):
        d.text((50, 150 + i * 50), f"{k}:", font=font(20), fill=(40, 40, 40))
        d.text((170, 150 + i * 50), v, font=font(20), fill=(10, 10, 10))
    im.paste(core.make_qr(doc_id, 120), (480, 290))
    arr = np.asarray(im).astype(np.float32) + rng.normal(0, rng.uniform(1, 3), (440, 640, 3))
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def forge(im, rng):
    """Edit a field (cover + retype) or paste a patch from elsewhere, then re-save at a different quality."""
    im = im.copy(); d = ImageDraw.Draw(im)
    if rng.random() < 0.6:
        y = int(rng.choice([200, 250])); d.rectangle([165, y - 4, 400, y + 30], fill=(250, 248, 240))
        d.text((170, y), str(int(rng.integers(55, 99))), font=font(20 + int(rng.integers(-2, 3))), fill=(10, 10, 10))
    else:
        x, y = int(rng.integers(40, 300)), int(rng.integers(140, 330))
        im.paste(im.crop((x, y, x + 150, y + 36)).resize((150, 36)), (x + int(rng.integers(5, 40)), y + int(rng.integers(-30, 30))))
    return im


def jpg(im, q, path):
    im.save(path, "JPEG", quality=int(q)); return Image.open(path).convert("RGB")


def synth(n=500, out="dataset_synth"):
    rng = np.random.default_rng(7); random.seed(7)
    for s in ("genuine", "forged"): os.makedirs(f"{out}/{s}", exist_ok=True)
    for i in range(n):
        base = render(random.choice(NAMES), random.choice(COURSES), str(random.randint(55, 99)), f"DV-{1000+i}", rng)
        jpg(base, rng.integers(82, 96), f"{out}/genuine/g{i}.jpg")
        first = jpg(base, rng.integers(70, 96), "/tmp/_t.jpg")
        jpg(forge(first, rng), rng.integers(82, 96), f"{out}/forged/f{i}.jpg")
    return out


def make_samples():
    """Demo files for the presentation: genuine, forged (marks edited), forged (QR not registered)."""
    os.makedirs("samples", exist_ok=True); rng = np.random.default_rng(99)
    g = render("Aarav Sharma", "B.Tech CSE", "86", "DV-1001", rng)
    jpg(g, 90, "samples/1_genuine_certificate.jpg")
    f = forge(jpg(g, 85, "/tmp/_t.jpg"), rng); d = ImageDraw.Draw(f)
    d.rectangle([165, 196, 400, 232], fill=(250, 248, 240)); d.text((170, 200), "99", font=font(20), fill=(10, 10, 10))
    jpg(f, 90, "samples/2_forged_marks_edited.jpg")
    h = render("Rohan Mehta", "MCA", "88", "DV-9999", rng); jpg(h, 90, "samples/3_unregistered_qr.jpg")


def forge_generic(im, rng):
    """Tamper any document image: copy-move a patch, or cover a field and retype digits."""
    im = im.copy(); W, H = im.size; d = ImageDraw.Draw(im)
    w, h = int(W * rng.uniform(.10, .25)), int(H * rng.uniform(.05, .10))
    x, y = int(rng.integers(0, W - w)), int(rng.integers(0, H - h))
    if rng.random() < 0.5:
        sx, sy = int(rng.integers(0, W - w)), int(rng.integers(0, H - h))
        patch = im.crop((sx, sy, sx + w, sy + h)).resize((int(w * rng.uniform(.95, 1.05)), h))
        im.paste(patch, (x, y))
    else:
        col = tuple(int(v) for v in np.asarray(im.crop((x, y, x + w, y + h))).reshape(-1, 3).mean(0))
        d.rectangle([x, y, x + w, y + h], fill=col)
        d.text((x + 3, y + 2), str(int(rng.integers(10**5, 10**8))), font=font(max(10, h - 6)), fill=(20, 20, 20))
    return im


def auto_forge(src="dataset/genuine", out="dataset_auto", per_image=2):
    """Real genuine images -> genuine/forged classes. Both go through a prior JPEG pass so the model
    learns local inconsistency, not just 'was re-saved'."""
    rng = np.random.default_rng(5)
    for s in ("genuine", "forged"): os.makedirs(f"{out}/{s}", exist_ok=True)
    for i, p in enumerate(sorted(glob.glob(f"{src}/*"))):
        try: base = Image.open(p).convert("RGB")
        except Exception: continue
        for k in range(per_image):
            pre = jpg(base, rng.integers(70, 96), "/tmp/_t.jpg")
            jpg(pre, rng.integers(82, 96), f"{out}/genuine/g{i}_{k}.jpg")
            jpg(forge_generic(pre, rng), rng.integers(82, 96), f"{out}/forged/f{i}_{k}.jpg")
    return out


if __name__ == "__main__":
    g_, f_ = glob.glob("dataset/genuine/*"), glob.glob("dataset/forged/*")
    root = "dataset" if g_ and f_ else auto_forge() if g_ else synth()
    print("Using dataset:", root)
    X, y = [], []
    for lbl, sub in enumerate(["genuine", "forged"]):
        for p in glob.glob(f"{root}/{sub}/*"):
            try: X.append(core.features(Image.open(p).convert("RGB"))); y.append(lbl)
            except Exception as e: print("skip", p, e)
    X, y = np.array(X), np.array(y)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=1)
    mk = lambda: VotingClassifier([("rf", RandomForestClassifier(400, random_state=1, class_weight="balanced")),
                                   ("et", ExtraTreesClassifier(500, random_state=1, class_weight="balanced")),
                                   ("hgb", HistGradientBoostingClassifier(random_state=1))], voting="soft")
    m = mk().fit(Xtr, ytr)
    pr = m.predict(Xte); rep = classification_report(yte, pr, target_names=["genuine", "forged"], output_dict=True)
    cm = confusion_matrix(yte, pr).tolist(); print(classification_report(yte, pr), cm)
    cvs = cross_val_score(mk(), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0))
    print(f"5-fold CV accuracy: {cvs.mean():.3f} +/- {cvs.std():.3f}")
    final = mk().fit(X, y)  # deployed model uses all data
    rf = RandomForestClassifier(400, random_state=1).fit(X, y); g = X[y == 0]
    joblib.dump(dict(model=final, mu=g.mean(0), sd=g.std(0), imp=rf.feature_importances_), "model.pkl")
    json.dump(dict(accuracy=accuracy_score(yte, pr), report=rep, cm=cm, n=len(y), source=root, cv_mean=cvs.mean(), cv_std=cvs.std(),
                   importance=dict(sorted(zip(core.FEATURES, rf.feature_importances_.tolist()), key=lambda kv: -kv[1])[:12])),
              open("metrics.json", "w"))
    make_samples(); core.seed_records(); print("Saved model.pkl, metrics.json, samples/")