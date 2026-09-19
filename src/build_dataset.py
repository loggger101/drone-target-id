#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Universal KaggleHub dataset builder with:
  • Drop-in dataset configs (Kaggle slugs)
  • Per-dataset label extraction strategies (folder/CSV/regex)
  • Global label aliasing to merge overlaps across datasets
  • Balanced splits, de-duplication, progress bar, metadata, optional zip
  • Notebook/Colab safe (no required flags; ignores Jupyter -f)

Usage (Colab/Notebook):
  pip install kagglehub pillow tqdm pandas
  # then run this cell/file; defaults will build a small sample

Usage (CLI):
  python src/build_dataset.py --out data_kagglehub_unified --images-per-class 1000
"""

import argparse, io, os, sys, json, csv, hashlib, shutil, random, re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterable
from PIL import Image
from tqdm import tqdm

try:
    import pandas as pd
except Exception:
    pd = None

try:
    import kagglehub
except Exception:
    print("This script needs kagglehub. Install with: pip install kagglehub", file=sys.stderr)
    raise

# ==============================================================================
# 0) GLOBAL LABELS & ALIASES (edit me)
#    - Put your canonical labels here (e.g., 'bird', 'cloud', 'building', 'airplane', etc.)
#    - Any dataset-specific label that appears as a key in ALIAS_TO_CANONICAL is normalized to the canonical.
#      If a label isn't listed, it will pass through as-is (lowercased, spaces->underscores).
# ==============================================================================

CANONICAL_LABELS: List[str] = [
    "bird",
    "cloud",
    "aerial_landscape",
    "building",
    "mountain",
    "airplane",

]

ALIAS_TO_CANONICAL: Dict[str, str] = {
    # birds
    "birds": "bird",
    "avian": "bird",
    # clouds
    "clouds": "cloud",
    "cloudscape": "cloud",
    # aerial landscapes
    "skyview": "aerial_landscape",
    "aerial": "aerial_landscape",
    "aerial_scene": "aerial_landscape",
    "overhead_landscape": "aerial_landscape",
    # buildings
    "buildings": "building",
    "skyscraper": "building",
    "rooftop": "building",
    # mountains
    "mountains": "mountain",
    "peak": "mountain",
    "ridge": "mountain",
    # airplanes
    "aircraft": "airplane",
    "airliner": "airplane",
    "plane": "airplane",
}

VALID_FINE: set = {'soldier','tank_av','flying_target'
}
INVALID_NONTARGET_FINE: set = {'civilian','civilian_vehicle'
}
INVALID_BACKGROUND_FINE: set = {'aerial_landscape', 'cloud_blanksky','tree_shrub','rock_debris','random_animal','bird'
}


def map_validity(fine_label: str) -> str:
    """Return 'valid', 'invalid', or 'unknown' for a given fine class."""
    if fine_label in VALID_FINE:
        return "valid"
    if fine_label in INVALID_NONTARGET_FINE:
        return "invalid_nontarget"
    if fine_label in INVALID_BACKGROUND_FINE:
        return "invalid_background"
    return "unknown"

# ==============================================================================
# 1) DATASET REGISTRY (edit me)
#    Add a new block to wire in a dataset.
#    Choose ONE of these label strategies:
#       - "folder": infer label from parent folder name(s)
#       - "csv":    read a CSV/TSV with image path & label columns
#       - "regex":  extract label with a regex that has a (?P<label>...) group
#
#    For overlapping labels across datasets, the alias map above merges to the same canonical label.
# ==============================================================================

DATASETS: List[dict] = [
{
         "name": "MASTER",
         "slug": "loggger/fpv-images",
         "enabled": True,
         "strategy": "folder",
         "params": {
              "root_glob": "*",
              "label_depth": 1,
              "exts": [".jpg",".jpeg",".png",".bmp",".webp",".tif",".tiff"],
              "local_rename": {},
         },
         "allow_labels": ['soldier','tank_av','flying_target', 'civilian','civilian_vehicle', 'aerial_landscape', 'cloud_blanksky','tree_shrub','rock_debris','random_animal','bird'],
         ###"default_label": "",
    },

    # TEMPLATE: CSV-based dataset (uncomment + edit to use)
    # {
    #     "name": "my_csv_dataset",
    #     "slug": "author/dataset-slug",
    #     "enabled": False,
    #     "strategy": "csv",
    #     "params": {
    #         "csv_relpath": "labels.csv",
    #         "image_col": "filepath",
    #         "label_col": "label",
    #         "sep": ",",    # or "\t"
    #         "local_rename": {"Clouds":"cloud","Mountains":"mountain"},
    #     },
    #     "allow_labels": None,       # or e.g., ["cloud","mountain"]
    # },

    # TEMPLATE: Regex-based labels from filenames
    # {
    #     "name":"my_regex_dataset",
    #     "slug":"author/slug",
    #     "enabled": False,
    #     "strategy":"regex",
    #     "params":{
    #         "glob":"**/*.jpg",
    #         "pattern": r".*/(?P<label>[A-Za-z_]+)/[^/]+\.jpg$",
    #     },
    #     "allow_labels": None,
    # },
]

# ==============================================================================
# 2) PIPELINE SETTINGS (CLI overridable)
# ==============================================================================

DEFAULT_SIZE = 512
DEFAULT_JPG_QUALITY = 512
RNG = random.Random(1337)

# ==============================================================================
# 3) CORE HELPERS
# ==============================================================================

def normalize_label(lbl: str) -> str:
    """Lowercase, strip, replace spaces with underscores, then alias to canonical if present."""
    if lbl is None:
        return ""
    s = lbl.strip().lower().replace(" ", "_")
    return ALIAS_TO_CANONICAL.get(s, s)

def canonical_ok(lbl: str, allowed: Optional[List[str]]) -> bool:
    if not lbl: return False
    if allowed is None: return True
    return lbl in set(allowed)

def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256(); h.update(b); return h.hexdigest()

def uid_from_bytes(b: bytes, salt: str="") -> str:
    return sha256_bytes(b + salt.encode("utf-8"))[:16]

def resize_to_jpeg(p: Path, size: int, quality: int) -> Optional[bytes]:
    try:
        im = Image.open(p).convert("RGB")
        if im.size != (size, size):
            im = im.resize((size, size), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=quality, optimize=True)
        return buf.getvalue()
    except Exception:
        return None

def choose_split(r: float, splits: Tuple[float,float,float]) -> str:
    tr, va, te = splits
    return "train" if r < tr else ("val" if r < tr + va else "test")

# ==============================================================================
# 4) DATASET STRATEGIES
# ==============================================================================

@dataclass
class LabeledPath:
    path: Path
    label: str
    provider: str

def scan_folder(root: Path, label_depth: int, exts: List[str],
                local_rename: Dict[str,str], default_label: Optional[str],
                allow_labels: Optional[List[str]], provider_name: str) -> Iterable[LabeledPath]:
    exts = set(x.lower() for x in exts)
    for p in root.rglob("*"):
        if not p.is_file(): continue
        if p.suffix.lower() not in exts: continue
        if default_label:
            lbl = normalize_label(default_label)
        else:
            # derive from parent at given depth
            cur = p.parent
            for _ in range(label_depth-1):
                cur = cur.parent
            lbl = normalize_label(local_rename.get(cur.name, cur.name))
        if canonical_ok(lbl, allow_labels):
            yield LabeledPath(p, lbl, provider_name)

def scan_csv(root: Path, csv_relpath: str, image_col: str, label_col: str, sep: str,
             local_rename: Dict[str,str], allow_labels: Optional[List[str]],
             provider_name: str) -> Iterable[LabeledPath]:
    if pd is None:
        raise RuntimeError("pandas required for CSV strategy. pip install pandas")
    csv_path = root / csv_relpath
    if not csv_path.exists():
        # try to find it somewhere beneath root
        found = list(root.rglob(Path(csv_relpath).name))
        if not found:
            raise FileNotFoundError(f"Could not find CSV '{csv_relpath}' under {root}")
        csv_path = found[0]
    df = pd.read_csv(csv_path, sep=sep)
    if image_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"CSV must have columns '{image_col}' and '{label_col}'")
    for _, row in df.iterrows():
        rel = str(row[image_col])
        lbl_local = str(row[label_col])
        p = (root / rel).resolve()
        if not p.exists():  # try locate by name
            candidates = list(root.rglob(Path(rel).name))
            if not candidates: continue
            p = candidates[0]
        lbl = normalize_label(local_rename.get(lbl_local, lbl_local))
        if canonical_ok(lbl, allow_labels):
            yield LabeledPath(p, lbl, provider_name)

def scan_regex(root: Path, glob_pat: str, regex_pat: str,
               allow_labels: Optional[List[str]], provider_name: str) -> Iterable[LabeledPath]:
    rx = re.compile(regex_pat)
    for p in root.glob(glob_pat):
        if not p.is_file(): continue
        m = rx.match(str(p))
        if not m or "label" not in m.groupdict(): continue
        lbl = normalize_label(m.group("label"))
        if canonical_ok(lbl, allow_labels):
            yield LabeledPath(p, lbl, provider_name)

# ==============================================================================
# 5) BUILD PIPELINE
# ==============================================================================

def build_dataset(
    out_dir: Path,
    size: int,
    images_per_class: int,
    splits: Tuple[float,float,float],
    seed: int,
    jpg_quality: int,
    zip_out: bool,
    only_classes: Optional[List[str]] = None,
):
    # Normalize splits
    ssum = sum(splits); splits = (splits[0]/ssum, splits[1]/ssum, splits[2]/ssum)
    ensure_dir(out_dir)
    rng = random.Random(seed)

    # 5.1 Download + collect labeled paths
    class_to_paths: Dict[str, List[LabeledPath]] = {}
    for ds in DATASETS:
        if not ds.get("enabled", True): continue

        # Download via kagglehub
        slug = ds["slug"]
        name = ds["name"]
        print(f"[DL] {name} ({slug})")
        root_path = Path(kagglehub.dataset_download(slug))
        strat = ds["strategy"]
        params = ds.get("params", {})
        allow = ds.get("allow_labels", None)
        default_label = ds.get("default_label", None)

        # Scan using the chosen strategy
        labeled_iter: Iterable[LabeledPath]
        if strat == "folder":
            labeled_iter = scan_folder(
                root=root_path,
                label_depth=params.get("label_depth", 1),
                exts=params.get("exts", [".jpg",".jpeg",".png",".bmp",".webp",".tif",".tiff"]),
                local_rename=params.get("local_rename", {}),
                default_label=default_label,
                allow_labels=allow,
                provider_name=name,
            )
        elif strat == "csv":
            labeled_iter = scan_csv(
                root=root_path,
                csv_relpath=params["csv_relpath"],
                image_col=params["image_col"],
                label_col=params["label_col"],
                sep=params.get("sep", ","),
                local_rename=params.get("local_rename", {}),
                allow_labels=allow,
                provider_name=name,
            )
        elif strat == "regex":
            labeled_iter = scan_regex(
                root=root_path,
                glob_pat=params.get("glob", "**/*"),
                regex_pat=params["pattern"],
                allow_labels=allow,
                provider_name=name,
            )
        else:
            raise ValueError(f"Unknown strategy: {strat}")

        # Accumulate per class (after global aliasing inside scanners)
        added=0
        for lp in labeled_iter:
            if only_classes and lp.label not in only_classes:
                continue
            class_to_paths.setdefault(lp.label, []).append(lp)
            added+=1
        print(f"     + collected {added} labeled paths")

    if not class_to_paths:
        raise RuntimeError("No images discovered. Check DATASETS config and paths.")

    # 5.2 Prepare output dirs (nest by coarse validity)
    all_labels = sorted(class_to_paths.keys())
    print(f"[INFO] Classes discovered: {all_labels}")

    out_img_root = out_dir / "images"
    coarse_buckets = {"valid","invalid","unknown"}  # ensure dirs exist even if unused yet
    for sp in ("train","val","test"):
        for lbl in all_labels:
            coarse = map_validity(lbl)
            if coarse not in coarse_buckets:
                coarse_buckets.add(coarse)
            ensure_dir(out_img_root / sp / coarse / lbl)

    # 5.3 Save with de-duplication and balanced per-class cap
    rows: List[dict] = []
    total_target = len(all_labels) * images_per_class
    seen_hashes: set = set()
    pbar = tqdm(total=total_target, desc="Saving", unit="img")

    for lbl in all_labels:
        pool = class_to_paths[lbl]
        rng.shuffle(pool)
        saved = 0; i = 0
        while saved < images_per_class and i < len(pool):
            src = pool[i].path
            prov = pool[i].provider
            i += 1
            data = resize_to_jpeg(src, size=size, quality=jpg_quality)
            if data is None:
                continue
            sig = sha256_bytes(data)
            if sig in seen_hashes:
                continue
            seen_hashes.add(sig)

            uid = uid_from_bytes(data, salt=f"{lbl}:{saved}")
            split = choose_split(rng.random(), splits)
            coarse = map_validity(lbl)  # NEW
            rel_dir = Path("images") / split / coarse / lbl  # NEW
            outp = out_dir / rel_dir / f"{uid}.jpg"
            with open(outp, "wb") as f:
              f.write(data)

            rows.append({
              "uid": uid,
              "split": split,
              "fine_label": lbl,
              "coarse_label": coarse,            # NEW
              "width": size,
              "height": size,
              "out_relpath": str(rel_dir / f"{uid}.jpg"),
              "sha256": sig,
              "source_path": str(src),
              "provider": prov,
          })

            saved += 1
            pbar.update(1)

        if saved < images_per_class:
            print(f"[WARN] Underfilled '{lbl}': {saved}/{images_per_class}. Add more datasets or relax filters.")

    pbar.close()

    # 5.4 Write metadata & label map
    write_metadata(out_dir, rows)
    save_label_map(out_dir, all_labels)

    # 5.5 Summary & optional zip
    by_split = {"train":0,"val":0,"test":0}
    by_label = {l:0 for l in all_labels}
    for r in rows:
        by_split[r["split"]] += 1
        by_label[r["fine_label"]] += 1

    print("\n[DONE]")
    print("[SUMMARY] by split:", by_split)
    print("[SUMMARY] by class:", by_label)

    if zip_out:
        make_zip(out_dir)

# ==============================================================================
# 6) IO WRITERS
# ==============================================================================

def write_metadata(out_dir: Path, rows: List[dict]):
    fields = [
        "uid","split","fine_label","coarse_label","width","height","out_relpath",
        "sha256","source_path","provider"
    ]
    with open(out_dir/"metadata.csv","w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    with open(out_dir/"metadata.jsonl","w",encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r)+"\n")

def save_label_map(out_dir: Path, fine_classes: List[str]):
    fine_sorted = sorted(fine_classes)
    fine_to_index = {c:i for i,c in enumerate(fine_sorted)}
    fine_to_coarse = {c: map_validity(c) for c in fine_sorted}
    coarse_classes = sorted(set(fine_to_coarse.values()))

    with open(out_dir/"label_map.json","w",encoding="utf-8") as f:
        json.dump({
            "fine_classes": fine_sorted,
            "fine_to_index": fine_to_index,
            "alias_to_canonical": ALIAS_TO_CANONICAL,
            "coarse_classes": coarse_classes,
            "fine_to_coarse": fine_to_coarse,
        }, f, indent=2)

def write_readme(out_dir: Path, size: int, images_per_class: int, splits: Tuple[float,float,float]):
    with open(out_dir/"README.txt","w",encoding="utf-8") as f:
        f.write("Universal KaggleHub Dataset Builder\n")
        f.write("===================================\n\n")
        f.write(f"Image size: {size}x{size}\n")
        f.write(f"Target per class: {images_per_class}\n")
        f.write(f"Splits: train={splits[0]:.3f}, val={splits[1]:.3f}, test={splits[2]:.3f}\n\n")
        f.write("Datasets:\n")
        for ds in DATASETS:
            if ds.get("enabled", True):
                f.write(f"  - {ds['name']}  ({ds['slug']})  strategy={ds['strategy']}\n")

def make_zip(out_dir: Path):
    zip_path = out_dir.with_suffix(".zip")
    if zip_path.exists(): zip_path.unlink()
    shutil.make_archive(str(out_dir), "zip", root_dir=str(out_dir))
    print(f"[OK] Wrote ZIP: {zip_path}")

# ==============================================================================
# 7) CLI (Notebook-safe)
# ==============================================================================

def main(argv: Optional[List[str]] = None):
    ap = argparse.ArgumentParser(description="Universal KaggleHub dataset builder (safe).",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--out", type=str, default="data_kagglehub_unified", help="Output directory")
    ap.add_argument("--images-per-class", type=int, default=1000, help="Target images saved per class")
    ap.add_argument("--size", type=int, default=DEFAULT_SIZE, help="Output square size (px)")
    ap.add_argument("--splits", type=float, nargs=3, default=[0.8,0.1,0.1], metavar=("TRAIN","VAL","TEST"))
    ap.add_argument("--seed", type=int, default=1337, help="RNG seed")
    ap.add_argument("--jpg-quality", type=int, default=DEFAULT_JPG_QUALITY, help="JPEG quality")
    ap.add_argument("--zip", action="store_true", help="Zip dataset at the end")
    ap.add_argument("--classes", type=str, default=None, help="Comma-separated subset of canonical labels to keep")
    args, _ = ap.parse_known_args(argv)

    only = None
    if args.classes:
        only = [normalize_label(x) for x in args.classes.split(",") if x.strip()]

    out_dir = Path(args.out).expanduser().resolve()
    ensure_dir(out_dir)
    write_readme(out_dir, args.size, args.images_per_class, tuple(args.splits))

    build_dataset(
        out_dir=out_dir,
        size=args.size,
        images_per_class=args.images_per_class,
        splits=(args.splits[0], args.splits[1], args.splits[2]),
        seed=args.seed,
        jpg_quality=args.jpg_quality,
        zip_out=args.zip,
        only_classes=only,
    )

if __name__ == "__main__":
    # Notebook-safe: unrecognized Jupyter kernel flags are ignored by parse_known_args
    main()
