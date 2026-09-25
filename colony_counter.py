"""Bacterial colony counting on Petri dish images with the Hough Circle Transform.

Implements the pipeline described in "A Comparison of Bacterial Colonies Count from
Petri Dishes Utilizing Hough Transform and Traditional Manual Counting":

    crop dish + resize 512x512 (PIL) -> grayscale -> Laplacian sharpening -> adaptive threshold
    -> Hough Circle Transform (OpenCV) -> count + annotations (YOLO / COCO)
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

IMG_SIZE = 512
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class HoughParams:
    dp: float = 1.2
    min_dist: int = 6
    param1: int = 100  # upper Canny threshold
    param2: int = 11  # accumulator threshold (lower = more circles)
    min_radius: int = 3
    max_radius: int = 18
    block: int = 31  # adaptive-threshold neighbourhood
    c: int = -18  # adaptive-threshold offset (more negative = stricter)
    min_fill: float = 0.5  # min fraction of a circle's area that must be colony pixels


@dataclass
class Detection:
    image: str
    circles: list[tuple[int, int, int]] = field(default_factory=list)
    plate: tuple[int, int, int] | None = None
    seconds: float = 0.0

    @property
    def count(self) -> int:
        return len(self.circles)


def find_plate(bgr: np.ndarray) -> tuple[int, int, int]:
    """Locate the Petri dish as the largest bright blob on the dark background."""
    gray = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (9, 9), 0)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape
    if contours:
        (x, y), r = cv2.minEnclosingCircle(max(contours, key=cv2.contourArea))
        if r > 0.2 * min(h, w):
            return int(x), int(y), int(r)
    return w // 2, h // 2, min(h, w) // 2


def load_plate(path: Path) -> np.ndarray:
    """Load with PIL, crop a square around the dish and scale it to 512x512 (BGR out).

    Cropping first keeps the dish circular; resizing the raw 3:2 photo straight to
    512x512 would squash it into an ellipse.
    """
    img = Image.open(path).convert("RGB")
    img.thumbnail((1024, 1024))
    bgr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    x, y, r = find_plate(bgr)
    pad = cv2.copyMakeBorder(bgr, r, r, r, r, cv2.BORDER_CONSTANT, value=0)
    crop = pad[y: y + 2 * r, x: x + 2 * r]
    return cv2.resize(crop, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)


def preprocess(bgr: np.ndarray, params: HoughParams) -> dict[str, np.ndarray]:
    """Grayscale conversion, Laplacian sharpening and adaptive thresholding."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    lap = cv2.Laplacian(blur, cv2.CV_16S, ksize=3)
    sharp = cv2.convertScaleAbs(blur.astype(np.int16) - lap // 2)
    thresh = cv2.adaptiveThreshold(
        sharp, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, params.block, params.c
    )
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    # fill ring-shaped colonies (bright edge, flatter centre) so each is a solid blob
    contours, _ = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(thresh, contours, -1, 255, -1)
    return {"gray": gray, "sharp": sharp, "thresh": thresh}


def detect(path: Path, params: HoughParams | None = None, rim_margin: float = 0.80) -> tuple[Detection, dict]:
    params = params or HoughParams()
    t0 = time.perf_counter()
    bgr = load_plate(path)
    stages = preprocess(bgr, params)
    px = py = pr = IMG_SIZE // 2  # the dish fills the cropped frame

    mask = np.zeros_like(stages["gray"])
    cv2.circle(mask, (px, py), int(pr * rim_margin), 255, -1)
    # Hough runs on the smoothed binary colony map, restricted to the plate interior.
    hough_in = cv2.GaussianBlur(cv2.bitwise_and(stages["thresh"], mask), (5, 5), 1.5)
    stages["hough_in"] = hough_in

    raw = cv2.HoughCircles(
        hough_in, cv2.HOUGH_GRADIENT, dp=params.dp, minDist=params.min_dist,
        param1=params.param1, param2=params.param2,
        minRadius=params.min_radius, maxRadius=params.max_radius,
    )
    circles = []
    if raw is not None:
        for x, y, r in np.round(raw[0]).astype(int):
            if not mask[min(y, IMG_SIZE - 1), min(x, IMG_SIZE - 1)]:
                continue
            # keep circles that are mostly filled by colony pixels
            disk = np.zeros_like(mask)
            cv2.circle(disk, (int(x), int(y)), max(int(r), 1), 255, -1)
            inside = stages["thresh"][disk > 0]
            if inside.size and (inside > 0).mean() >= params.min_fill:
                circles.append((int(x), int(y), int(r)))

    det = Detection(image=str(path), circles=circles, plate=(px, py, pr),
                    seconds=time.perf_counter() - t0)
    stages["bgr"] = bgr
    return det, stages


def annotate(stages: dict, det: Detection) -> np.ndarray:
    """Raw | preprocessed | Hough result panel (Figure 3 of the paper)."""
    out = stages["bgr"].copy()
    if det.plate:
        cv2.circle(out, det.plate[:2], det.plate[2], (255, 0, 0), 1)
    for x, y, r in det.circles:
        cv2.circle(out, (x, y), r, (0, 0, 255), 1)
    cv2.putText(out, f"count: {det.count}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    thresh = cv2.cvtColor(stages["thresh"], cv2.COLOR_GRAY2BGR)
    return np.hstack([stages["bgr"], thresh, out])


def to_yolo(det: Detection, class_id: int = 0) -> str:
    lines = []
    for x, y, r in det.circles:
        lines.append(f"{class_id} {x / IMG_SIZE:.6f} {y / IMG_SIZE:.6f} {2 * r / IMG_SIZE:.6f} {2 * r / IMG_SIZE:.6f}")
    return "\n".join(lines)


def to_coco(dets: list[Detection], categories: list[str]) -> dict:
    images, anns = [], []
    ann_id = 1
    for img_id, det in enumerate(dets, 1):
        images.append({"id": img_id, "file_name": Path(det.image).name, "width": IMG_SIZE, "height": IMG_SIZE})
        cat = infer_species(det.image, categories)
        for x, y, r in det.circles:
            anns.append({
                "id": ann_id, "image_id": img_id, "category_id": categories.index(cat) + 1 if cat else 1,
                "bbox": [x - r, y - r, 2 * r, 2 * r], "area": float(np.pi * r * r), "iscrowd": 0,
            })
            ann_id += 1
    return {
        "images": images, "annotations": anns,
        "categories": [{"id": i + 1, "name": c} for i, c in enumerate(categories)] or [{"id": 1, "name": "colony"}],
    }


# checked in order; "paureginosa" is a typo present in the Rodrigues et al. dataset
SPECIES = {"p_aeruginosa": ("aerug", "paureg", "pseudomonas"),
           "e_coli": ("coli",),
           "s_aureus": ("aureus", "staph")}


def infer_species(path: str, categories: list[str] | None = None) -> str | None:
    p = Path(path).name.lower().replace(" ", "")
    for name, keys in SPECIES.items():
        if any(k in p for k in keys):
            return name
    return None


def iter_images(root: Path):
    if root.is_file():
        yield root
        return
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() in IMAGE_EXTS and "annotated" not in p.parts:
            yield p


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path, help="image file or folder (searched recursively)")
    ap.add_argument("-o", "--out", type=Path, default=Path("results"))
    ap.add_argument("--param2", type=int, default=HoughParams.param2)
    ap.add_argument("--min-dist", type=int, default=HoughParams.min_dist)
    ap.add_argument("--min-radius", type=int, default=HoughParams.min_radius)
    ap.add_argument("--max-radius", type=int, default=HoughParams.max_radius)
    ap.add_argument("--block", type=int, default=HoughParams.block, help="adaptive threshold block size")
    ap.add_argument("--c", type=int, default=HoughParams.c, help="adaptive threshold offset")
    ap.add_argument("--rim", type=float, default=0.80, help="fraction of dish radius analysed")
    ap.add_argument("--no-images", action="store_true", help="skip writing annotated panels")
    args = ap.parse_args()

    params = HoughParams(param2=args.param2, min_dist=args.min_dist,
                         min_radius=args.min_radius, max_radius=args.max_radius,
                         block=args.block, c=args.c)
    (args.out / "annotated").mkdir(parents=True, exist_ok=True)
    (args.out / "yolo").mkdir(parents=True, exist_ok=True)

    dets = []
    rows = ["image,species,auto_count,seconds"]
    for path in iter_images(args.input):
        det, stages = detect(path, params, rim_margin=args.rim)
        dets.append(det)
        rel = path.relative_to(args.input) if args.input.is_dir() else Path(path.name)
        stem = "__".join(rel.with_suffix("").parts)
        if not args.no_images:
            cv2.imwrite(str(args.out / "annotated" / f"{stem}.jpg"), annotate(stages, det))
        (args.out / "yolo" / f"{stem}.txt").write_text(to_yolo(det))
        rows.append(f"\"{rel}\",{infer_species(str(rel)) or ''},{det.count},{det.seconds:.4f}")
        print(f"{rel}: {det.count} colonies ({det.seconds * 1000:.0f} ms)")

    (args.out / "counts.csv").write_text("\n".join(rows) + "\n")
    (args.out / "coco.json").write_text(json.dumps(to_coco(dets, list(SPECIES)), indent=1))
    print(f"\n{len(dets)} images -> {args.out}/counts.csv, coco.json, yolo/, annotated/")


if __name__ == "__main__":
    main()
