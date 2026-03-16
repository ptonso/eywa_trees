from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image


@dataclass(frozen=True)
class CropConfig:
    """Pixel crop margins from each side."""
    left: int = 0
    top: int = 125
    right: int = 0
    bottom: int = 55


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def iter_images(input_dir: Path, recursive: bool) -> Iterable[Path]:
    pattern = "**/*" if recursive else "*"
    for path in input_dir.glob(pattern):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def crop_image(image_path: Path, output_path: Path, cfg: CropConfig) -> None:
    with Image.open(image_path) as img:
        width, height = img.size

        left = cfg.left
        top = cfg.top
        right = width - cfg.right
        bottom = height - cfg.bottom

        if left >= right or top >= bottom:
            raise ValueError(
                f"Invalid crop for {image_path.name}: "
                f"image size=({width}, {height}), crop={cfg}"
            )

        cropped = img.crop((left, top, right, bottom))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cropped.save(output_path)


def build_output_path(input_root: Path, output_root: Path, image_path: Path) -> Path:
    return output_root / image_path.relative_to(input_root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch crop screenshot borders from a directory of images."
    )
    parser.add_argument("input_dir", type=Path, help="Directory with input images")
    parser.add_argument("output_dir", type=Path, help="Directory for cropped images")
    parser.add_argument("--left", type=int, default=0, help="Pixels to cut from left")
    parser.add_argument("--top", type=int, default=145, help="Pixels to cut from top")
    parser.add_argument("--right", type=int, default=0, help="Pixels to cut from right")
    parser.add_argument("--bottom", type=int, default=40, help="Pixels to cut from bottom")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Process images recursively inside subdirectories",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir

    if not input_dir.exists() or not input_dir.is_dir():
        raise NotADirectoryError(f"Invalid input directory: {input_dir}")

    cfg = CropConfig(
        left=args.left,
        top=args.top,
        right=args.right,
        bottom=args.bottom,
    )

    images = list(iter_images(input_dir, recursive=args.recursive))
    if not images:
        print("No supported images found.")
        return

    ok = 0
    failed = 0

    for image_path in images:
        out_path = build_output_path(input_dir, output_dir, image_path)
        try:
            crop_image(image_path, out_path, cfg)
            ok += 1
            print(f"[OK] {image_path} -> {out_path}")
        except Exception as exc:
            failed += 1
            print(f"[FAIL] {image_path}: {exc}")

    print(f"\nDone. Success: {ok}, Failed: {failed}")


if __name__ == "__main__":
    main()
