#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import re
import time
from collections import deque
from pathlib import Path
from urllib.parse import urljoin, urlparse

import numpy as np
import requests
from bs4 import BeautifulSoup
from PIL import Image
from tqdm import tqdm


BASE = "https://pokemondb.net"
SPRITES_INDEX = "https://pokemondb.net/sprites"
IMG_PREFIX = "https://img.pokemondb.net/sprites/"


def slugify(s: str) -> str:
    s = s.lower()
    s = s.replace("♀", "-f").replace("♂", "-m")
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-")


def parse_csv_set(s: str | None):
    if s is None:
        return None
    out = {x.strip().lower() for x in s.split(",") if x.strip()}
    return out or None


def get_soup(session: requests.Session, url: str, delay: float) -> BeautifulSoup:
    time.sleep(delay)
    r = session.get(url, timeout=30)
    r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser")


def find_pokemon_pages(
    session: requests.Session,
    delay: float,
    gens: set[int] | None,
    pokemon_names: set[str] | None,
    limit: int | None,
):
    soup = get_soup(session, SPRITES_INDEX, delay)

    pages = []
    seen = set()
    current_gen = None

    for el in soup.find_all(["h2", "a"]):
        if el.name == "h2":
            text = el.get_text(" ", strip=True)
            match = re.search(r"Generation\s+(\d+)", text, flags=re.I)
            if match:
                current_gen = int(match.group(1))
            continue

        href = el.get("href")
        name = el.get_text(" ", strip=True)

        if not href:
            continue
        if not href.startswith("/sprites/"):
            continue
        if href == "/sprites":
            continue
        if current_gen is None:
            continue

        name_slug = slugify(name)

        if gens is not None and current_gen not in gens:
            continue

        if pokemon_names is not None:
            if name.lower() not in pokemon_names and name_slug not in pokemon_names:
                continue

        url = urljoin(BASE, href)

        if url in seen:
            continue

        seen.add(url)

        pages.append(
            {
                "pokemon": name,
                "pokemon_slug": name_slug,
                "generation": current_gen,
                "page_url": url,
            }
        )

        if limit is not None and len(pages) >= limit:
            break

    return pages


def extract_sprite_urls(soup: BeautifulSoup):
    urls = set()

    for a in soup.select("a[href]"):
        href = a["href"]
        if href.startswith(IMG_PREFIX):
            urls.add(href)

    for img in soup.select("img"):
        for attr in ["src", "data-src", "data-original"]:
            val = img.get(attr)
            if not val:
                continue

            full = urljoin(BASE, val)
            if full.startswith(IMG_PREFIX):
                urls.add(full)

    return sorted(urls)


def rel_path_from_img_url(img_url: str) -> Path:
    parsed = urlparse(img_url)
    path = parsed.path

    marker = "/sprites/"
    if marker not in path:
        return Path(Path(path).name)

    return Path(path.split(marker, 1)[1])


def sprite_game_variant(img_url: str):
    rel = rel_path_from_img_url(img_url)
    parts = rel.parts

    if len(parts) >= 3:
        game = parts[0].lower()
        variant = parts[1].lower()
    elif len(parts) == 2:
        game = parts[0].lower()
        variant = ""
    else:
        game = ""
        variant = ""

    return game, variant


def keep_sprite_url(
    img_url: str,
    games: set[str] | None,
    variants: set[str] | None,
    skip_back: bool,
    skip_shiny: bool,
):
    game, variant = sprite_game_variant(img_url)

    if games is not None and game not in games:
        return False

    if variants is not None and variant not in variants:
        return False

    if skip_back and "back" in variant:
        return False

    if skip_shiny and "shiny" in variant:
        return False

    return True


def download_file(
    session: requests.Session,
    url: str,
    out_path: Path,
    delay: float,
    overwrite: bool,
):
    if out_path.exists() and out_path.stat().st_size > 0 and not overwrite:
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)

    time.sleep(delay)
    r = session.get(url, timeout=60)
    r.raise_for_status()
    out_path.write_bytes(r.content)

    return True


def sha1_file(path: Path) -> str:
    h = hashlib.sha1()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def connected_background_mask(rgb: np.ndarray, threshold: float):
    h, w, _ = rgb.shape

    corners = np.array(
        [
            rgb[0, 0],
            rgb[0, w - 1],
            rgb[h - 1, 0],
            rgb[h - 1, w - 1],
        ],
        dtype=np.float32,
    )

    bg_color = np.median(corners, axis=0)

    dist = np.linalg.norm(
        rgb.astype(np.float32) - bg_color[None, None, :],
        axis=-1,
    )

    similar_to_bg = dist <= threshold

    bg = np.zeros((h, w), dtype=bool)
    q = deque()

    for y, x in [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]:
        if similar_to_bg[y, x] and not bg[y, x]:
            bg[y, x] = True
            q.append((y, x))

    while q:
        y, x = q.popleft()

        for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if yy < 0 or yy >= h or xx < 0 or xx >= w:
                continue
            if bg[yy, xx]:
                continue
            if not similar_to_bg[yy, xx]:
                continue

            bg[yy, xx] = True
            q.append((yy, xx))

    return bg


def foreground_mask_from_rgba(arr: np.ndarray, bg_threshold: float):
    rgb = arr[..., :3]
    alpha = arr[..., 3]

    if alpha.min() < 255:
        return alpha > 0

    bg = connected_background_mask(rgb, threshold=bg_threshold)
    return ~bg


def make_black_white_sprite(
    img_path: Path,
    bw_path: Path,
    size: int | None,
    bg_threshold: float,
    bw_threshold: int,
    dither: bool,
    overwrite: bool,
):
    if bw_path.exists() and not overwrite:
        return False

    img = Image.open(img_path).convert("RGBA")

    if size is not None:
        img = img.resize((size, size), Image.Resampling.NEAREST)

    arr = np.asarray(img)
    rgb = arr[..., :3]
    fg = foreground_mask_from_rgba(arr, bg_threshold=bg_threshold)

    clean = np.full_like(rgb, 255)
    clean[fg] = rgb[fg]

    gray = Image.fromarray(clean.astype(np.uint8), mode="RGB").convert("L")

    if dither:
        bw = gray.convert("1", dither=Image.Dither.FLOYDSTEINBERG).convert("L")
    else:
        bw = gray.point(lambda p: 255 if p > bw_threshold else 0, mode="L")

    bw_path.parent.mkdir(parents=True, exist_ok=True)
    bw.save(bw_path)

    return True


def make_mask_sprite(
    img_path: Path,
    mask_path: Path,
    size: int | None,
    bg_threshold: float,
    overwrite: bool,
):
    if mask_path.exists() and not overwrite:
        return False

    img = Image.open(img_path).convert("RGBA")

    if size is not None:
        img = img.resize((size, size), Image.Resampling.NEAREST)

    arr = np.asarray(img)
    fg = foreground_mask_from_rgba(arr, bg_threshold=bg_threshold)

    mask = (fg.astype(np.uint8) * 255)

    mask_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask, mode="L").save(mask_path)

    return True


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--out", default="pokemondb_sprites")
    parser.add_argument("--delay", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=None)

    parser.add_argument("--gens", default=None)
    parser.add_argument("--pokemon", default=None)
    parser.add_argument("--pokemon-file", default=None)

    parser.add_argument("--games", default=None)
    parser.add_argument("--variants", default=None)

    parser.add_argument("--skip-back", action="store_true")
    parser.add_argument("--skip-shiny", action="store_true")

    parser.add_argument("--make-bw", action="store_true")
    parser.add_argument("--make-mask", action="store_true")

    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--bg-threshold", type=float, default=10.0)
    parser.add_argument("--bw-threshold", type=int, default=128)
    parser.add_argument("--no-dither", action="store_true")

    parser.add_argument("--overwrite-images", action="store_true")
    parser.add_argument("--overwrite-bw", action="store_true")
    parser.add_argument("--overwrite-mask", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument(
        "--user-agent",
        default="pokemon-sprite-research-crawler/0.1",
    )

    args = parser.parse_args()

    gens = None
    if args.gens is not None:
        gens = {int(x.strip()) for x in args.gens.split(",") if x.strip()}

    pokemon_names = parse_csv_set(args.pokemon)

    if args.pokemon_file is not None:
        from_file = set()

        with open(args.pokemon_file, "r", encoding="utf-8") as f:
            for line in f:
                x = line.strip()
                if x and not x.startswith("#"):
                    from_file.add(x.lower())
                    from_file.add(slugify(x))

        pokemon_names = (pokemon_names or set()) | from_file

    games = parse_csv_set(args.games)
    variants = parse_csv_set(args.variants)

    out_root = Path(args.out)
    img_root = out_root / "images"
    bw_root = out_root / "bw"
    mask_root = out_root / "masks"
    metadata_path = out_root / "metadata.csv"

    out_root.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({"User-Agent": args.user_agent})

    pages = find_pokemon_pages(
        session=session,
        delay=args.delay,
        gens=gens,
        pokemon_names=pokemon_names,
        limit=args.limit,
    )

    print(f"Found {len(pages)} Pokémon pages")

    rows = []

    for page in tqdm(pages, desc="Pokémon"):
        try:
            soup = get_soup(session, page["page_url"], args.delay)
            sprite_urls = extract_sprite_urls(soup)
        except Exception as e:
            print(f"[WARN] failed page {page['page_url']}: {e}")
            continue

        sprite_urls = [
            u
            for u in sprite_urls
            if keep_sprite_url(
                u,
                games=games,
                variants=variants,
                skip_back=args.skip_back,
                skip_shiny=args.skip_shiny,
            )
        ]

        for img_url in sprite_urls:
            rel = rel_path_from_img_url(img_url)
            game, variant = sprite_game_variant(img_url)

            local_path = img_root / page["pokemon_slug"] / rel
            bw_path = bw_root / page["pokemon_slug"] / rel.with_suffix(".png")
            mask_path = mask_root / page["pokemon_slug"] / rel.with_suffix(".png")

            row = {
                "pokemon": page["pokemon"],
                "pokemon_slug": page["pokemon_slug"],
                "generation": page["generation"],
                "page_url": page["page_url"],
                "image_url": img_url,
                "game": game,
                "variant": variant,
                "local_path": str(local_path),
                "bw_path": str(bw_path) if args.make_bw else "",
                "mask_path": str(mask_path) if args.make_mask else "",
                "downloaded_now": "",
                "bw_created_now": "",
                "mask_created_now": "",
                "sha1": "",
            }

            if args.dry_run:
                rows.append(row)
                continue

            try:
                downloaded = download_file(
                    session=session,
                    url=img_url,
                    out_path=local_path,
                    delay=args.delay,
                    overwrite=args.overwrite_images,
                )
            except Exception as e:
                print(f"[WARN] failed image {img_url}: {e}")
                continue

            row["downloaded_now"] = downloaded
            row["sha1"] = sha1_file(local_path)

            if args.make_bw:
                try:
                    made_bw = make_black_white_sprite(
                        img_path=local_path,
                        bw_path=bw_path,
                        size=args.size,
                        bg_threshold=args.bg_threshold,
                        bw_threshold=args.bw_threshold,
                        dither=not args.no_dither,
                        overwrite=args.overwrite_bw,
                    )
                    row["bw_created_now"] = made_bw
                except Exception as e:
                    print(f"[WARN] failed BW {local_path}: {e}")
                    row["bw_path"] = ""

            if args.make_mask:
                try:
                    made_mask = make_mask_sprite(
                        img_path=local_path,
                        mask_path=mask_path,
                        size=args.size,
                        bg_threshold=args.bg_threshold,
                        overwrite=args.overwrite_mask,
                    )
                    row["mask_created_now"] = made_mask
                except Exception as e:
                    print(f"[WARN] failed mask {local_path}: {e}")
                    row["mask_path"] = ""

            rows.append(row)

    with metadata_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "pokemon",
            "pokemon_slug",
            "generation",
            "page_url",
            "image_url",
            "game",
            "variant",
            "local_path",
            "bw_path",
            "mask_path",
            "downloaded_now",
            "bw_created_now",
            "mask_created_now",
            "sha1",
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("Done.")
    print(f"Images:   {img_root}")
    print(f"B/W:      {bw_root if args.make_bw else '(not requested)'}")
    print(f"Masks:    {mask_root if args.make_mask else '(not requested)'}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()