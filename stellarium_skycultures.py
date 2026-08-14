#!/usr/bin/env python3
"""
stellarium_skycultures.py

Parser for the Stellarium sky-culture data set.

Stellarium ships one directory per culture under skycultures/, each containing
an index.json that describes the culture's constellations as polylines of
Hipparcos numbers. The culture identifier is taken from the directory name; a
directory without a readable index.json is skipped.

Reusable as a module:

    from stellarium_skycultures import load_skycultures
    cultures, constellations, members, segments = load_skycultures("skycultures")

or as a command-line tool that dumps the four tables as CSV:

    python3 stellarium_skycultures.py --root skycultures --outdir skycultures_csv

Tables
------
cultures        one row per culture
                culture, region, classification, n_constellations, n_stars,
                n_segments, n_skipped_refs

constellations  one row per constellation
                culture, constellation_id, name_english, name_native,
                n_stars, n_segments, is_single_star

members         one row per (constellation, star): the join key for astronomy
                culture, constellation_id, HIP, source

                Membership is drawn from all three places an index.json records
                a star, because each is evidence the culture knew it:

                  line    a vertex of a drawn constellation figure
                  anchor  a star registering the figure's illustration onto the
                          sky, so a star of that constellation by construction,
                          though it defines no segment
                  named   a star carrying a proper name of its own under
                          `common_names`, whether or not any figure uses it.
                          The strongest evidence there is, and independent of
                          the figures: Sirius in the Inuit sky is Flickering and
                          appears in no line at all.

                Reading only the line vertices, as an earlier version did,
                dropped Arcturus from the Leiden Aratea and Sirius from several
                canons, and made them look like stars those cultures had failed
                to notice.

segments        one row per drawn line segment: this is what defines the shape
                culture, constellation_id, seg_index, HIP_a, HIP_b

Non-stellar line references, e.g. "DSO:M42" for the Orion Nebula, cannot be
joined against a Hipparcos catalogue and are counted in n_skipped_refs rather
than silently dropped. A constellation drawn as a single repeated star, the
usual way of marking an isolated named star such as Polaris, is kept with the
star as a member, no segment, and is_single_star set.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, UTC
from pathlib import Path

import pandas as pd

# Directories Stellarium commonly installs its sky cultures into.
DEFAULT_ROOTS = (
    "/usr/share/stellarium/skycultures",
    "/usr/local/share/stellarium/skycultures",
    "/Applications/Stellarium.app/Contents/Resources/skycultures",
    "~/.stellarium/skycultures",
    "~/Library/Application Support/Stellarium/skycultures",
)


def log(msg: str) -> None:
    print(f"[{datetime.now(UTC).strftime('%H:%M:%S')}] {msg}")


def find_root(explicit: str | None = None) -> Path:
    """Locate the skycultures directory, preferring an explicit path."""
    if explicit:
        p = Path(explicit).expanduser()
        if not p.is_dir():
            raise SystemExit(f"Not a directory: {p}")
        return p
    for cand in DEFAULT_ROOTS:
        p = Path(cand).expanduser()
        if p.is_dir():
            return p
    raise SystemExit(
        "Could not locate a skycultures directory. Pass --root explicitly.\n"
        "Looked in:\n  " + "\n  ".join(DEFAULT_ROOTS)
    )


def _name_of(entry: dict) -> tuple[str, str]:
    """English and native name of a constellation, empty string when absent."""
    cn = entry.get("common_name") or {}
    if isinstance(cn, str):
        return cn, ""
    return str(cn.get("english", "") or ""), str(cn.get("native", "") or "")


def _polyline_segments(polyline) -> tuple[list[tuple[int, int]], list[int], int]:
    """Split one polyline into segments, member stars and unusable references.

    A polyline is a list of Hipparcos numbers to be joined in order. Entries
    that are not integers are non-stellar references such as "DSO:M42"; they are
    counted and excluded, and any segment that would have used them is dropped.
    """
    segments: list[tuple[int, int]] = []
    members: list[int] = []
    skipped = 0

    nodes: list[int | None] = []
    for node in polyline:
        if isinstance(node, bool):
            skipped += 1
            nodes.append(None)
        elif isinstance(node, int):
            nodes.append(node)
            members.append(node)
        elif isinstance(node, str) and node.lstrip("+-").isdigit():
            v = int(node)
            nodes.append(v)
            members.append(v)
        else:
            skipped += 1
            nodes.append(None)

    for a, b in zip(nodes[:-1], nodes[1:]):
        if a is None or b is None or a == b:
            continue
        segments.append((a, b))

    return segments, members, skipped


_HIP_KEY = re.compile(r"^HIP\s*(\d+)$", re.IGNORECASE)


def _anchor_hips(entry: dict) -> list[int]:
    """Hipparcos numbers anchoring a constellation's illustration.

    An anchor registers the artwork onto the sky, so it is by construction a
    star of that constellation, and evidence the culture knew it just as good as
    a line vertex. It defines no drawn segment, though, so anchors join the
    membership table and stay out of the segment table.
    """
    img = entry.get("image") or {}
    out = []
    for a in img.get("anchors") or []:
        hip = a.get("hip")
        if isinstance(hip, int) and not isinstance(hip, bool):
            out.append(hip)
    return out


def _named_star_hips(data: dict) -> list[int]:
    """Stars the culture gives a proper name of their own.

    The strongest evidence of all, and independent of any figure: a star can
    carry a name without belonging to a drawn constellation. Sirius in the Inuit
    sky is Flickering and appears in no line at all.
    """
    out = []
    for key in (data.get("common_names") or {}):
        m = _HIP_KEY.match(str(key).strip())
        if m:
            out.append(int(m.group(1)))
    return out


def parse_index(path: Path, culture: str) -> dict | None:
    """Parse one index.json. Returns None if the file is unreadable."""
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        log(f"  SKIP {culture}: {type(exc).__name__}: {exc}")
        return None

    classification = data.get("classification") or []
    if isinstance(classification, str):
        classification = [classification]

    constellations: list[dict] = []
    members: list[dict] = []
    segments: list[dict] = []
    skipped_total = 0

    for entry in data.get("constellations") or []:
        cid = str(entry.get("id", "")) or f"{culture}:{len(constellations)}"
        name_en, name_nat = _name_of(entry)

        seg_pairs: list[tuple[int, int]] = []
        star_set: list[int] = []
        for polyline in entry.get("lines") or []:
            segs, mem, skipped = _polyline_segments(polyline)
            seg_pairs.extend(segs)
            star_set.extend(mem)
            skipped_total += skipped

        anchors = _anchor_hips(entry)
        line_stars = sorted(set(star_set))
        uniq_stars = sorted(set(star_set) | set(anchors))
        # A constellation drawn as one star repeated has members but no segment;
        # this is how Stellarium marks an isolated named star.
        is_single = len(line_stars) == 1 and not seg_pairs

        constellations.append({
            "culture": culture,
            "constellation_id": cid,
            "name_english": name_en,
            "name_native": name_nat,
            "n_stars": len(uniq_stars),
            "n_line_stars": len(line_stars),
            "n_anchor_only": len(set(anchors) - set(line_stars)),
            "n_segments": len(seg_pairs),
            "is_single_star": is_single,
        })
        for hip in uniq_stars:
            members.append({
                "culture": culture, "constellation_id": cid, "HIP": hip,
                "source": "line" if hip in set(line_stars) else "anchor",
            })
        for k, (a, b) in enumerate(seg_pairs):
            segments.append({"culture": culture, "constellation_id": cid,
                             "seg_index": k, "HIP_a": a, "HIP_b": b})

    # Stars carrying a proper name of their own, whether or not any figure uses
    # them. Attached to no constellation, so they take an empty identifier.
    in_figures = {m["HIP"] for m in members}
    named_only = sorted(set(_named_star_hips(data)) - in_figures)
    for hip in named_only:
        members.append({"culture": culture, "constellation_id": "",
                        "HIP": hip, "source": "named"})

    all_stars = {m["HIP"] for m in members}
    culture_row = {
        "culture": culture,
        "region": str(data.get("region", "") or ""),
        "classification": ";".join(str(c) for c in classification),
        "n_constellations": len(constellations),
        "n_stars": len(all_stars),
        "n_stars_lines_only": len({m["HIP"] for m in members
                                   if m["source"] == "line"}),
        "n_anchor_only": len({m["HIP"] for m in members
                              if m["source"] == "anchor"} - {
                                  m["HIP"] for m in members
                                  if m["source"] == "line"}),
        "n_named_only": len(named_only),
        "n_segments": len(segments),
        "n_skipped_refs": skipped_total,
    }

    return {"culture": culture_row, "constellations": constellations,
            "members": members, "segments": segments}


def load_skycultures(root: str | Path, quiet: bool = False):
    """Parse every culture under root. Returns four DataFrames.

    Directories with no index.json, or whose index.json cannot be parsed, are
    skipped rather than raising, so a partial or in-progress Stellarium tree
    still yields the cultures it does contain.
    """
    root = Path(root).expanduser()
    dirs = sorted(d for d in root.iterdir() if d.is_dir())

    cultures, constellations, members, segments = [], [], [], []
    n_skipped_dirs = 0

    for d in dirs:
        index = d / "index.json"
        if not index.is_file():
            n_skipped_dirs += 1
            continue
        parsed = parse_index(index, d.name)
        if parsed is None:
            n_skipped_dirs += 1
            continue
        cultures.append(parsed["culture"])
        constellations.extend(parsed["constellations"])
        members.extend(parsed["members"])
        segments.extend(parsed["segments"])

    if not quiet:
        log(f"Parsed {len(cultures)} cultures from {root} "
            f"({n_skipped_dirs} directories skipped)")

    return (
        pd.DataFrame(cultures),
        pd.DataFrame(constellations),
        pd.DataFrame(members),
        pd.DataFrame(segments),
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Parse Stellarium sky cultures into tabular CSV.")
    p.add_argument("--root", default=None,
                   help="skycultures directory (auto-detected if omitted)")
    p.add_argument("--outdir", default="skycultures_csv",
                   help="output directory for the four CSV tables")
    args = p.parse_args()

    root = find_root(args.root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cultures, constellations, members, segments = load_skycultures(root)

    if cultures.empty:
        raise SystemExit(f"No parsable sky culture found under {root}")

    for name, df in (("cultures", cultures), ("constellations", constellations),
                     ("members", members), ("segments", segments)):
        path = outdir / f"{name}.csv"
        df.to_csv(path, index=False)
        log(f"  {path}  {len(df):,} rows")

    log("")
    log("Largest cultures by number of catalogued stars:")
    top = cultures.sort_values("n_stars", ascending=False).head(12)
    for _, r in top.iterrows():
        log(f"  {r['culture']:28s} {r['n_constellations']:4d} cost. "
            f"{r['n_stars']:5d} stelle  {r['n_segments']:5d} segmenti"
            + (f"  [{r['n_skipped_refs']} rif. non stellari]"
               if r["n_skipped_refs"] else ""))


if __name__ == "__main__":
    main()
