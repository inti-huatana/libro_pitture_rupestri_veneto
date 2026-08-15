#!/usr/bin/env python3
"""
star_table.py

Fast, cached reading of the large star tables the other programs share.

Three things make the difference, in decreasing order of effect.

Only the needed columns are parsed. A trajectory file carries some two dozen,
and any one program wants five or six; the rest cost their full parsing time and
memory for nothing.

The first read writes a binary cache beside the CSV, and later reads take that
instead, as long as it is not older than the CSV. Reading a column-oriented
binary file is one to two orders of magnitude faster than parsing text, and the
cache is derived data: deleting it costs only the next read.

Where pyarrow is installed its CSV parser is used, being multithreaded and
several times faster than the default one. It is optional, and its absence only
loses that factor.

None of this changes what the programs read: the CSV stays the source of truth
and the cache is rebuilt whenever it goes stale.

Usage
-----
    from star_table import read_star_table
    df = read_star_table("BSGRID/trajectories.csv",
                         ["epoch_kyr_from_year0", "HIP", "ra_deg", "dec_deg"])
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CACHE_SUFFIX = ".cache.parquet"


def pivot_epoch_star(df: pd.DataFrame, value_cols: list[str]):
    """Reshape a long trajectory table into (epoch, star) arrays.

    Does the job of a chain of pandas pivots, but in one pass over integer
    codes and with a single allocation per column, which is what makes it
    usable once the catalogue reaches naked-eye depth: at some nine thousand
    stars a pivot per quantity costs a sort and a full intermediate frame each
    time, and every program here wants four or five quantities.

    Combinations absent from the table stay NaN rather than raising, so a star
    missing from some epochs does not bring the reshape down.
    """
    ep = np.sort(df["epoch"].unique())
    hp = np.sort(df["HIP"].unique())
    ie = np.searchsorted(ep, df["epoch"].to_numpy())
    ih = np.searchsorted(hp, df["HIP"].to_numpy())
    out = {}
    for c in value_cols:
        a = np.full((ep.size, hp.size), np.nan)
        a[ie, ih] = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
        out[c] = a
    return ep.astype(float), hp.astype(np.int64), out


def star_labels(df: pd.DataFrame) -> dict[int, str]:
    """One display name per HIP: proper name, else Bayer, else the number."""
    have = [c for c in ("NAME", "Bayer") if c in df.columns]
    if not have:
        return {int(h): f"HIP {int(h)}" for h in df["HIP"].unique()}
    first = df.groupby("HIP")[have].first()
    out = {}
    for hip, r in first.iterrows():
        nm = str(r["NAME"]).strip() if "NAME" in have else ""
        by = str(r["Bayer"]).strip() if "Bayer" in have else ""
        out[int(hip)] = (nm if nm and nm != "nan"
                         else by if by and by != "nan" else f"HIP {int(hip)}")
    return out


def cache_path_for(path: Path) -> Path:
    return path.with_name(path.name + CACHE_SUFFIX)


def _read_csv(path: Path, columns: list[str] | None) -> pd.DataFrame:
    """Parse the CSV, preferring pyarrow's reader when it is available."""
    try:
        return pd.read_csv(path, usecols=columns, engine="pyarrow")
    except Exception:
        return pd.read_csv(path, usecols=columns, low_memory=False)


def read_star_table(path, columns: list[str] | None = None,
                    use_cache: bool = True, verbose: bool = True) -> pd.DataFrame:
    """Read a star table, going through a binary cache when one is usable.

    The cache holds every column, not the subset asked for, so that a later call
    wanting different ones still hits it. Building it therefore costs one full
    parse, which is paid back by the first reuse.
    """
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"Not found: {path}")
    cache = cache_path_for(path)

    if use_cache and cache.exists() and cache.stat().st_mtime >= path.stat().st_mtime:
        try:
            df = pd.read_parquet(cache, columns=columns)
            if verbose:
                print(f"  cache: {cache.name} ({len(df):,} righe)", flush=True)
            return df
        except Exception:
            pass          # unreadable or written by another pandas: reparse

    full = _read_csv(path, None)
    if use_cache:
        try:
            full.to_parquet(cache, index=False)
            if verbose:
                print(f"  cache scritta: {cache.name}", flush=True)
        except Exception as exc:
            if verbose:
                print(f"  cache non scritta ({type(exc).__name__}); "
                      f"serve pyarrow o fastparquet", flush=True)
    return full[columns] if columns else full


def normalise_epoch(df: pd.DataFrame) -> pd.DataFrame:
    """Give the epoch column the single name the programs expect."""
    for old in ("epoch_kyr_from_year0", "epoch_kyr"):
        if old in df.columns and "epoch" not in df.columns:
            df = df.rename(columns={old: "epoch"})
            break
    if "epoch" in df.columns:
        df["epoch"] = df["epoch"].round(6)
    return df
