"""Selectively download a Xenium sample from GEO.

A full Xenium "outs" bundle is typically 4-5GB, almost all of it morphology
images, per-molecule transcript detections, segmentation boundary polygons,
and 10x's own analysis/report outputs -- none of which `io_xenium.read_xenium`
reads. This module keeps only the handful of small files it (and the manual
mtx-triplet fallback) actually need, which for a typical sample totals well
under 100MB.

Two download paths, tried in order, per file:

1. If GEO lists a GSM's supplementary files individually (some depositions
   do), fetch only the ones matching the wanted set directly -- exact,
   minimal-bandwidth downloads.
2. Otherwise -- the common case for Xenium GEO deposits: one combined
   per-GSM archive bundling the whole outs/ folder, sometimes nested inside
   a further *.tar.gz within that -- stream the archive and extract only
   matching members, never writing the huge unwanted ones to disk. This
   always saves disk space. It also saves network bandwidth whenever every
   wanted file is found before the stream reaches the huge unwanted
   members: tar has no directory/index, so this can't be guaranteed, but
   Xenium's own packaging tends to place small metadata/matrix files ahead
   of the multi-GB images and transcripts.
"""

import tarfile
from pathlib import Path

# Files read_xenium() needs no matter which internal path it ends up taking:
# spatialdata-io's cells_table path if it's installed and succeeds, or the
# manual cell_feature_matrix.h5 reader otherwise (read_xenium() falls back
# to that on any failure from spatialdata-io, not just it being missing --
# see io_xenium.py -- so these two are enough regardless of which
# spatialdata-io version, if any, is installed).
REQUIRED_FILES = (
    "cells.csv.gz",           # find_outs_dir()'s marker file + obsm["spatial"] fallback
    "cell_feature_matrix.h5",  # single-file cell x gene matrix
)
# Only spatialdata-io's cells_table path benefits from these; harmless to
# skip (warn, don't fail) if any are absent -- older Xenium software
# versions don't always produce every one of these.
RECOMMENDED_FILES = (
    "cells.parquet",       # spatialdata-io's preferred cell-table format, if present
    "experiment.xenium",   # run/panel metadata
    "gene_panel.json",     # feature-type metadata used by gex_only filtering
    "metrics_summary.csv",  # tiny QC summary, essentially free to keep
)
# Only needed by read_xenium()'s manual fallback path if
# cell_feature_matrix.h5 isn't present either (very old Xenium output).
MTX_FALLBACK_FILES = (
    "cell_feature_matrix/matrix.mtx.gz",
    "cell_feature_matrix/features.tsv.gz",
    "cell_feature_matrix/barcodes.tsv.gz",
)
# Only needed for the manual per-transcript QC path in
# xenium_gse250346_loader.py, not by read_xenium() itself.
TRANSCRIPT_FILES = ("transcripts.csv.gz",)

_ARCHIVE_SUFFIXES = (".tar.gz", ".tgz", ".tar")


def _wanted_files(include_mtx_fallback, include_transcripts, extra_files):
    files = set(REQUIRED_FILES) | set(RECOMMENDED_FILES)
    if include_mtx_fallback:
        files |= set(MTX_FALLBACK_FILES)
    if include_transcripts:
        files |= set(TRANSCRIPT_FILES)
    files |= set(extra_files or ())
    return files


def _match_wanted(member_name, wanted):
    """Return the `wanted` entry a tar member's path corresponds to, or None.

    Matches on a path suffix (with a `/` boundary) rather than exact
    equality, since archives commonly nest the outs/ folder under one or
    more wrapper directories.
    """
    for w in wanted:
        if member_name == w or member_name.endswith("/" + w):
            return w
    return None


class _ProgressReader:
    """Wraps a file-like object, printing periodic cumulative-bytes-read progress.

    Used as the `fileobj` for streaming tar reads so progress reflects real
    bytes pulled over the network, regardless of how much of the archive we
    end up skipping vs. extracting.
    """

    def __init__(self, fileobj, total=0, label="", report_every=25 * 1024 * 1024):
        self._fileobj = fileobj
        self._total = total
        self._label = label
        self._report_every = report_every
        self._read = 0
        self._last_reported = 0

    def read(self, size=-1):
        chunk = self._fileobj.read(size)
        self._read += len(chunk)
        if self._read - self._last_reported >= self._report_every:
            self._last_reported = self._read
            total_str = f" / {self._total / 1e6:.0f} MB" if self._total else ""
            print(f"\r  {self._label}: {self._read / 1e6:.0f}{total_str} MB read", end="", flush=True)
        return chunk


def _geo_supplementary_files(gsm_id):
    """Query GEO's lightweight text summary for a GSM's individual
    supplementary-file URLs (`{filename: url}`), without downloading anything."""
    import requests

    url = f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={gsm_id}&targ=self&form=text&view=quick"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    files = {}
    for line in resp.text.splitlines():
        if line.startswith("!Sample_supplementary_file"):
            file_url = line.split("=", 1)[1].strip()
            # GEO lists these as ftp://, but requests has no FTP adapter --
            # ftp.ncbi.nlm.nih.gov serves the identical path tree over HTTPS.
            if file_url.startswith("ftp://"):
                file_url = "https://" + file_url[len("ftp://"):]
            files[Path(file_url).name] = file_url
    return files


def _download_file(url, dest_path, label=None):
    """Plain direct download of one file (used when GEO lists it individually)."""
    import requests

    label = label or dest_path.name
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        written = 0
        with open(dest_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                written += len(chunk)
                total_str = f" / {total / 1e6:.1f} MB" if total else ""
                print(f"\r  {label}: {written / 1e6:.1f}{total_str}", end="", flush=True)
    print()


def _write_tar_member(tar, member, dest_path):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    src = tar.extractfile(member)
    if src is None:
        return
    with open(dest_path, "wb") as out:
        while True:
            chunk = src.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    print(f"\n  extracted {dest_path.name} ({member.size / 1e6:.1f} MB)")


def _stream_tar_and_extract(fileobj, outs_dir, wanted, found):
    """Iterate a (possibly compressed) tar stream, extracting members that
    match `wanted` and recursing into any nested archive members, stopping
    as soon as everything in `wanted` has been found.

    Uses tarfile's "r|*" streaming mode, which auto-detects gzip/bz2/xz
    compression (or none) from the stream itself, so callers don't need to
    know a member's compression from its name.
    """
    with tarfile.open(fileobj=fileobj, mode="r|*") as tar:
        for member in tar:
            if found >= wanted:
                break
            if not member.isfile():
                continue

            match = _match_wanted(member.name, wanted - found)
            if match:
                _write_tar_member(tar, member, outs_dir / match)
                found.add(match)
            elif member.name.endswith(_ARCHIVE_SUFFIXES):
                nested = tar.extractfile(member)
                if nested is not None:
                    found = _stream_tar_and_extract(nested, outs_dir, wanted, found)
    return found


def _extract_selected_from_remote_archive(url, outs_dir, wanted):
    import requests

    # (connect timeout, read timeout) -- reading through a multi-GB remote
    # archive to find the wanted members can take a while even though we
    # never write most of it to disk, so a generous per-read timeout avoids
    # spuriously aborting a slow-but-healthy transfer.
    with requests.get(url, stream=True, timeout=(10, 180)) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        reader = _ProgressReader(resp.raw, total=total, label=Path(url).name)
        found = _stream_tar_and_extract(reader, outs_dir, wanted, set())
    print()
    return found


def download_xenium_sample(gsm_id, dest_dir=".", *, include_mtx_fallback=False,
                             include_transcripts=False, extra_files=None):
    """Download only the files `read_xenium` needs for a Xenium sample on GEO.

    See the module docstring for the two-pass strategy (direct individual
    files where GEO lists them, else selective streaming extraction from
    whatever combined archive(s) it does list).

    Parameters
    ----------
    gsm_id : str
        GEO sample accession, e.g. "GSM7990532".
    dest_dir : str or Path
        Local directory; the sample lands in `dest_dir/<gsm_id>_outs/`.
    include_mtx_fallback : bool
        Also fetch the raw mtx triplet (`cell_feature_matrix/*.gz`), for
        environments without spatialdata-io installed (read_xenium's manual
        fallback path). Off by default since `cell_feature_matrix.h5` alone
        covers the primary (spatialdata-io) path and is much smaller.
    include_transcripts : bool
        Also fetch `transcripts.csv.gz` (can be hundreds of MB to GB) for
        the manual per-transcript QC path in `xenium_gse250346_loader.py`;
        not needed by `read_xenium` itself. Off by default.
    extra_files : list of str, optional
        Additional relative filenames (e.g. `"cell_boundaries.parquet"`) to
        keep, on top of the defaults above.

    Returns
    -------
    Path to the extracted outs/ folder, ready for `read_xenium`.
    """
    dest_dir = Path(dest_dir)
    outs_dir = dest_dir / f"{gsm_id}_outs"
    outs_dir.mkdir(parents=True, exist_ok=True)

    wanted = _wanted_files(include_mtx_fallback, include_transcripts, extra_files)
    already_present = {w for w in wanted if (outs_dir / w).exists()}
    found = set(already_present)
    if found >= wanted:
        print(f"All {len(wanted)} wanted file(s) already present in {outs_dir}; nothing to download.")
        return outs_dir
    if already_present:
        print(f"{len(already_present)}/{len(wanted)} wanted file(s) already present in {outs_dir}, skipping those.")

    print(f"Looking up {gsm_id} supplementary files on GEO ...")
    suppl_files = _geo_supplementary_files(gsm_id)
    if not suppl_files:
        raise RuntimeError(
            f"No supplementary files found for {gsm_id} on GEO -- check the accession, "
            "or that ncbi.nlm.nih.gov is reachable from this environment."
        )

    # Pass 1: anything GEO lists as an individually-downloadable file whose
    # name matches exactly -- the cheap, exact case.
    for filename, url in suppl_files.items():
        if filename in wanted and filename not in found:
            _download_file(url, outs_dir / filename)
            found.add(filename)

    # Pass 2: whatever's still missing must be inside one of the remaining
    # (archive-shaped) supplementary files -- stream and filter it.
    remaining = wanted - found
    archive_urls = [
        url for filename, url in suppl_files.items()
        if filename.endswith(_ARCHIVE_SUFFIXES) and filename not in found
    ]
    for url in archive_urls:
        if not remaining:
            break
        print(f"{len(remaining)} file(s) not individually listed on GEO; "
              f"streaming {Path(url).name} and extracting only what's needed ...")
        found |= _extract_selected_from_remote_archive(url, outs_dir, remaining)
        remaining = wanted - found

    missing_required = set(REQUIRED_FILES) - found
    if missing_required:
        raise RuntimeError(
            f"Could not find required file(s) {sorted(missing_required)} for {gsm_id} "
            f"in any GEO supplementary file. read_xenium() needs these -- check the "
            "sample's GEO page manually, or pass them via extra_files if they're "
            "present under a different name."
        )
    missing_recommended = (set(RECOMMENDED_FILES) & wanted) - found
    if missing_recommended:
        import warnings
        warnings.warn(
            f"Could not find recommended file(s) {sorted(missing_recommended)} for {gsm_id}; "
            "read_xenium() may still work without them, but double-check if it errors.",
            stacklevel=2,
        )

    print(f"\nDone: {outs_dir} ({len(found)}/{len(wanted)} wanted files)")
    return outs_dir
