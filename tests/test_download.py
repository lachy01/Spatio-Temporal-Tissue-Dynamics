"""Tests for the selective Xenium GEO downloader. No real network calls --
`requests.get` is monkeypatched everywhere."""

import io
import tarfile

import pytest

from tissue_dynamics import download as dl


class _FakeStreamResponse:
    """Stands in for `requests.get(..., stream=True)`'s return value."""

    def __init__(self, content, headers=None):
        self._content = content
        self.headers = headers or {}

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size=1024 * 1024):
        for i in range(0, len(self._content), chunk_size):
            yield self._content[i:i + chunk_size]

    @property
    def raw(self):
        return io.BytesIO(self._content)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


class _FakeTextResponse:
    """Stands in for the plain (non-streaming) `requests.get` used for GEO's text summary."""

    def __init__(self, text):
        self.text = text

    def raise_for_status(self):
        pass


class _CountingBytesIO(io.BytesIO):
    """A BytesIO that tracks cumulative bytes actually read from it."""

    def __init__(self, data):
        super().__init__(data)
        self.bytes_read = 0

    def read(self, size=-1):
        chunk = super().read(size)
        self.bytes_read += len(chunk)
        return chunk


def _tar_bytes(entries, mode="w"):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode=mode) as tar:
        for name, data in entries.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


# --------------------------------------------------------------------------
# _wanted_files / _match_wanted
# --------------------------------------------------------------------------

def test_wanted_files_defaults_to_required_and_recommended_only():
    files = dl._wanted_files(False, False, None)
    assert set(dl.REQUIRED_FILES) <= files
    assert set(dl.RECOMMENDED_FILES) <= files
    assert not (set(dl.MTX_FALLBACK_FILES) & files)
    assert not (set(dl.TRANSCRIPT_FILES) & files)


def test_wanted_files_includes_optional_extras_when_requested():
    files = dl._wanted_files(True, True, ["custom_extra.csv"])
    assert set(dl.MTX_FALLBACK_FILES) <= files
    assert set(dl.TRANSCRIPT_FILES) <= files
    assert "custom_extra.csv" in files


def test_match_wanted_respects_directory_boundary():
    wanted = {"cells.csv.gz", "cell_feature_matrix/matrix.mtx.gz"}
    assert dl._match_wanted("cells.csv.gz", wanted) == "cells.csv.gz"
    assert dl._match_wanted("outs/cells.csv.gz", wanted) == "cells.csv.gz"
    assert dl._match_wanted("wrapper/cell_feature_matrix/matrix.mtx.gz", wanted) == "cell_feature_matrix/matrix.mtx.gz"
    # no false positive on a bare string-suffix match without a path boundary
    assert dl._match_wanted("xcells.csv.gz", wanted) is None
    assert dl._match_wanted("unrelated.txt", wanted) is None


# --------------------------------------------------------------------------
# _stream_tar_and_extract
# --------------------------------------------------------------------------

def test_stream_tar_and_extract_keeps_wanted_skips_unwanted(tmp_path):
    tar_bytes = _tar_bytes({"cells.csv.gz": b"wanted", "analysis_summary.html": b"unwanted"})

    found = dl._stream_tar_and_extract(io.BytesIO(tar_bytes), tmp_path, {"cells.csv.gz"}, set())

    assert found == {"cells.csv.gz"}
    assert (tmp_path / "cells.csv.gz").read_bytes() == b"wanted"
    assert not (tmp_path / "analysis_summary.html").exists()


def test_stream_tar_and_extract_recurses_into_nested_tar_gz(tmp_path):
    nested_bytes = _tar_bytes({"experiment.xenium": b"{}"}, mode="w:gz")
    outer_bytes = _tar_bytes({
        "cells.csv.gz": b"data",
        "wrapper/sample_outs.tar.gz": nested_bytes,
    })

    wanted = {"cells.csv.gz", "experiment.xenium"}
    found = dl._stream_tar_and_extract(io.BytesIO(outer_bytes), tmp_path, wanted, set())

    assert found == wanted
    assert (tmp_path / "cells.csv.gz").read_bytes() == b"data"
    assert (tmp_path / "experiment.xenium").read_bytes() == b"{}"


def test_stream_tar_and_extract_stops_reading_once_all_wanted_found(tmp_path):
    padding = b"x" * (5 * 1024 * 1024)
    tar_bytes = _tar_bytes({"cells.csv.gz": b"wanted", "morphology.ome.tif": padding})

    reader = _CountingBytesIO(tar_bytes)
    found = dl._stream_tar_and_extract(reader, tmp_path, {"cells.csv.gz"}, set())

    assert found == {"cells.csv.gz"}
    assert not (tmp_path / "morphology.ome.tif").exists()
    # proves the stream was abandoned well before reading through the ~5MB
    # unwanted member's body -- the real bandwidth-saving guarantee
    assert reader.bytes_read < len(padding)


# --------------------------------------------------------------------------
# _geo_supplementary_files / _download_file (requests.get mocked)
# --------------------------------------------------------------------------

def test_geo_supplementary_files_parses_listing(monkeypatch):
    text = (
        "!Sample_supplementary_file_1 = ftp://ftp.ncbi.nlm.nih.gov/geo/samples/"
        "GSM799nnn/GSM7990532/suppl/GSM7990532_output.tar.gz\n"
        "!Sample_supplementary_file_2 = ftp://ftp.ncbi.nlm.nih.gov/geo/samples/"
        "GSM799nnn/GSM7990532/suppl/GSM7990532_transcripts.csv.gz\n"
        "!Sample_title = some unrelated metadata line\n"
    )
    monkeypatch.setattr("requests.get", lambda *a, **k: _FakeTextResponse(text))

    files = dl._geo_supplementary_files("GSM7990532")

    # ftp:// is rewritten to https:// -- requests has no FTP adapter, and
    # ftp.ncbi.nlm.nih.gov serves the identical path tree over HTTPS.
    assert files == {
        "GSM7990532_output.tar.gz": (
            "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM799nnn/GSM7990532/suppl/GSM7990532_output.tar.gz"
        ),
        "GSM7990532_transcripts.csv.gz": (
            "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM799nnn/GSM7990532/suppl/GSM7990532_transcripts.csv.gz"
        ),
    }


def test_geo_supplementary_files_empty_when_no_listing(monkeypatch):
    monkeypatch.setattr("requests.get", lambda *a, **k: _FakeTextResponse("!Sample_title = x\n"))
    assert dl._geo_supplementary_files("GSM0000000") == {}


def test_download_file_writes_content(tmp_path, monkeypatch):
    content = b"hello world"
    monkeypatch.setattr(
        "requests.get",
        lambda *a, **k: _FakeStreamResponse(content, headers={"content-length": str(len(content))}),
    )

    dest = tmp_path / "sub" / "cells.csv.gz"
    dl._download_file("http://example.com/cells.csv.gz", dest)

    assert dest.read_bytes() == content


# --------------------------------------------------------------------------
# download_xenium_sample (end to end, GEO + network fully mocked)
# --------------------------------------------------------------------------

def test_download_xenium_sample_uses_individually_listed_files(tmp_path, monkeypatch):
    written = {}

    def fake_geo_supplementary_files(gsm_id):
        listing = {name: f"http://example.com/{name}" for name in dl.REQUIRED_FILES + dl.RECOMMENDED_FILES}
        listing["morphology.ome.tif"] = "http://example.com/morphology.ome.tif"  # must never be fetched
        return listing

    def fake_download_file(url, dest_path, label=None):
        written[dest_path.name] = url
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(b"fake")

    monkeypatch.setattr(dl, "_geo_supplementary_files", fake_geo_supplementary_files)
    monkeypatch.setattr(dl, "_download_file", fake_download_file)

    outs_dir = dl.download_xenium_sample("GSM0000001", dest_dir=tmp_path)

    assert outs_dir == tmp_path / "GSM0000001_outs"
    assert set(written) == set(dl.REQUIRED_FILES) | set(dl.RECOMMENDED_FILES)
    assert "morphology.ome.tif" not in written


def test_download_xenium_sample_falls_back_to_archive_extraction(tmp_path, monkeypatch):
    def fake_geo_supplementary_files(gsm_id):
        return {"GSM0000001_output.tar.gz": "http://example.com/output.tar.gz"}

    extract_calls = []

    def fake_extract(url, outs_dir, wanted):
        extract_calls.append((url, set(wanted)))
        found = set()
        for name in wanted:
            path = outs_dir / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fake")
            found.add(name)
        return found

    monkeypatch.setattr(dl, "_geo_supplementary_files", fake_geo_supplementary_files)
    monkeypatch.setattr(dl, "_extract_selected_from_remote_archive", fake_extract)

    outs_dir = dl.download_xenium_sample("GSM0000001", dest_dir=tmp_path)

    assert len(extract_calls) == 1
    url, wanted = extract_calls[0]
    assert url == "http://example.com/output.tar.gz"
    assert wanted == set(dl.REQUIRED_FILES) | set(dl.RECOMMENDED_FILES)
    for name in dl.REQUIRED_FILES:
        assert (outs_dir / name).exists()


def test_download_xenium_sample_skips_geo_lookup_when_all_files_cached(tmp_path, monkeypatch):
    outs_dir = tmp_path / "GSM0000001_outs"
    for name in dl.REQUIRED_FILES + dl.RECOMMENDED_FILES:
        path = outs_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"already here")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("should not query GEO when every wanted file is already on disk")

    monkeypatch.setattr(dl, "_geo_supplementary_files", fail_if_called)

    result = dl.download_xenium_sample("GSM0000001", dest_dir=tmp_path)

    assert result == outs_dir


def test_download_xenium_sample_raises_on_missing_required_file(tmp_path, monkeypatch):
    def fake_geo_supplementary_files(gsm_id):
        # only lists non-matching files -- nothing wanted is ever found
        return {"unrelated.txt": "http://example.com/unrelated.txt"}

    monkeypatch.setattr(dl, "_geo_supplementary_files", fake_geo_supplementary_files)

    with pytest.raises(RuntimeError, match="Could not find required file"):
        dl.download_xenium_sample("GSM0000001", dest_dir=tmp_path)


def test_download_xenium_sample_warns_on_missing_recommended_file(tmp_path, monkeypatch):
    def fake_geo_supplementary_files(gsm_id):
        # every REQUIRED file present, but no RECOMMENDED ones
        return {name: f"http://example.com/{name}" for name in dl.REQUIRED_FILES}

    def fake_download_file(url, dest_path, label=None):
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(b"fake")

    monkeypatch.setattr(dl, "_geo_supplementary_files", fake_geo_supplementary_files)
    monkeypatch.setattr(dl, "_download_file", fake_download_file)

    with pytest.warns(UserWarning, match="Could not find recommended file"):
        dl.download_xenium_sample("GSM0000001", dest_dir=tmp_path)
