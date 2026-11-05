import pytest

from ornl_lotf.manifest import Record, read_manifest, update_status, write_manifest


def test_round_trip_preserves_fields_and_extra(tmp_path):
    recs = [
        Record(id="a", stage="00_bulk", mp_id="mp-1143", facet=[1, 1, 1], extra={"note": "x"}),
        Record(id="b", stage="00_bulk", defect_sites=[3, 7], defect_count=2),
    ]
    write_manifest(tmp_path, recs)
    back = read_manifest(tmp_path)
    assert back == recs


def test_duplicate_ids_rejected(tmp_path):
    with pytest.raises(ValueError):
        write_manifest(tmp_path, [Record(id="a", stage="s"), Record(id="a", stage="s")])


def test_update_status(tmp_path):
    write_manifest(tmp_path, [Record(id="a", stage="s")])
    update_status(tmp_path, {"a": "done"})
    assert read_manifest(tmp_path)[0].status == "done"
    with pytest.raises(ValueError):
        update_status(tmp_path, {"a": "bogus"})
    with pytest.raises(KeyError):
        update_status(tmp_path, {"zz": "done"})
