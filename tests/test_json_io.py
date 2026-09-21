import json

import pytest

from hlv_toolkits.data.json_io import load_records, write_records


def test_write_records_uses_json_arrays_and_rejects_jsonl(tmp_path):
    records = [{"id": "first", "value": 1}, {"id": "second", "value": 2}]
    json_path = tmp_path / "records.json"

    write_records(json_path, records)

    assert json.loads(json_path.read_text(encoding="utf-8")) == records
    assert load_records(json_path) == records
    with pytest.raises(ValueError, match=r"\.json"):
        write_records(tmp_path / "records.jsonl", records)
    (tmp_path / "legacy.jsonl").write_text('{"id": "legacy"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"\.json"):
        load_records(tmp_path / "legacy.jsonl")
