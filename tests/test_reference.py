import sys

import pytest


def test_reference_child_failure_is_reported_in_python(tmp_path):
    pytest.importorskip("torch")
    from micro_rts_rs.reference import open_match

    child = [
        sys.executable, "-c",
        "import sys; sys.stdin.buffer.read(1); sys.stderr.write('test child failure'); sys.exit(7)",
    ]
    with open_match((tmp_path, child, None), "unused", 8) as match:
        with pytest.raises(RuntimeError, match="test child failure"):
            match.reset()
