"""Phase 2: remote client must not import server implementation."""
from pathlib import Path


def test_client_py_has_no_server_imports():
    root = Path(__file__).resolve().parent.parent
    text = (root / "client.py").read_text(encoding="utf-8")
    assert "from server" not in text
    assert "import server" not in text


def test_inference_entrypoint_uses_http_client_only():
    root = Path(__file__).resolve().parent.parent
    text = (root / "inference.py").read_text(encoding="utf-8")
    assert "from server" not in text
    assert "import server" not in text
