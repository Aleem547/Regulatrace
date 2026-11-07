"""Publishing guards: no credentials, real accounts or remote writes in these tests."""

import hashlib
import importlib.util
import json
from importlib.metadata import requires
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

pytest.importorskip("huggingface_hub")

_PATH = Path(__file__).resolve().parents[1] / "deploy/huggingface/publish_space.py"
_SPEC = importlib.util.spec_from_file_location("space_publisher", _PATH)
publisher = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(publisher)


@pytest.fixture
def bundle(tmp_path):
    folder = tmp_path / "HuggingFace_Space"
    folder.mkdir()
    for name in ("app.py", "README.md", "requirements.txt", "UPSTREAM.md"):
        (folder / name).write_text("reviewed example\n")
    manifest = {
        "repo_id": publisher.REPO,
        "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir()},
    }
    (tmp_path / "space-manifest.json").write_text(json.dumps(manifest))
    return folder


def test_changed_files_and_additional_secrets_are_rejected(bundle):
    assert len(publisher.validate_bundle(bundle)) == 4
    (bundle / ".env").write_text("synthetic-key=not-a-real-credential")
    with pytest.raises(ValueError, match="contents changed"):
        publisher.validate_bundle(bundle)
    path = bundle.parent / "space-manifest.json"
    manifest = json.loads(path.read_text())
    manifest["files"][".env"] = hashlib.sha256((bundle / ".env").read_bytes()).hexdigest()
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="credentials"):
        publisher.validate_bundle(bundle)


@pytest.mark.parametrize(
    "owner, hardware", [("wrong-owner", "zero-a10g"), ("abdul547", "t4-small")]
)
def test_wrong_account_or_paid_hardware_cannot_be_uploaded(bundle, monkeypatch, owner, hardware):
    import huggingface_hub

    api = Mock()
    api.whoami.return_value = {"name": owner}
    api.space_info.return_value = SimpleNamespace(sdk="gradio", sha="expected-head")
    api.get_space_runtime.return_value = SimpleNamespace(requested_hardware=hardware, hardware=None)
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: api)
    monkeypatch.setattr(huggingface_hub, "get_token", lambda: "synthetic-token")
    monkeypatch.setattr("sys.argv", ["publisher", "--folder", str(bundle), "--wait-seconds", "0"])
    assert publisher.main() == 1
    api.upload_folder.assert_not_called()


def test_upload_is_targeted_and_checks_remote_head(bundle, monkeypatch):
    import huggingface_hub

    api = Mock()
    api.whoami.return_value = {"name": "abdul547"}
    api.space_info.return_value = SimpleNamespace(sdk="gradio", sha="expected-head")
    api.get_space_runtime.return_value = SimpleNamespace(
        requested_hardware="zero-a10g", hardware=None
    )
    api.upload_folder.return_value = SimpleNamespace(oid="new-commit")
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: api)
    monkeypatch.setattr(huggingface_hub, "get_token", lambda: "synthetic-token")
    monkeypatch.setattr("sys.argv", ["publisher", "--folder", str(bundle), "--wait-seconds", "0"])
    assert publisher.main() == 0
    kwargs = api.upload_folder.call_args.kwargs
    assert kwargs["repo_id"] == "abdul547/Regulatrace"
    assert kwargs["repo_type"] == "space" and kwargs["parent_commit"] == "expected-head"
    assert set(kwargs["allow_patterns"]) == {
        "app.py",
        "README.md",
        "requirements.txt",
        "UPSTREAM.md",
    }


def test_sdk_build_dependencies_work_before_source_copy():
    """The SDK pip step has requirements.txt, but no installable local project."""
    tomllib = pytest.importorskip("tomllib")  # Space build uses Python 3.12.
    root = _PATH.parents[2]
    locked = {}
    for line in (root / "deploy/huggingface/requirements.txt").read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        requirement = Requirement(line.strip())
        assert requirement.url is None, "SDK build dependencies must come from the registry"
        assert len(list(requirement.specifier)) == 1
        pin = next(iter(requirement.specifier))
        assert pin.operator == "=="
        locked[canonicalize_name(requirement.name)] = (requirement, pin.version)
    project = tomllib.loads((root / "pyproject.toml").read_text())
    for value in project["project"]["dependencies"]:
        requirement = Requirement(value)
        assert locked[canonicalize_name(requirement.name)][1] in requirement.specifier
    assert {"oauth", "mcp"} <= locked["gradio"][0].extras
    assert "spaces" not in locked, "The Gradio SDK controls the spaces package version"
    # Check the installed SDK's actual metadata, including the extras it injects.
    for value in requires("gradio"):
        requirement = Requirement(value)
        if requirement.marker and not any(
            requirement.marker.evaluate({"extra": extra}) for extra in ("", "oauth", "mcp")
        ):
            continue
        version = locked[canonicalize_name(requirement.name)][1]
        assert version in requirement.specifier, str(requirement)


def test_failure_logs_are_bounded_and_redacted(capsys):
    api = Mock()
    api.fetch_space_logs.return_value = iter(
        [f"old line {i}\n" for i in range(65)]
        + ["error: hf_syntheticToken gsk_syntheticKey Authorization: Bearer syntheticBearer\n"]
    )
    publisher.show_failure_logs(api, build=True)
    output = capsys.readouterr().out
    assert "old line 0\n" not in output and "old line 64\n" in output
    assert "synthetic" not in output and output.count("[redacted]") == 3
    api.fetch_space_logs.assert_called_once_with(publisher.REPO, build=True, follow=False)


def test_failure_to_fetch_logs_does_not_hide_build_error(capsys):
    api = Mock()
    api.fetch_space_logs.side_effect = RuntimeError("private diagnostic must not be echoed")
    publisher.show_failure_logs(api, build=False)
    output = capsys.readouterr().out
    assert "RuntimeError" in output and "private diagnostic" not in output


def test_retry_ignores_the_preceding_build_error(bundle, monkeypatch):
    import huggingface_hub

    api = Mock()
    api.whoami.return_value = {"name": "abdul547"}
    api.space_info.return_value = SimpleNamespace(sdk="gradio", sha="previous-head")
    api.get_space_runtime.side_effect = [
        SimpleNamespace(requested_hardware="zero-a10g", hardware=None),
        SimpleNamespace(stage="BUILD_ERROR"),
        SimpleNamespace(stage="BUILDING"),
        SimpleNamespace(stage="RUNNING"),
    ]
    api.upload_folder.return_value = SimpleNamespace(oid="corrected-commit")
    clock = [0]
    monkeypatch.setattr(publisher.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        publisher.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds)
    )
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: api)
    monkeypatch.setattr(huggingface_hub, "get_token", lambda: "synthetic-token")
    monkeypatch.setattr("sys.argv", ["publisher", "--folder", str(bundle), "--wait-seconds", "100"])
    assert publisher.main() == 0
    api.fetch_space_logs.assert_not_called()
