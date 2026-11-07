"""Cloud cleanup stays within the linked application and requires replacement settings."""

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[1]


def driver():
    spec = importlib.util.spec_from_file_location(
        "branding_cloud_driver", ROOT / "deploy/fullstack/deploy_fullstack.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("failure", [None, "owner", "replacement", "preview"])
def test_cleanup_requires_owned_projects_and_confirmed_production_replacements(
    tmp_path, monkeypatch, failure
):
    import huggingface_hub

    module = driver()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    team = {"id": "team-owned", "slug": "owned-team"}
    state = {
        "retired_project_names": ["earlierapp"],
        "api_project": {"id": "api-owned", "name": "regulatrace-api"},
        "web_project": {"id": "web-owned", "name": "regulatrace-web"},
    }
    variables = [
        {
            "key": "EARLIERAPP_DATABASE_URL",
            "target": ["preview" if failure == "preview" else "production"],
        },
        {"key": "REGULATRACE_DATABASE_URL", "target": ["production"]},
        {"key": "UNRELATED_SETTING", "target": ["production"]},
    ]
    if failure == "replacement":
        variables.pop(1)
    records = {"api-owned": list(variables), "web-owned": []}

    def api(endpoint, *, team):
        assert team["id"] == "team-owned"
        key = "api-owned" if "api-owned" in endpoint else "web-owned"
        if "/env?" in endpoint:
            assert "decrypt=false" in endpoint
            return {"envs": list(records[key])}
        return {
            "accountId": "other-team" if failure == "owner" else "team-owned",
            "name": state["api_project" if key == "api-owned" else "web_project"]["name"],
        }

    def command(arguments):
        assert arguments[:3] == ["env", "rm", "EARLIERAPP_DATABASE_URL"]
        assert arguments[3:7] == ["production", "--yes", "--scope", "owned-team"]
        linked = json.loads((tmp_path / "bundles/api/.vercel/project.json").read_text())
        assert linked == {
            "orgId": "team-owned",
            "projectId": "api-owned",
            "projectName": "regulatrace-api",
        }
        records["api-owned"] = [v for v in records["api-owned"] if v["key"] != arguments[2]]

    execute = Mock(side_effect=command)
    monkeypatch.setattr(module, "vercel_api", api)
    monkeypatch.setattr(module, "cli", execute)
    hub = Mock()
    secrets = {"EARLIERAPP_MODEL_WORKER_KEY": object(), "REGULATRACE_MODEL_WORKER_KEY": object()}
    hub.get_space_secrets.side_effect = lambda space: dict(secrets)
    hub.get_space_variables.return_value = {}
    hub.delete_space_secret.side_effect = lambda space, key: secrets.pop(key)
    monkeypatch.setattr(huggingface_hub, "HfApi", Mock(return_value=hub))
    ready = Mock()
    monkeypatch.setattr(module, "wait_for_model_worker", ready)
    if failure:
        with pytest.raises(module.DeploymentError):
            module.cleanup_retired_settings(state, team)
        execute.assert_not_called()
        hub.delete_space_secret.assert_not_called()
    else:
        assert module.cleanup_retired_settings(state, team) == 2
        execute.assert_called_once()
        assert {v["key"] for v in records["api-owned"]} == {
            "REGULATRACE_DATABASE_URL",
            "UNRELATED_SETTING",
        }
        hub.delete_space_secret.assert_called_once_with(module.SPACE, "EARLIERAPP_MODEL_WORKER_KEY")
        ready.assert_called_once_with(hub)


@pytest.mark.parametrize("missing_secret", [False, True])
def test_space_rename_deletes_only_retired_source_and_migrates_known_secrets(
    tmp_path, monkeypatch, missing_secret
):
    import huggingface_hub

    module = driver()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    folder = tmp_path / "bundles/HuggingFace_Space"
    folder.mkdir(parents=True)
    (folder / "app.py").write_text("# synthetic worker")
    files = {"app.py": hashlib.sha256((folder / "app.py").read_bytes()).hexdigest()}
    (tmp_path / "bundles/space-manifest.json").write_text(
        json.dumps({"repo_id": module.SPACE, "files": files})
    )
    hub = Mock()
    hub.whoami.return_value = {"name": "abdul547"}
    hub.space_info.return_value = SimpleNamespace(sdk="gradio", sha="previous-worker-commit")
    hub.get_space_runtime.return_value = SimpleNamespace(
        requested_hardware="zero-a10g", hardware="zero-a10g"
    )
    old_key = "EARLIERAPP_UNKNOWN_KEY" if missing_secret else "EARLIERAPP_GROQ_API_KEY"
    hub.get_space_secrets.return_value = {old_key: object()}
    hub.get_space_variables.return_value = {}
    hub.upload_folder.return_value = SimpleNamespace(oid="renamed-worker-commit")
    monkeypatch.setattr(huggingface_hub, "HfApi", Mock(return_value=hub))
    monkeypatch.setattr(huggingface_hub, "get_token", lambda: "synthetic-authenticated-session")
    monkeypatch.setattr(module, "wait_for_model_worker", Mock())
    config = {
        "REGULATRACE_GROQ_API_KEY": "synthetic-private-groq",
        "REGULATRACE_MODEL_WORKER_SECRET": "synthetic-private-worker",
    }
    if missing_secret:
        with pytest.raises(module.DeploymentError):
            module.update_model_space(config, previous_packages=["earlierapp"])
        hub.add_space_secret.assert_not_called()
        hub.upload_folder.assert_not_called()
    else:
        assert module.update_model_space(config, previous_packages=["earlierapp"]) == (
            "renamed-worker-commit"
        )
        assert hub.upload_folder.call_args.kwargs["delete_patterns"] == ["src/earlierapp/**"]
        assert hub.upload_folder.call_args.kwargs["allow_patterns"] == ["app.py"]
        assert hub.upload_folder.call_args.kwargs["parent_commit"] == "previous-worker-commit"


@pytest.mark.parametrize("name", ["../outside", "regulatrace", "a/b", "", None])
def test_unsafe_retired_package_metadata_is_rejected_before_cloud_calls(name):
    module = driver()
    with pytest.raises(module.DeploymentError):
        module.retired_prefixes([name])
