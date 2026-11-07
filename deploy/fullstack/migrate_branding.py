import argparse
import json
import os
import re
import tempfile
import time
from pathlib import Path

import tomllib

PROJECT = "regulatrace"
PREFIX = "REGULATRACE_"


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".branding-", suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
        Path(temporary).chmod(0o600)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _json(path: Path, value: dict) -> None:
    _write(path, json.dumps(value, indent=2) + "\n")


def _backup(path: Path, destination: Path) -> None:
    if path.is_file() and not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(path.read_bytes())
        destination.chmod(0o600)


def retired_projects(state: dict, previous_project: str) -> list[str]:
    names = set(state.get("retired_project_names", []))
    if previous_project != PROJECT:
        names.add(previous_project)
    if any(
        not isinstance(name, str)
        or not re.fullmatch(r"[a-z][a-z0-9_]{1,63}", name)
        or name == PROJECT
        for name in names
    ):
        raise ValueError("Previous application metadata is invalid.")
    return sorted(names)


def migrate_settings(package: Path, previous_project: str) -> dict:
    private = package / ".fullstack-state"
    config_path, state_path = private / "secrets.json", private / "deployment.json"
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    state = json.loads(state_path.read_text(encoding="utf-8-sig"))
    if not isinstance(config, dict) or any(
        not isinstance(key, str) or not isinstance(value, str) for key, value in config.items()
    ):
        raise ValueError("Private settings must contain string names and values.")
    names = retired_projects(state, previous_project)
    migrated = dict(config)
    moved = 0
    for name in names:
        old_prefix = name.upper() + "_"
        for key in list(migrated):
            if key.startswith(old_prefix):
                target = PREFIX + key[len(old_prefix) :]
                if target in migrated and migrated[target] != migrated[key]:
                    raise ValueError("Conflicting private settings require reconciliation.")
                migrated[target] = migrated.pop(key)
                moved += 1
    if names and not migrated.get(PREFIX + "PREVIOUS_TOKEN_ISSUER"):
        migrated[PREFIX + "PREVIOUS_TOKEN_ISSUER"] = names[0]
        migrated[PREFIX + "PREVIOUS_TOKEN_VALID_UNTIL"] = str(int(time.time()) + 7 * 86400)
    required = ("DATABASE_URL", "QDRANT_URL", "QDRANT_API_KEY", "JWT_SECRET")
    if any(not migrated.get(PREFIX + suffix) for suffix in required):
        raise ValueError("Required private deployment settings are missing.")
    if not migrated.get("REGULATRACE_GROQ_API_KEY") or not migrated.get(
        "REGULATRACE_MODEL_WORKER_SECRET"
    ):
        raise ValueError("Required model credentials are missing.")
    backup = private / "branding-backup"
    _backup(config_path, backup / "secrets.json")
    _backup(state_path, backup / "deployment.json")
    state["retired_project_names"] = names
    state.setdefault("branding_migration", {})["configuration"] = "migrated"
    state["branding_migration"]["keys_migrated"] = max(
        moved, state["branding_migration"].get("keys_migrated", 0)
    )
    _json(config_path, migrated)
    _json(state_path, state)
    return {"settings_migrated": moved, "retired_projects": len(names)}


def migrate_dotenv(repository: Path, package: Path) -> bool:
    path = repository / ".env"
    if not path.is_file():
        return False
    private = package / ".fullstack-state"
    state = json.loads((private / "deployment.json").read_text(encoding="utf-8-sig"))
    names = retired_projects(state, PROJECT)
    original = path.read_text(encoding="utf-8-sig")
    lines = original.splitlines(keepends=True)
    existing = {}
    for line in lines:
        match = re.match(r"\s*(?:export\s+)?([A-Z][A-Z0-9_]*)\s*=(.*)", line)
        if match:
            existing[match[1]] = match[2]
    for name in names:
        prefix = name.upper() + "_"
        for index, line in enumerate(lines):
            match = re.match(r"(\s*(?:export\s+)?)([A-Z][A-Z0-9_]*)(\s*=)(.*)", line)
            if match and match[2].startswith(prefix):
                target = PREFIX + match[2][len(prefix) :]
                if target in existing and existing[target] != match[4]:
                    raise ValueError("Conflicting local settings require reconciliation.")
                lines[index] = line.replace(match[2], target, 1) if target not in existing else ""
                existing[target] = match[4]
    updated = "".join(lines)
    if PREFIX + "JWT_SECRET" in existing:
        settings = json.loads((private / "secrets.json").read_text(encoding="utf-8-sig"))
        for suffix in ("PREVIOUS_TOKEN_ISSUER", "PREVIOUS_TOKEN_VALID_UNTIL"):
            key = PREFIX + suffix
            if key not in existing and settings.get(key):
                if updated and not updated.endswith("\n"):
                    updated += "\n"
                updated += key + "=" + json.dumps(settings[key]) + "\n"
    if updated != original:
        _backup(path, private / "branding-backup" / "repository.env")
        _write(path, updated)
        return True
    return False


def remove_retired_bundle_modules(package: Path) -> None:
    import shutil

    state = json.loads((package / ".fullstack-state/deployment.json").read_text())
    for name in retired_projects(state, PROJECT):
        for kind in ("api", "HuggingFace_Space"):
            path = package / "bundles" / kind / "src" / name
            if path.is_symlink():
                raise ValueError("Unexpected source directory link.")
            if path.is_dir():
                backup = package / ".fullstack-state/branding-backup/bundles" / kind / "src" / name
                if not backup.exists():
                    backup.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copytree(path, backup)
                shutil.rmtree(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--env-only", action="store_true")
    args = parser.parse_args()
    try:
        if args.env_only:
            migrate_dotenv(args.repository, args.package)
            print("Local configuration names checked; private values preserved.")
        else:
            metadata = tomllib.loads((args.repository / "pyproject.toml").read_text())
            result = migrate_settings(args.package, metadata["project"]["name"])
            remove_retired_bundle_modules(args.package)
            print(f"Migrated {result['settings_migrated']} configuration names; keys preserved.")
        return 0
    except Exception:
        print("Branding migration stopped; inspect the package and private backup locally.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
