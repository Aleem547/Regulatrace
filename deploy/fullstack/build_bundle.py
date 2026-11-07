"""Assemble isolated deployment roots without heavy root-project API dependencies."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def copy_file(source: Path, destination: Path) -> None:
    if source.is_symlink() or not source.is_file():
        raise ValueError("Only ordinary reviewed files can be bundled")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def build(destination: Path) -> dict:
    destination.mkdir(parents=True, exist_ok=False)
    api, web, space = (destination / n for n in ("api", "web", "HuggingFace_Space"))
    for source in (ROOT / "src").rglob("*.py"):
        for target in (api, space):
            copy_file(source, target / source.relative_to(ROOT))
    for folder in ("alembic", "prompts"):
        for source in (ROOT / folder).rglob("*"):
            if source.is_file() and "__pycache__" not in source.parts:
                copy_file(source, api / source.relative_to(ROOT))
    for source in (ROOT / "prompts").glob("*.txt"):
        copy_file(source, space / source.relative_to(ROOT))
    copy_file(ROOT / "alembic.ini", api / "alembic.ini")
    if (ROOT / "data/tokenizer-cache").is_dir():
        for source in (ROOT / "data/tokenizer-cache").iterdir():
            if source.is_file() and len(source.name) == 40:
                copy_file(source, api / "tiktoken-cache" / source.name)
    for source in (ROOT / "deploy/vercel-api").iterdir():
        if source.is_file():
            copy_file(source, api / source.name)
    for source in (ROOT / "frontend").rglob("*"):
        relative = source.relative_to(ROOT / "frontend")
        if (
            source.is_file()
            and not source.name.startswith(".env")
            and not any(p in {"node_modules", "dist", ".vercel", ".git"} for p in relative.parts)
        ):
            copy_file(source, web / relative)
    for name in ("app.py", "UPSTREAM.md"):
        copy_file(ROOT / name, space / name)
    copy_file(ROOT / "deploy/huggingface/requirements.txt", space / "requirements.txt")
    (space / "README.md").write_text(
        """---
title: Regulatrace
emoji: 📚
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.31.0
app_file: app.py
python_version: 3.12
pinned: false
---

# Regulatrace

Document upload demo and authenticated model worker for the full application.
Original source attribution is retained in [UPSTREAM.md](UPSTREAM.md).
""",
        encoding="utf-8",
    )
    manifest = {
        "repo_id": "abdul547/Regulatrace",
        "files": {
            p.relative_to(space).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(space.rglob("*"))
            if p.is_file()
        },
    }
    (destination / "space-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for folder in (api, web):
        (folder / ".vercelignore").write_text(".env*\n**/__pycache__\n**/*.pyc\nnode_modules\n")
    return {
        "api_files": sum(p.is_file() for p in api.rglob("*")),
        "web_files": sum(p.is_file() for p in web.rglob("*")),
        "space_files": len(manifest["files"]),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    print(json.dumps(build(parser.parse_args().destination)))
