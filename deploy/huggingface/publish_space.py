"""Upload only the reviewed bundle to the existing free Space, without cloning."""

import argparse
import hashlib
import json
import re
import sys
import time
from collections import deque
from pathlib import Path

REPO = "abdul547/Regulatrace"
OWNER = "abdul547"


def show_failure_logs(api, *, build: bool) -> None:
    """Use the owner's cached login to show a bounded, redacted diagnostic tail."""
    try:
        lines = deque(maxlen=60)
        for chunk in api.fetch_space_logs(REPO, build=build, follow=False):
            for line in chunk.splitlines():
                line = re.sub(r"(?:hf_|gsk_)[A-Za-z0-9_-]+", "[redacted]", line)
                line = re.sub(r"(?i)(bearer\s+)\S+", r"\1[redacted]", line)
                lines.append(line[:2000])
        if lines:
            print("Recent build logs:" if build else "Recent runtime logs:", flush=True)
            print("\n".join(lines), flush=True)
        else:
            print("No buffered logs available; open the Space's Logs panel.", flush=True)
    except Exception as exc:
        print(
            f"Logs unavailable ({type(exc).__name__}); open the Space's Logs panel.",
            flush=True,
        )


def validate_bundle(folder: Path) -> list[str]:
    manifest = json.loads((folder.parent / "space-manifest.json").read_text(encoding="utf-8"))
    if manifest.get("repo_id") != REPO:
        raise ValueError("Unexpected upload target.")
    expected = manifest["files"]
    actual = {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()}
    if actual != set(expected):
        raise ValueError("Bundle contents changed; extract a fresh copy of the package.")
    for name, digest in expected.items():
        path = folder / name
        if path.is_symlink() or not path.resolve().is_relative_to(folder.resolve()):
            raise ValueError("Unsafe bundle path.")
        if (
            path.name == ".env"
            or path.name.startswith(".env.")
            or ".git" in path.parts
            or path.suffix.lower() in {".pdf", ".key", ".pem"}
        ):
            raise ValueError("Private documents/credentials cannot be uploaded by this script.")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("A reviewed bundle file was modified.")
    if not {"app.py", "README.md", "requirements.txt", "UPSTREAM.md"} <= actual:
        raise ValueError("The Space bundle is incomplete.")
    return sorted(actual)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=Path, default=Path(__file__).parent / "HuggingFace_Space")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--wait-seconds", type=int, default=480)
    args = parser.parse_args()
    try:
        files = validate_bundle(args.folder)
        print(f"Verified {len(files)} reviewed source files for {REPO}.", flush=True)
        if args.check_only:
            return 0
        from huggingface_hub import HfApi, get_token, login

        if not get_token():
            print("Sign in using a Hugging Face token with write access to this Space.", flush=True)
            login(add_to_git_credential=False)
        api = HfApi()
        if api.whoami().get("name") != OWNER:
            raise ValueError("The authenticated account is not the intended Space owner.")
        info = api.space_info(REPO)
        if info.sdk != "gradio":
            raise ValueError("The target must use the Gradio SDK.")
        runtime = api.get_space_runtime(REPO)
        hardware = runtime.requested_hardware or runtime.hardware
        if getattr(hardware, "value", hardware) != "zero-a10g":
            raise ValueError("Select free ZeroGPU in the Space before uploading.")
        print(
            "Uploading source to the existing ZeroGPU Space; no hardware or billing changes.",
            flush=True,
        )
        commit = api.upload_folder(
            repo_id=REPO,
            repo_type="space",
            folder_path=args.folder,
            allow_patterns=files,
            parent_commit=info.sha,
            commit_message="Deploy Regulatrace upload-based ZeroGPU demo",
        )
        print(f"Source commit: {commit.oid}", flush=True)
        print(f"Space: https://huggingface.co/spaces/{REPO}", flush=True)
        uploaded_at = time.monotonic()
        deadline = uploaded_at + max(0, args.wait_seconds)
        observed_build = False
        previous = None
        while time.monotonic() < deadline:
            runtime = api.get_space_runtime(REPO)
            stage = getattr(runtime.stage, "value", runtime.stage)
            if stage != previous:
                print(f"Build status: {stage}", flush=True)
                previous = stage
            if stage in {"BUILDING", "RUNNING_BUILDING", "APP_STARTING", "RUNNING_APP_STARTING"}:
                observed_build = True
            # The runtime endpoint can briefly retain the preceding build's status.
            if not observed_build and time.monotonic() - uploaded_at < 20:
                time.sleep(5)
                continue
            if stage == "RUNNING":
                print(
                    "The app process is running. "
                    "Test a PDF question and its citations in the browser."
                )
                return 0
            if stage in {"BUILD_ERROR", "RUNTIME_ERROR", "CONFIG_ERROR"}:
                show_failure_logs(api, build=stage == "BUILD_ERROR")
                return 2
            time.sleep(5)
        print("Source uploaded. The build is still pending; check the Space page.")
        return 0
    except ValueError as exc:
        print(f"Stopped: {exc}", file=sys.stderr)
    except Exception as exc:
        code = getattr(getattr(exc, "response", None), "status_code", None)
        print(
            f"Upload could not be completed ({type(exc).__name__}, HTTP {code}). "
            "Check connectivity, account/token permissions and the Space logs.",
            file=sys.stderr,
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
