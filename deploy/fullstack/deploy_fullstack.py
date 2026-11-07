import argparse
import getpass
import hashlib
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import time
from contextlib import closing
from pathlib import Path
from urllib.parse import urlencode, urlsplit
from uuid import UUID

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "bundles/api/src"))

import httpx
from sqlalchemy import delete, inspect, select, text
from sqlalchemy.orm import sessionmaker

from regulatrace.cloud_corpus import collection_names, load_prepared, publish_prepared
from regulatrace.config import get_settings
from regulatrace.db.models import ChatRequestAttempt, ChatSession, Message, QueryLog, User
from regulatrace.db.session import get_engine, normalize_database_url

CLI_VERSION = "65.12.1"
SPACE = "abdul547/Regulatrace"


class DeploymentError(Exception):
    """Only intentionally safe operator messages are displayed."""


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    path.chmod(0o600)


def validate_config(config):
    db = normalize_database_url(config["REGULATRACE_DATABASE_URL"])
    if (
        not db.host
        or not db.host.endswith(".neon.tech")
        or db.database != "regulatrace"
        or not db.username
        or not db.password
        or db.query.get("sslmode") not in {"require", "verify-full"}
    ):
        raise DeploymentError("Use a dedicated Neon database named regulatrace with TLS enabled.")
    url = urlsplit(config["REGULATRACE_QDRANT_URL"])
    if (
        url.scheme != "https"
        or not url.hostname
        or not url.hostname.endswith(".cloud.qdrant.io")
        or url.username
        or url.password
        or url.query
        or url.fragment
        or url.path not in {"", "/"}
        or url.port not in {None, 443, 6333}
    ):
        raise DeploymentError("Use the HTTPS endpoint of your free Qdrant Cloud cluster.")
    for name in (
        "REGULATRACE_QDRANT_API_KEY",
        "REGULATRACE_GROQ_API_KEY",
        "REGULATRACE_MODEL_WORKER_SECRET",
        "REGULATRACE_JWT_SECRET",
    ):
        if len(config.get(name, "")) < 24:
            raise DeploymentError("A required private setting is missing or too short.")


def configure(folder):
    path = folder / "secrets.json"
    if path.exists():
        config = json.loads(path.read_text())
    else:
        print("Private settings are entered locally; keep this folder private.")
        print("Use a FREE Neon project with database regulatrace and a FREE Qdrant cluster.")
        config = {
            "REGULATRACE_DATABASE_URL": getpass.getpass(
                "Neon regulatrace connection URL: "
            ).strip(),
            "REGULATRACE_QDRANT_URL": input("Qdrant cluster HTTPS URL: ").strip(),
            "REGULATRACE_QDRANT_API_KEY": getpass.getpass("Qdrant API key: ").strip(),
            "REGULATRACE_GROQ_API_KEY": getpass.getpass("Your Groq API key: ").strip(),
            "REGULATRACE_HF_TOKEN": getpass.getpass(
                "Optional HF read token for daily GPU quota (Enter to skip): "
            ).strip(),
            "REGULATRACE_JWT_SECRET": secrets.token_hex(32),
            "REGULATRACE_MODEL_WORKER_SECRET": secrets.token_hex(32),
        }
        validate_config(config)
        save_json(path, config)
    validate_config(config)
    config.update(
        {
            "REGULATRACE_ENVIRONMENT": "production",
            "REGULATRACE_DEBUG": "false",
            "REGULATRACE_LLM_PROVIDER": "groq",
            "REGULATRACE_LLM_MODEL": "openai/gpt-oss-120b",
            "REGULATRACE_MODEL_EXECUTION": "space",
            "REGULATRACE_DATABASE_NULL_POOL": "true",
            "REGULATRACE_CHAT_RATE_LIMIT_STORE": "postgres",
            "REGULATRACE_MODEL_WORKER_URL": "https://abdul547-regulatrace.hf.space",
            "REGULATRACE_CORS_ALLOWED_ORIGINS": "[]",
        }
    )
    manifest, _ = load_prepared(ROOT / "corpus")
    dense, sparse = collection_names(manifest)
    config.update(REGULATRACE_QDRANT_COLLECTION=dense, REGULATRACE_BM25_COLLECTION=sparse)
    return config


def cli(arguments, *, payload=None, interactive=False, timeout=900):
    npx = shutil.which("npx.cmd" if os.name == "nt" else "npx")
    if not npx:
        raise DeploymentError("Install Node.js 22 LTS or newer and reopen PowerShell.")
    command = [npx, "--yes", f"vercel@{CLI_VERSION}", *arguments]
    # Windows npx.cmd must not parse API URL ampersands as shell commands.
    if os.name == "nt":
        node = shutil.which("node.exe") or shutil.which("node")
        if not node:
            raise DeploymentError("Install Node.js 22 LTS or newer and reopen PowerShell.")
        candidates = [
            Path(npx).parent / "node_modules/npm/bin/npx-cli.js",
            Path(node).parent / "node_modules/npm/bin/npx-cli.js",
        ]
        entry = next((path for path in candidates if path.is_file()), None)
        if entry is None:
            raise DeploymentError(
                "The Node.js npm launcher is missing. Repair the Node.js installation."
            )
        command = [node, str(entry), *command[1:]]
    if interactive:
        result = subprocess.run(command, check=False)
    else:
        result = subprocess.run(
            command,
            input=payload,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
            encoding="utf-8",
        )
    if result.returncode:
        raise DeploymentError("Vercel did not complete the operation; check its dashboard.")
    return "" if interactive else result.stdout


def vercel_api(endpoint, *, method="GET", body=None, team=None):
    if team:
        endpoint += ("&" if "?" in endpoint else "?") + urlencode({"teamId": team["id"]})
    args = ["api", endpoint, "--method", method, "--raw"]
    if team:
        args += ["--scope", team["slug"]]
    if body is not None:
        args += ["--input", "-"]
    raw = cli(args, payload=json.dumps(body) if body is not None else None)
    try:
        result = json.loads(raw)
    except ValueError:
        raise DeploymentError("Vercel returned an unexpected API response.") from None
    if not isinstance(result, dict) or result.get("error"):
        raise DeploymentError("Vercel rejected the account or project operation.")
    return result


def select_hobby_team(state, state_path):
    try:
        teams = vercel_api("/v2/teams?limit=100")["teams"]
    except DeploymentError:
        print("Complete official Vercel browser sign-in.")
        cli(["login"], interactive=True)
        teams = vercel_api("/v2/teams?limit=100")["teams"]
    hobby = [t for t in teams if (t.get("billing") or {}).get("plan") == "hobby"]
    if "team_id" in state:
        hobby = [t for t in hobby if t["id"] == state["team_id"]]
    if not hobby:
        raise DeploymentError(
            "No free Hobby team is available. Select/create Hobby in Vercel first."
        )
    team = hobby[0]
    if len(hobby) > 1:
        for index, item in enumerate(hobby, 1):
            print(f"{index}. {item['slug']} (Hobby)")
        try:
            choice = int(input("Choose the free team number: ")) - 1
            if not 0 <= choice < len(hobby):
                raise ValueError
            team = hobby[choice]
        except ValueError:
            raise DeploymentError("Choose one of the displayed free teams.") from None
    confirmed = vercel_api(f"/v2/teams/{team['id']}")
    if (confirmed.get("billing") or {}).get("plan") != "hobby":
        raise DeploymentError("The selected account is not on Hobby; no deployment was made.")
    state["team_id"] = team["id"]
    save_json(state_path, state)
    return {"id": team["id"], "slug": team["slug"]}


def project(kind, framework, team, state, state_path):
    key = kind + "_project"
    if key in state:
        item = vercel_api(f"/v9/projects/{state[key]['id']}", team=team)
        if item.get("accountId") != team["id"] or item.get("name") != state[key]["name"]:
            raise DeploymentError("Saved project does not belong to this Regulatrace setup.")
    else:
        item = vercel_api(
            "/v11/projects",
            method="POST",
            team=team,
            body={
                "name": f"regulatrace-{kind}-{state['suffix']}",
                "framework": framework,
                "ssoProtection": {"deploymentType": "prod_deployment_urls_and_all_previews"},
            },
        )
        state[key] = {"id": item["id"], "name": item["name"]}
        save_json(state_path, state)
    save_json(
        ROOT / "bundles" / kind / ".vercel/project.json",
        {
            "orgId": team["id"],
            "projectId": item["id"],
            "projectName": item["name"],
        },
    )
    return item


def set_environment(item, values, team):
    body = [
        {
            "key": key,
            "value": value,
            "target": ["production"],
            "type": "sensitive"
            if any(s in key for s in ("KEY", "SECRET", "TOKEN", "DATABASE_URL"))
            else "plain",
        }
        for key, value in values.items()
        if value
    ]
    # The pinned CLI JSON-encodes object bodies, but passes array bodies to fetch unchanged.
    for variable in body:
        print(f"Saving Vercel setting: {variable['key']}", flush=True)
        result = vercel_api(
            f"/v10/projects/{item['id']}/env?upsert=true",
            method="POST",
            body=variable,
            team=team,
        )
        created = result.get("created")
        if isinstance(created, list) and len(created) == 1:
            created = created[0]
        if (
            result.get("failed")
            or not isinstance(created, dict)
            or created.get("key") != variable["key"]
            or "production" not in (created.get("target") or [])
        ):
            raise DeploymentError("Vercel did not confirm the required environment setting.")


def deploy(kind, item, team):
    print(f"Building Regulatrace {kind} on free Vercel Hobby...", flush=True)
    cli(
        [
            "deploy",
            "--prod",
            "--yes",
            "--scope",
            team["slug"],
            "--cwd",
            str(ROOT / "bundles" / kind),
        ]
    )
    result = vercel_api(f"/v9/projects/{item['id']}/domains", team=team)
    domains = [
        d["name"]
        for d in result["domains"]
        if d.get("verified")
        and not d.get("gitBranch")
        and not d.get("redirect")
        and d["name"].endswith(".vercel.app")
    ]
    if not domains:
        raise DeploymentError("Vercel has not assigned a public production domain yet.")
    preferred = item["name"] + ".vercel.app"
    return "https://" + (preferred if preferred in domains else sorted(domains)[0])


def migrate(engine):
    from alembic import command
    from alembic.config import Config
    from alembic.script import ScriptDirectory

    cfg = Config(str(ROOT / "bundles/api/alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "bundles/api/alembic"))
    known = {r.revision for r in ScriptDirectory.from_config(cfg).walk_revisions()}
    expected = {
        "alembic_version",
        "documents",
        "users",
        "chat_sessions",
        "messages",
        "feedback",
        "query_logs",
        "chat_request_attempts",
    }
    with engine.connect() as connection:
        tables = set(inspect(connection).get_table_names())
        if tables - expected or (tables and "alembic_version" not in tables):
            raise DeploymentError("Neon contains another application. Use a dedicated database.")
        if tables:
            versions = set(
                connection.execute(text("SELECT version_num FROM alembic_version")).scalars()
            )
            if not versions or not versions <= known:
                raise DeploymentError("The database migration history is not recognized.")
    command.upgrade(cfg, "head")
    print("Neon schema migrated to the reviewed revision.", flush=True)


def retired_prefixes(names):
    if not isinstance(names, list) or any(
        not isinstance(name, str)
        or not re.fullmatch(r"[a-z][a-z0-9_]{1,63}", name)
        or name == "regulatrace"
        for name in names
    ):
        raise DeploymentError("Previous application metadata is invalid.")
    return [name.upper() + "_" for name in names]


def replacement_key(key, prefixes):
    if not isinstance(key, str):
        return None
    key = key.upper()
    for prefix in prefixes:
        if key.startswith(prefix):
            return "REGULATRACE_" + key[len(prefix) :]
    return None


def cleanup_retired_settings(state, team):
    prefixes = retired_prefixes(state.get("retired_project_names", []))
    if not prefixes:
        return 0
    removed = 0
    for kind in ("api", "web"):
        item = state[kind + "_project"]
        owned = vercel_api(f"/v9/projects/{item['id']}", team=team)
        if owned.get("accountId") != team["id"] or owned.get("name") != item["name"]:
            raise DeploymentError("Saved project ownership changed; configuration was not removed.")
        response = vercel_api(f"/v10/projects/{item['id']}/env?decrypt=false", team=team)
        variables = response.get("envs")
        if not isinstance(variables, list):
            raise DeploymentError("Could not inspect application configuration names.")
        canonical = {v.get("key") for v in variables if "production" in (v.get("target") or [])}
        retired = [v for v in variables if replacement_key(v.get("key", ""), prefixes)]
        for variable in retired:
            if (
                set(variable.get("target") or []) != {"production"}
                or variable.get("gitBranch")
                or variable.get("customEnvironmentIds")
                or replacement_key(variable["key"], prefixes) not in canonical
            ):
                raise DeploymentError(
                    "An additional private setting needs migration before cleanup."
                )
        for variable in retired:
            save_json(
                ROOT / "bundles" / kind / ".vercel/project.json",
                {"orgId": team["id"], "projectId": item["id"], "projectName": item["name"]},
            )
            cli(
                [
                    "env",
                    "rm",
                    variable["key"],
                    "production",
                    "--yes",
                    "--scope",
                    team["slug"],
                    "--cwd",
                    str(ROOT / "bundles" / kind),
                ]
            )
            removed += 1
        checked = vercel_api(f"/v10/projects/{item['id']}/env?decrypt=false", team=team)
        remaining = checked.get("envs")
        if not isinstance(remaining, list) or any(
            replacement_key(v.get("key", ""), prefixes) for v in remaining
        ):
            raise DeploymentError("Configuration name cleanup is not complete.")
    from huggingface_hub import HfApi

    api = HfApi()
    secret_names, variable_names = api.get_space_secrets(SPACE), api.get_space_variables(SPACE)
    canonical_names = set(secret_names) | set(variable_names)
    space_removed = 0
    for values, remove in (
        (secret_names, api.delete_space_secret),
        (variable_names, api.delete_space_variable),
    ):
        for key in values:
            target = replacement_key(key, prefixes)
            if target:
                if target not in canonical_names:
                    raise DeploymentError("A private model-worker setting still needs migration.")
                remove(SPACE, key)
                removed += 1
                space_removed += 1
    if space_removed:
        wait_for_model_worker(api)
    remaining_names = set(api.get_space_secrets(SPACE)) | set(api.get_space_variables(SPACE))
    if any(replacement_key(key, prefixes) for key in remaining_names):
        raise DeploymentError("Model-worker configuration name cleanup is not complete.")
    print(f"Migrated {removed} retired configuration names; verifying deployment next.", flush=True)
    return removed


def update_model_space(config, *, previous_packages=None):
    from huggingface_hub import HfApi, get_token, login

    if not get_token():
        print("Complete Hugging Face browser sign-in for the existing Space.")
        login(add_to_git_credential=False)
    api = HfApi()
    if api.whoami().get("name") != "abdul547":
        raise DeploymentError("Sign in to Hugging Face as the Space owner abdul547.")
    info = api.space_info(SPACE)
    runtime = api.get_space_runtime(SPACE)
    hardware = runtime.requested_hardware or runtime.hardware
    if info.sdk != "gradio" or getattr(hardware, "value", hardware) != "zero-a10g":
        raise DeploymentError("Keep the existing Gradio Space on free ZeroGPU.")
    folder = ROOT / "bundles/HuggingFace_Space"
    manifest = json.loads((ROOT / "bundles/space-manifest.json").read_text())
    actual = {
        p.relative_to(folder).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in folder.rglob("*")
        if p.is_file()
    }
    if manifest.get("repo_id") != SPACE or actual != manifest["files"]:
        raise DeploymentError("The reviewed Space bundle changed. Extract a fresh package.")
    names = previous_packages or []
    prefixes = retired_prefixes(names)
    old_secrets = api.get_space_secrets(SPACE) if prefixes else {}
    old_variables = api.get_space_variables(SPACE) if prefixes else {}
    for key in old_secrets:
        target = replacement_key(key, prefixes)
        if target and not config.get(target):
            raise DeploymentError("A private model-worker setting must be migrated locally first.")
    for key, value in old_variables.items():
        target = replacement_key(key, prefixes)
        if target:
            if any(part in target for part in ("KEY", "SECRET", "TOKEN", "DATABASE_URL")):
                api.add_space_secret(SPACE, target, config.get(target) or value.value)
            elif target not in old_variables and target not in old_secrets:
                api.add_space_variable(SPACE, target, value.value)
    for key in old_secrets:
        target = replacement_key(key, prefixes)
        if target:
            api.add_space_secret(SPACE, target, config[target])
    api.add_space_secret(
        SPACE, "REGULATRACE_MODEL_WORKER_SECRET", config["REGULATRACE_MODEL_WORKER_SECRET"]
    )
    api.add_space_secret(SPACE, "REGULATRACE_GROQ_API_KEY", config["REGULATRACE_GROQ_API_KEY"])
    commit = api.upload_folder(
        repo_id=SPACE,
        repo_type="space",
        folder_path=folder,
        allow_patterns=sorted(actual),
        delete_patterns=[f"src/{name}/**" for name in names],
        parent_commit=info.sha,
        commit_message="Add authenticated Regulatrace model worker and retain upload demo",
    )
    print("Space source uploaded; waiting for the free model runtime...", flush=True)
    wait_for_model_worker(api)
    return commit.oid


def wait_for_model_worker(api):
    uploaded, previous = time.monotonic(), None
    while time.monotonic() - uploaded < 900:
        runtime = api.get_space_runtime(SPACE)
        stage = getattr(runtime.stage, "value", runtime.stage)
        if stage != previous:
            print(f"Space status: {stage}", flush=True)
            previous = stage
        if time.monotonic() - uploaded > 20:
            if stage == "RUNNING":
                from regulatrace.remote_models import call_worker

                try:
                    if call_worker({"action": "health"}).get("ready") is True:
                        return
                except Exception:
                    pass
            if stage in {"BUILD_ERROR", "RUNTIME_ERROR", "CONFIG_ERROR"}:
                raise DeploymentError(
                    "The Space build failed. Inspect its Logs panel before retrying."
                )
        time.sleep(5)
    raise DeploymentError("The Space has not passed its worker health check yet.")


def smoke_test(api_url, web_url, session_factory):
    """Use real retrieval/generation and temporary owned accounts; remove test rows afterward."""
    from concurrent.futures import ThreadPoolExecutor

    from regulatrace.db.rate_limit import admit_chat_attempt

    users, checks = [], {}

    def require(condition, name):
        if not condition:
            raise DeploymentError(
                f"Live verification failed: {name}. Deployment remains incomplete."
            )
        checks[name] = "passed"

    try:
        with httpx.Client(base_url=api_url, timeout=230, follow_redirects=False) as client:
            require(client.get("/health").status_code == 200, "api_health")
            require(client.get("/ready").status_code == 200, "postgres_qdrant_readiness")
            require(
                client.post("/chat", json={"question": "test"}).status_code == 401,
                "chat_authentication",
            )
            password, tokens = secrets.token_urlsafe(28), []
            for _ in range(2):
                email = f"regulatrace-smoke-{secrets.token_hex(8)}@example.com"
                registered = client.post(
                    "/auth/register", json={"email": email, "password": password}
                )
                require(registered.status_code == 201, "registration")
                users.append(UUID(registered.json()["id"]))
                logged_in = client.post("/auth/login", json={"email": email, "password": password})
                require(logged_in.status_code == 200, "login")
                tokens.append(logged_in.json())
            headers = {"Authorization": "Bearer " + tokens[0]["access_token"]}
            answer = client.post(
                "/chat",
                headers=headers,
                json={
                    "question": (
                        "How often must banks test Note Sorting Machines "
                        "for accuracy and consistency?"
                    )
                },
            )
            require(answer.status_code == 200, "qdrant_worker_groq_chat")
            value = answer.json()
            require(
                not value["abstained"] and "quarter" in value["answer"].lower(), "source_answer"
            )
            corpus, _ = load_prepared(ROOT / "corpus")
            document = next(
                d
                for d in corpus["documents"]
                if d["source_file"] == "473MDD1229693F0604B6D996B1DF75C81E466.PDF"
            )
            require(
                {"doc_id": document["metadata"]["doc_id"], "page": 8}
                in value["citation_validation"]["valid_citations"],
                "real_pdf_page_citation",
            )
            sid = value["session_id"]
            follow = client.post(
                "/chat",
                headers=headers,
                json={"question": "How often must this testing be repeated?", "session_id": sid},
            )
            require(
                follow.status_code == 200 and "quarter" in follow.json()["answer"].lower(),
                "follow_up_memory",
            )
            renewed = client.post(
                "/auth/refresh", json={"refresh_token": tokens[0]["refresh_token"]}
            )
            require(renewed.status_code == 200, "token_refresh")
            settings = get_settings()
            if settings.previous_token_issuer and time.time() < settings.previous_token_valid_until:
                import jwt

                from regulatrace.api.auth import _jwt_key

                claims = jwt.decode(
                    tokens[0]["access_token"],
                    _jwt_key(),
                    algorithms=["HS256"],
                    issuer="regulatrace",
                    audience="regulatrace-api",
                )
                claims.update(
                    iss=settings.previous_token_issuer, aud=settings.previous_token_issuer + "-api"
                )
                previous = jwt.encode(claims, _jwt_key(), algorithm="HS256")
                require(
                    client.get(
                        "/sessions", headers={"Authorization": "Bearer " + previous}
                    ).status_code
                    == 200,
                    "existing_token_compatibility",
                )
            headers = {"Authorization": "Bearer " + renewed.json()["access_token"]}
            history = client.get(f"/sessions/{sid}", headers=headers)
            require(
                history.status_code == 200 and len(history.json()["messages"]) == 4,
                "persistent_history",
            )
            listed = client.get("/sessions", headers=headers)
            require(
                listed.status_code == 200 and any(s["id"] == sid for s in listed.json()),
                "saved_session_list",
            )
            other = {"Authorization": "Bearer " + tokens[1]["access_token"]}
            require(
                client.get(f"/sessions/{sid}", headers=other).status_code == 404,
                "session_owner_isolation",
            )

            def admit(_):
                with session_factory() as db:
                    return admit_chat_attempt(db, users[1])

            with ThreadPoolExecutor(max_workers=4) as executor:
                admissions = list(executor.map(admit, range(12)))
            require(admissions.count(0) == 10, "postgres_concurrent_rate_limit")
            limited = client.post("/chat", headers=other, json={"question": "test"})
            require(
                limited.status_code == 429 and int(limited.headers.get("retry-after", "0")) > 0,
                "replica_shared_rate_limit",
            )
            cors = client.options(
                "/chat",
                headers={
                    "Origin": web_url,
                    "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": "authorization,content-type",
                },
            )
            require(
                cors.status_code == 200
                and cors.headers.get("access-control-allow-origin") == web_url,
                "frontend_api_cors",
            )
        with httpx.Client(timeout=30) as client:
            page = client.get(web_url)
            require(page.status_code == 200 and 'id="root"' in page.text, "frontend_page")
            scripts = re.findall(r'<script[^>]+src="([^"]+)"', page.text)
            require(
                bool(scripts)
                and all(
                    client.get(web_url + s).status_code == 200 for s in scripts if s.startswith("/")
                ),
                "frontend_assets",
            )
        return checks
    finally:
        if users:
            with session_factory() as db, db.begin():
                sessions = select(ChatSession.id).where(ChatSession.user_id.in_(users))
                db.execute(delete(Message).where(Message.session_id.in_(sessions)))
                db.execute(delete(ChatSession).where(ChatSession.user_id.in_(users)))
                db.execute(delete(QueryLog).where(QueryLog.user_id.in_(users)))
                db.execute(delete(ChatRequestAttempt).where(ChatRequestAttempt.user_id.in_(users)))
                db.execute(delete(User).where(User.id.in_(users)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    private = ROOT / ".fullstack-state"
    state_path = private / "deployment.json"
    try:
        manifest, _ = load_prepared(ROOT / "corpus")
        if args.check_only:
            for name in (
                "api/app.py",
                "api/pyproject.toml",
                "web/package-lock.json",
                "HuggingFace_Space/app.py",
            ):
                if not (ROOT / "bundles" / name).is_file():
                    raise DeploymentError("The deployment package is incomplete.")
            print(
                f"Verified {manifest['document_count']} documents and "
                f"{manifest['point_count']} prepared chunks."
            )
            print("Local package checks passed. No external service was changed.")
            return 0
        config = configure(private)
        os.environ.update(config)
        get_settings.cache_clear()
        state = (
            json.loads(state_path.read_text())
            if state_path.exists()
            else {"suffix": secrets.token_hex(4)}
        )
        team = select_hobby_team(state, state_path)
        engine = get_engine(connect_args={"connect_timeout": 15})
        try:
            migrate(engine)
            sessions = sessionmaker(bind=engine, expire_on_commit=False)
            from qdrant_client import QdrantClient

            with closing(
                QdrantClient(
                    url=config["REGULATRACE_QDRANT_URL"],
                    api_key=config["REGULATRACE_QDRANT_API_KEY"],
                    cloud_inference=True,
                    timeout=60,
                )
            ) as client:
                state["corpus"] = publish_prepared(ROOT / "corpus", client, sessions)
                save_json(state_path, state)
            state["space_commit"] = update_model_space(
                config, previous_packages=state.get("retired_project_names", [])
            )
            save_json(state_path, state)
            api_project = project("api", "fastapi", team, state, state_path)
            set_environment(api_project, config, team)
            state["retired_settings_removed"] = cleanup_retired_settings(state, team)
            api_url = deploy("api", api_project, team)
            state["api_url"] = api_url
            save_json(state_path, state)
            web_project = project("web", "vite", team, state, state_path)
            set_environment(web_project, {"VITE_API_BASE_URL": api_url}, team)
            web_url = deploy("web", web_project, team)
            state["web_url"] = web_url
            save_json(state_path, state)
            set_environment(
                api_project, {"REGULATRACE_CORS_ALLOWED_ORIGINS": json.dumps([web_url])}, team
            )
            deploy("api", api_project, team)
            state["verification"] = smoke_test(api_url, web_url, sessions)
            state["status"] = "live_core_verified"
            save_json(state_path, state)
            receipt = {
                k: state[k]
                for k in ("status", "web_url", "api_url", "space_commit", "corpus", "verification")
            }
            receipt["branding"] = {
                "name": "Regulatrace",
                "python_package": "regulatrace",
                "configuration_prefix": "REGULATRACE_",
                "retired_settings_removed": state["retired_settings_removed"],
            }
            save_json(ROOT / "DEPLOYMENT_RESULT.json", receipt)
            print(f"Verified live Regulatrace: {web_url}")
            print(
                "Saved DEPLOYMENT_RESULT.json. Run Update-GitHub.ps1 "
                "to publish source and the live link."
            )
            return 0
        finally:
            engine.dispose()
    except DeploymentError as exc:
        print(f"Stopped: {exc}", file=sys.stderr)
    except KeyboardInterrupt:
        print("Stopped. Run the same command to resume the saved setup.", file=sys.stderr)
    except Exception as exc:
        print(
            f"Deployment remains incomplete ({type(exc).__name__}). "
            "Check the provider dashboard and retry. No paid plan was selected.",
            file=sys.stderr,
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
