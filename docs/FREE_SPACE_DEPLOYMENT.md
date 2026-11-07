# Free Regulatrace Space deployment

Target: <https://huggingface.co/spaces/abdul547/Regulatrace>, Gradio SDK, ZeroGPU.
Use the separate `Regulatrace_HuggingFace_Package.zip` for this workflow.

## Scope

This profile adapts the original question-answering workflow to session-local PDF
uploads. It uses the existing parser/chunker, local FastEmbed BM25/Qdrant, BGE
top-5 reranking, the original evidence gate/prompt budget and Groq generation.
The original API's default retrieval and public-only filtering are unchanged.
Only ranking runs inside `spaces.GPU`; retrieval and Groq calls stay outside it.

There is no PostgreSQL or remote Qdrant dependency for this profile. Uploaded
documents/indexes and the last five conversation exchanges are temporary. State
expires after 30 minutes without an update and is lost on restart. Gradio's file
cache cleanup runs every ten minutes for files older than ten minutes; this is
not a guarantee of deletion from the hosting provider's backups or logs.
The existing full app retains its separate authentication/persistence setup.

The original 42 PDFs are still missing. Uploaded filenames are not treated as
proof of RBI/SEBI origin. This profile makes no claim to reproduce historical
retrieval scores or answer accuracy. Missing or foreign citations are withheld;
membership checks do not establish that each claim is supported.

Limits: five text-based PDFs, 10 MiB each, 300 pages total, two million extracted
characters, 2,000 chunks, 1,000-character questions, eight queued jobs and one
active application job. No OCR is provided. Browser sessions use separate states;
the server retains at most 16 sessions, and older sessions can be evicted.

## Publish from Windows

1. Extract the Hugging Face package and open PowerShell in the folder containing
   `Publish-HuggingFace.ps1`. Keep its `HuggingFace_Space` subfolder and manifest
   together. No clone is required.
2. In the Space's **Settings / Variables and secrets**, add the private Secret
   `REGULATRACE_GROQ_API_KEY` using your own Groq Free Plan organization key.
   `REGULATRACE_GROQ_API_KEY` remains accepted. Do not put either key in source or chat.
3. Run:

   ```powershell
   powershell -NoProfile -ExecutionPolicy Bypass -File .\Publish-HuggingFace.ps1
   ```

   The uploader needs local Python 3.10+ and installs only the upload client into
   its own virtual environment. It does not install the ML application locally.
   Authenticate in the private terminal prompt with an `abdul547` Hugging Face
   token scoped to write this Space: <https://huggingface.co/settings/tokens>.
   This Hugging Face token is distinct from the Groq key stored in Space Secrets.
4. The script verifies the account, target, reviewed file hashes, Gradio SDK and
   free ZeroGPU selection. It uploads source with a remote-head guard. It does
   not create a new Space, change hardware, add storage or enable billing.
5. Watch build/runtime logs. The uploader waits up to eight minutes; if still
   building, check the Space page. On a failure it now prints a bounded, redacted
   build/runtime log excerpt using your cached login. A running process does not
   prove chat works.

## Repair the first build

The 6 November source upload succeeded, but its first requirements installation
failed. Download the corrected package and extract it into the same existing
package folder, replacing its files. This updates the deployment requirements,
their reviewed hashes, the uploader and the GitHub patch together. Run the same
Publish-HuggingFace.ps1 command again. Keep the Space's existing private Groq
secret; Hugging Face authentication is normally reused from the saved login.

The corrected lock removes the editable local installation and includes the
OAuth/MCP extras injected by Gradio's SDK. It selects Pydantic 2.12.5 to satisfy
those extras. To regenerate it for the current SDK platform pin:

```powershell
uv pip compile requirements-space.txt --python-version 3.12 --no-emit-package spaces --no-strip-extras --no-annotate --output-file deploy/huggingface/requirements.txt
```

This command is for maintaining the source. The upload package already contains
the compiled requirements; the Windows publisher does not need uv or local ML
dependencies.

## First hosted test

Use a text-based PDF you are permitted to process. A clearly fictional
`SYNTHETIC_TEST_ONLY.pdf` is included outside the upload folder for a software
smoke test; it is not a regulatory source. Its expected reporting deadline is
thirty days. Upload it in the app, click **Use these PDFs**, and ask when its
fictional reporting return is due. Check that the answer cites the actual page.
Then test an unrelated question, a follow-up, clearing/replacing documents and a
second browser session. Confirm that foreign or missing citations are withheld.
Only after these tests record the deployed source commit and live result.

## Publish the source update to GitHub

The package includes `Regulatrace_ZeroGPU_Update.patch` and `Update-GitHub.ps1`.
It applies the changes to the existing Windows Git checkout, validates the target
remote and clean/up-to-date main branch, stages the reviewed allowlist, commits
and pushes without force:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\Update-GitHub.ps1
```

Its default repository path is the original extracted package folder under
`Downloads\Compressed`. If different, supply `-RepositoryPath "C:\path\to\Regulatrace"`.
The script stops on incompatible source or concurrent remote changes. Do not
initialize a new unrelated history over the existing repository.

## Free-service limits and validation boundaries

Keep ZeroGPU selected and Storage Bucket / Dev Mode off. Free ZeroGPU includes
daily quotas and queues (currently five GPU minutes for a signed-in free visitor,
two minutes for an anonymous visitor). Use a Groq Free Plan organization; its
limits can produce HTTP 429 responses. No paid plan is required by this workflow.
These are current allowances, not a lifetime or always-on hosting guarantee.

Local checks cover upload validation, isolation, refusal/citation behavior,
entry-point/UI construction and upload guards. The real BM25 model loaded and
produced overlapping query/document terms. GPU inference and live Groq generation
have not been tested here. The PowerShell wrappers were reviewed, not executed
on Windows in this session. Docker and persistent full-app deployment are separate.

References checked 6 November 2026:
- <https://huggingface.co/docs/hub/spaces-zerogpu>
- <https://huggingface.co/docs/huggingface_hub/guides/upload>
- <https://console.groq.com/docs/rate-limits>
