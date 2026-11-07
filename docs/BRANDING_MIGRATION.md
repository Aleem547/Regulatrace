# Regulatrace naming migration

The current application, Python package, command examples, environment settings,
frontend package and documentation use Regulatrace. Source attribution identifies
the upstream author and repository separately in `UPSTREAM.md`.

The updater targets the existing configured full-stack package and existing
`Aleem547/Regulatrace` checkout. It checks the repository, clean branch, exact
source baseline and payload checksums before changing the setup. The existing
database, document identifiers, vector collections, signing key, model-worker key
and deployment project identifiers are preserved.

The package folder is resolved as a literal filesystem path after the script
starts. An omitted or empty `-PackagePath` uses the updater's own directory.
To select the current folder explicitly, pass `-PackagePath (Get-Location).Path`.

Private settings are moved to the `REGULATRACE_` namespace. Original local settings
and replaced deployment files are backed up under
`.fullstack-state/branding-backup`. The backup remains private and excluded from Git.
Signed tokens issued before the rename can be accepted for a bounded seven-day
transition; new tokens identify Regulatrace. Refresh never extends the original
refresh-token expiry.

The model worker is rebuilt with the renamed module. Only retired module paths
derived from the previous package metadata are removed. Existing model secrets
with matching locally configured replacements are migrated before obsolete names
are removed. Unmapped private secrets require review rather than deletion.

Vercel replacement variables must be confirmed before retired production names
are removed. Cleanup checks the saved project identity and explicitly links each
CLI operation to that project. Other projects and non-production variables are
outside this cleanup. The renamed API and frontend then deploy with the canonical
configuration and repeat the hosted chat, citation, account, session and readiness
checks. The resulting receipt records the canonical application name and namespace.

The GitHub update requires the new branding receipt. It applies the reviewed rename
patch, updates local configuration names, and stages only the reviewed source
allowlist and public deployment receipt. It uses normal Git authentication and a
normal push to `main`.

Historical benchmark measurements remain upstream results. Renaming package and
configuration labels does not rerun those experiments or establish new accuracy.
