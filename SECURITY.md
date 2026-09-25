# Security Policy

## Supported versions

| Version | Supported |
|---|---|
| latest `main` | yes |
| older commits / tags | no |

MinTok is a local CLI and library. It reads source code you point it at and runs
verification commands only when you explicitly invoke `verify`.

## Reporting a vulnerability

Use GitHub's **private vulnerability reporting** for this repository
(Security → Report a vulnerability), or contact the maintainers through the
organization `magebase`. Do **not** open a public issue for a security problem.

You can expect an initial response within 7 days.

## Disclosure policy

1. We confirm and triage the report privately.
2. We prepare a fix on a private branch and release a patched version.
3. We publish an advisory crediting the reporter (unless anonymity is requested).

## What is NOT a vulnerability

- The tool reads whatever files the invoking user can already read; local access
  is assumed, not elevated.
- `verify` executes a command you explicitly pass, with the privileges of the
  invoking user. Treat `AgentABI.verify` with untrusted argv the same way you
  would treat `subprocess.run` with untrusted argv.

## For contributors

- Never commit secrets, tokens, passwords, `.env*` files, or personal data.
- Never include real repositories, customer code, or personal information in
  issues, test fixtures, or screenshots.
- If you believe a secret was committed by mistake, rotate the credential
  immediately and contact maintainers; do not open a public issue.
