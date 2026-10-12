# Security Policy

## Scope

`dreamforge` runs simulation code and loads model checkpoints. It does not
expose a network service and does not execute untrusted input by design.
Relevant risks:

- **Checkpoint loading** (`torch.load`) executes pickle data. Only load
  checkpoints you trust (prefer `weights_only=True` where supported).
- **Simulator environments** (LIBERO / robosuite / MuJoCo) render offscreen and
  may write into the configured `runs/` directory.
- **HTML dashboard** embeds run numbers; it contains no JavaScript and loads no
  external resources, so it is safe to open offline.

## Reporting a vulnerability

Open a private security advisory on the repository (GitHub → Security →
Advisories) or contact the maintainer directly. Please include:

- a description of the issue and its impact,
- a minimal reproduction (command, config, and the artifact produced),
- any suggested remediation.

We aim to acknowledge reports within a few days and to credit reporters unless
they prefer otherwise.

## Supported versions

The `main` branch is the supported version. This is research code and no
long-term support is promised.
