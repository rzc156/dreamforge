# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project aims to
follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- `dashboard` command: a self-contained HTML run report (inline CSS + SVG
  charts, no CDN or JavaScript) for offline viewing and GitHub Pages.
  The page is **bilingual (English / 中文)** with a CSS-only switch — hidden
  radio inputs and sibling selectors, so it still ships no script.
- `diagnose` command: world-model pre-flight check for action response
  (reproduces the "action vacancy" phenomenon).
- `build --max-decision-step`: restrict corrections to early (recoverable)
  states.
- GitHub Pages workflow publishing the dashboard, issue/PR templates,
  contribution and security docs.
- Bilingual README (English / 中文), banner and pipeline diagrams.
- Curated `results/` artifacts (report, figures, paired records, evaluations).

### Changed
- CEM now uses a **per-dimension trust region** (0.5σ of the dataset action
  std) instead of a uniform bound; the `random` control uses the same budget.
- Mining stores the dataset-space (un-normalized) state; legacy picks are
  de-normalized from the policy statistics.
- Checkpoints are saved together with their processor pipelines.
- Reports include an episode-clustered summary and a budget comparison.

### Fixed
- Vector-env auto-reset semantics replaced with robust raw stepping.
- LIBERO episode caps follow the standard suite limits (280/300/400/520)
  instead of robosuite's default 1000.
- Observation batching matches the vector-env format expected by the
  pre-processors.
- `build` overwrites previous shards instead of appending to them.

## [0.1.0] - 2026-10-12

### Added
- Initial pipeline: `mine`, `imagine`, `verify`, `build`, `train`, `eval`,
  `report`, `figures` plus statistics (Wilson, exact McNemar), unit tests and
  CI, and the first full LIBERO-Spatial run (360 failure states, 1080 paired
  replays, fine-tuning comparisons).
