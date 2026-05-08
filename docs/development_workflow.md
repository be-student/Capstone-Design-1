# Development Workflow

This project uses **GitHub Flow**. The repository is a single deployable
capstone application with a Docker Compose entrypoint, so a lightweight
feature-branch workflow is simpler than long-lived Git Flow release and
develop branches.

## Branch Strategy

- `main` is the only permanent branch and represents the latest releasable
  state.
- All implementation work starts from a short-lived **feature branch** created
  from the current `main`.
- Feature branches use descriptive names such as
  `feature/uplift-report-balance-check`, `fix/monitoring-performance-alert`, or
  `docs/github-flow-policy`.
- Long-lived Git Flow branches such as `develop`, `release/*`, and `hotfix/*`
  are not used unless the project later needs multiple supported production
  release lines.

## Main Branch Protection

The `main` branch must be protected in GitHub repository settings:

- Direct pushes to `main` are blocked.
- Every change must arrive through a pull request.
- Pull requests require at least one review approval before merge.
- Required CI checks must pass before merge.
- Branches must be up to date with `main` before merge when GitHub reports a
  conflict or stale required check.

## Feature Branch Workflow

1. Sync the local repository with `main`.
   ```bash
   git checkout main
   git pull --ff-only origin main
   ```
2. Create a feature branch.
   ```bash
   git checkout -b feature/<short-description>
   ```
3. Keep commits scoped to one requirement or bug fix.
4. Run the relevant local verification before opening a pull request.
5. Push the branch and open a pull request into `main`.

## Pull Request Review

Each pull request must include:

- A concise summary of the behavior or documentation changed.
- The related requirement or blocker, when applicable.
- The commands used for verification, including tests, `rg` evidence, or Docker
  checks.
- Screenshots or artifact paths when dashboard or generated result behavior is
  affected.

Reviewers should check that the change stays within its intended scope, does not
overwrite unrelated work, and keeps generated artifacts consistent with the
documented requirements.

## CI Policy

CI is required for pull requests into `main`. The required checks are:

- Python tests for the touched area, with the full test suite required for broad
  pipeline, dashboard, or data-contract changes.
- Static or formatting checks configured for the repository.
- Docker Compose configuration validation when container files or runtime
  entrypoints change.

The pull request cannot merge while required CI checks are failing.

## Merge Policy

- Use squash merge for ordinary feature branches so `main` keeps a compact,
  requirement-oriented history.
- Use a regular merge commit only when preserving a multi-commit branch history
  is necessary for auditability.
- Delete the feature branch after merge.
- Do not merge unrelated source, test, documentation, and generated-result
  changes in one pull request unless they are required by the same blocker.

## Release and Tag Policy

This project releases from `main` after the required pipeline and documentation
evidence pass.

- Create a Git tag for each submitted release candidate using
  `v<major>.<minor>.<patch>` or a capstone cycle tag such as `v1.0.0`.
- Tag only commits already merged into `main`.
- The tag description should list the verification commands and key submitted
  artifacts, including `results/required_artifacts_checklist.json` when present.
- Patch tags are used for blocker fixes that do not change the public execution
  contract.

