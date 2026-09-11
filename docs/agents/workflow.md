# Workflow

How work gets from an idea to a merged branch in this repo.

## Planning

Planning follows the [Matt Pocock main flow](https://www.aihero.dev/skills): `/grill-with-docs`
→ `/to-spec` → `/to-tickets`. The spec and the tickets it produces are published as GitHub
issues — see [`issue-tracker.md`](issue-tracker.md), never as files under `docs/`.

## Branch and draft PR

Open the draft PR **when you start the first ticket** (or the first set of parallel tickets) of a
spec, not when the work is finished. The PR targets `main` and tracks the development branch from
its first commit, so review can follow the work as it lands. Link it to the originating issue.


Mark the PR ready for review once the verification gate in
[`AGENTS.md`](../../AGENTS.md) passes.

## Commits

Each realised phase of a ticket ends in its own commit, with a conventional-style message
(`feat:`, `fix:`, `docs:`, `chore:`, …). Keep the subject meaningful about the behaviour that
changed, not the files that moved.
