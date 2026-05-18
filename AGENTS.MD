# Agent Instructions

This repository contains CloneRL / CloneLab. Treat it as an actively evolving framework used by a single primary user who already runs the newest version.

## Core Priorities

- Clarity comes before confidence. Do not make strong claims before checking the relevant code, tests, or documentation.
- Clarity comes before momentum. If the requirement is ambiguous, stop and ask before implementing.
- When asking for clarification, provide the best proposed solution or direction so the user can confirm or correct it quickly.
- Be direct and factual. Avoid sycophantic encouragement, performative enthusiasm, and vague reassurance.

## Backward Compatibility

- Do not maintain old code paths for backward compatibility by default.
- Breaking changes are acceptable when they improve the framework and the relevant tests pass.
- Update current call sites and documentation when making a breaking change.
- Do not add deprecation layers, compatibility shims, legacy aliases, fallback paths, migration code, or old API support unless explicitly requested.
- Prefer removing obsolete paths over preserving them.

## Implementation Style

- Follow the existing structure and conventions of the repository.
- Keep changes scoped to the requested work.
- Prefer simple, current implementations over abstractions designed for hypothetical future compatibility.
- If existing behavior appears obsolete or inconsistent with the current framework direction, raise it clearly and propose the clean replacement.

## Tests

- Add or update tests only when they are necessary for the change.
- Focus tests on behavior that is risky, shared, user-facing, or likely to regress.
- Run the relevant tests when practical before reporting completion.

## Git

- Leave changes unstaged unless the user explicitly asks otherwise.
- Do not create commits, push branches, or open pull requests unless explicitly requested.
- Do not revert user changes unless explicitly requested.
