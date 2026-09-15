# OpenFOAM Agent v5.2.3

v5.2.3 is a focused pre-commit repair-context hotfix on top of v5.2.2.

## What changed

- Retained candidate repair now carries the bounded controller-owned deterministic failure diagnostic, not only the failed path/artifact.
- `case_bundle_preflight` failures caused by unsafe/executable OpenFOAM directives are classified in the replan capsule as security-policy failures while the existing fail-closed workspace policy remains unchanged.
- Candidate replan prompts explicitly require preserving the frozen CFD/BC intent with ordinary declarative OpenFOAM syntax; the model must not weaken, bypass, encode around, or reproduce the prohibited construct.
- The latest candidate failure diagnostic is checkpointed with the retained candidate and cleared after successful transactional preflight/commit.
- Candidate-replan partition metrics report the visible deterministic diagnostic size.

## Regression target

The regression reproduces the observed 3-D cylinder-wake failure mode:

1. candidate `0/U` contains `codedFixedValue` / `#codeStream`;
2. deterministic workspace preflight rejects the bundle before commit;
3. the replan capsule contains both `0/U` and the exact unsafe-directive diagnostic;
4. a replacement of only `0/U` with ordinary declarative `fixedValue` syntax passes the same workspace content policy.

The security allowlist itself was not relaxed.
