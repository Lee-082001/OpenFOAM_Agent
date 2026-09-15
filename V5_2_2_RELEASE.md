# OpenFOAM Agent v5.2.2

Hotfix release based on v5.2.1. This release closes the runtime result-output and
human-acceptance mismatch observed in a completed two-solid battery/heater run.

## Changes

- Final result evidence is now explicit. `completion.required_result_fields` is the
  only authority for final-time field verification; solver inputs under `0/` are no
  longer inferred as required final outputs.
- Staged executable designs must declare non-empty result fields. Each declared field
  must correspond to a declared initial solution field, while termination or physical
  quality criteria may remain review-oriented where appropriate.
- Runtime output verification checks only the Agent-declared result fields. A case
  with `0/battery/p` and `0/heater/p` may therefore verify successfully when the
  declared final evidence is only `battery/T` and `heater/T`.
- Human result acceptance now has a single deterministic `result_acceptance_blockers`
  gate shared by state and CLI behavior. Incomplete termination, non-clean native
  execution, or unverified required output artifacts cannot be overridden by `/accept`.
- Interactive `/accept` is fail-safe. A blocked acceptance prints the blockers and
  returns to the session instead of propagating `ValueError` and terminating the CLI.
- Reports and interactive guidance no longer advertise `/accept` when deterministic
  result evidence is incomplete.
- Engineering-plan revisions cannot clear an existing explicit result-artifact
  contract or leave declared result fields detached from their initial solution
  artifacts.

## Regression target

The primary regression reproduces the observed battery/heater pattern: the native
solver reaches `Time = 300` and exits with status zero, while pressure fields exist as
initial solver inputs. When the Agent declares only `battery/T` and `heater/T` as
required final evidence, those two fresh finite final fields are sufficient for output
verification; final pressure files are not required. Conversely, missing declared
result evidence keeps acceptance blocked without crashing the interactive CLI.

## Compatibility and limits

Legacy direct `EngineeringPlan` objects without an explicit result-output contract are
still loadable, but the runtime compiler no longer guesses final outputs from `0/`.
Such plans therefore remain output-unverified until an explicit result contract is
provided. New staged designs fail closed before sealing when result fields are absent.
Human acceptance remains a review acknowledgement, not proof of grid/time-step
independence or experimental validation.
