# OpenFOAM Agent v3.5.0 changes

## Evidence projection contract hardening

- Fixed a workflow-fatal cardinality bug where a valid evidence batch with 25 observed items exceeded `EngineeringEvent.observed_evidence` (`max_length=24`) and was misreported as an evidence-retrieval infrastructure failure.
- `EngineeringEvidenceRecord.observed_evidence` is now durable structured state with no small arbitrary item ceiling; full deterministic evidence remains available to provenance validation and later context compilation.
- `EngineeringEvent` is explicitly a bounded progress/audit projection. The central `_event()` factory deterministically selects at most 24 descriptors, and the model also fail-soft normalizes oversized direct inputs before field validation.
- Large retrieval batches therefore cannot disable the evidence phase merely because the event/UI projection is smaller than the durable evidence set.
- Added regression coverage for 100-item durable evidence records, 100-item event inputs, and an end-to-end `gather_evidence` batch returning 100 capability observations.

## Audit

- Confirmed all production `EngineeringEvent` construction routes through the central bounded `_event()` factory.
- Audited Engineering event/record list cardinality constraints for the same durable-vs-projection mismatch; no additional Engineering progress payload store uses a small cardinality ceiling.
- Existing semantic/action list limits (region assignments, command arguments, case-plan fields) remain intentional contract bounds rather than progress-projection storage.
