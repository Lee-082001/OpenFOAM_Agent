# v2 Evaluation Plan

The primary v2 question is no longer whether a deterministic capability-gap planner selects the expected route. It is whether an engineering agent can use bounded OpenFOAM tools autonomously while deterministic code preserves safety and evidence.

Suggested evaluation dimensions:

1. **Engineering autonomy**: fraction of unseen CFD tasks reaching a valid case without a task-specific Python template.
2. **Tool-use correctness**: environment/reference/capability queries and native validation actions used appropriately.
3. **Recovery**: mesh/dictionary/runtime failures repaired from real logs within bounded steps.
4. **User-fact preservation**: confirmed values unchanged across final plan and generated case.
5. **Version correctness**: use of release-matched installed source/tutorial evidence for version-sensitive syntax.
6. **Safety-gate precision**: unsafe paths/code/commands/tampering are blocked without encoding CFD design preferences.
7. **Approval integrity**: no solver execution before `/solve`; no silent solver switch during automatic repair.
8. **Scientific quality**: post-run reference comparison, conservation, force/field metrics, and grid/time-step sensitivity evaluated separately from runtime completion.

Recommended baselines:

- direct one-shot LLM case generation without tools;
- agent with tools but without capability/reference retrieval;
- v2 full engineering agent;
- task-specific template automation, reported separately as an automation baseline rather than an autonomous-agent baseline.
