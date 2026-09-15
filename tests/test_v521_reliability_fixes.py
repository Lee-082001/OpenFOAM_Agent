from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from conftest import FakeOpenFOAMTools, foam_header, make_plan, make_state
from test_v510_entry_level_repair import _approved
from test_v510_requirement_assurance import _typed_intake, _workspace, _plan
from openfoam_agent.contracts.execution import ExecutionApproval
from openfoam_agent.engineering.case_build_graph import compile_case_build_graph
from openfoam_agent.engineering.case_delta_graph import compile_case_delta_graph
from openfoam_agent.schemas.engineering import (
    CaseBundleFile, CaseContentAssertion, ConfirmedFactBinding, ExecuteCasePlanAction,
    NativeOpenFOAMCommand, NumericEvidenceTerm, NumericRelationAssertion,
)
from openfoam_agent.qualification.reliability import BenchmarkTrialSpec, evaluate_report
from openfoam_agent.tools.workspace import CaseWorkspace
from openfoam_agent.verification.safety import DeterministicSafetyGate


def velocity_workspace(tmp_path, velocity=1):
    ws = _workspace(tmp_path)
    ws.write_text('0/U', foam_header('0/U', 'volVectorField') +
                  'dimensions [0 1 -1 0 0 0 0];\ninternalField uniform (2 0 0);\n'
                  f'boundaryField {{ inlet {{ type fixedValue; value uniform ({velocity} 0 0); }} }}\n')
    return ws


def check_velocity(ws, binding):
    intake = _typed_intake()
    return DeterministicSafetyGate(FakeOpenFOAMTools(), ws).validate_plan(_plan(intake, [binding]), intake)


@pytest.mark.parametrize('assertion', [
    CaseContentAssertion(path='0/U', anchor='internalField uniform'),
    CaseContentAssertion(path='0/U', entry_path='boundaryField.inlet.value', expected_value='uniform (1 0 0)'),
])
def test_v521_text_or_model_expected_value_cannot_override_frozen_target(tmp_path, assertion):
    binding = ConfirmedFactBinding(fact_id='boundary.inlet_velocity', case_assertions=[assertion])
    assert not check_velocity(velocity_workspace(tmp_path), binding).valid


@pytest.mark.parametrize('entry,multiplier', [('internalField', 1), ('boundaryField.inlet.value', 2)])
def test_v521_wrong_patch_or_arbitrary_multiplier_is_not_proof(tmp_path, entry, multiplier):
    binding = ConfirmedFactBinding(fact_id='boundary.inlet_velocity', numeric_relation=NumericRelationAssertion(
        numerator=[NumericEvidenceTerm(path='0/U', entry_path=entry, multiplier=multiplier)]))
    assert not check_velocity(velocity_workspace(tmp_path), binding).valid


def test_v521_actual_patch_value_matches_frozen_quantity(tmp_path):
    binding = ConfirmedFactBinding(fact_id='boundary.inlet_velocity', numeric_relation=NumericRelationAssertion(
        numerator=[NumericEvidenceTerm(path='0/U', entry_path='boundaryField.inlet.value')]))
    result = check_velocity(velocity_workspace(tmp_path, 2), binding)
    assert result.valid, result.failures


def test_v521_duplicate_boundary_entry_is_not_quantitative_proof(tmp_path):
    ws = velocity_workspace(tmp_path, 2)
    ws.write_text('0/U', ws.read_text('0/U').replace('value uniform (2 0 0);', 'value uniform (2 0 0); value uniform (1 0 0);'))
    binding = ConfirmedFactBinding(fact_id='boundary.inlet_velocity', numeric_relation=NumericRelationAssertion(
        numerator=[NumericEvidenceTerm(path='0/U', entry_path='boundaryField.inlet.value')]))
    assert not check_velocity(ws, binding).valid


def test_v521_value_only_boundary_is_not_a_fixed_boundary_proof(tmp_path):
    ws = velocity_workspace(tmp_path, 2)
    ws.write_text('0/U', ws.read_text('0/U').replace('type fixedValue;', 'type zeroGradient;'))
    binding = ConfirmedFactBinding(fact_id='boundary.inlet_velocity', numeric_relation=NumericRelationAssertion(
        numerator=[NumericEvidenceTerm(path='0/U', entry_path='boundaryField.inlet.value')]))
    assert not check_velocity(ws, binding).valid


@pytest.mark.parametrize('value', ['-1', '0', '1000000', 'notANumber', 'nan', 'inf'])
def test_v521_invalid_or_increased_timestep_rejected_without_authorization(tmp_path, value):
    ws, plan, approval = _approved(tmp_path)
    original = ws.read_text('system/controlDict')
    with pytest.raises(ValueError):
        approval.check_repair_delta(ws, {'system/controlDict': original.replace('deltaT 0.01;', f'deltaT {value};')})
    assert ws.read_text('system/controlDict') == original
    assert approval.authorized_repair_hashes == {}


def test_v521_approved_reduction_survives_checkpoint_and_rejects_increase(tmp_path):
    ws, plan, approval = _approved(tmp_path)
    changed = ws.read_text('system/controlDict').replace('deltaT 0.01;', 'deltaT 0.005;')
    approval.check_repair_delta(ws, {'system/controlDict': changed})
    approval.check_repair_delta(ws, {'system/controlDict': changed})
    ws.write_text('system/controlDict', changed)
    approval = ExecutionApproval.model_validate_json(approval.model_dump_json())
    approval.check_repair_files(ws.seal(plan))
    with pytest.raises(ValueError):
        approval.check_repair_delta(ws, {'system/controlDict': changed.replace('deltaT 0.005;', 'deltaT 0.006;')})


def test_v521_protected_leaf_allows_unrelated_numerical_edit(tmp_path):
    ws, plan, _ = _approved(tmp_path)
    text = foam_header('system/fvSolution') + 'solvers { p { tolerance 1e-6; relTol 0.1; } }\nrelaxationFactors { fields { p 0.3; } }\n'
    ws.write_text('system/fvSolution', text)
    plan = plan.model_copy(update={'confirmed_fact_bindings': [ConfirmedFactBinding(
        fact_id='objective.primary', case_assertions=[CaseContentAssertion(
            path='system/fvSolution', entry_path='solvers.p.tolerance', expected_value='1e-6')])]})
    approval = ExecutionApproval.issue(plan, ws.seal(plan))
    changed = text.replace('p 0.3;', 'p 0.2;')
    approval.check_repair_delta(ws, {'system/fvSolution': changed})
    ws.write_text('system/fvSolution', changed)
    approval.check_repair_files(ws.seal(plan))
    with pytest.raises(ValueError, match='frozen requirement'):
        approval.check_repair_delta(ws, {'system/fvSolution': changed.replace('1e-6', '1e-4')})


def test_v521_failed_multifile_candidate_leaves_no_partial_authorization(tmp_path):
    ws, _, approval = _approved(tmp_path)
    changed = ws.read_text('system/controlDict').replace('deltaT 0.01;', 'deltaT 0.005;')
    with pytest.raises(ValueError):
        approval.check_repair_delta(ws, {'system/controlDict': changed, 'constant/physicalProperties': 'nu 1;'})
    assert approval.authorized_repair_hashes == {}
    assert approval.repair_baselines == {}


def test_v521_postseal_requires_exact_authorized_numerical_hash(tmp_path):
    ws, plan, approval = _approved(tmp_path)
    original = ws.read_text('system/fvSolution')
    approval.check_repair_delta(ws, {'system/fvSolution': original + 'relaxationFactors {}\n'})
    ws.write_text('system/fvSolution', original + 'unexpected 1;\n')
    with pytest.raises(ValueError):
        approval.check_repair_files(ws.seal(plan))


def mesh_workspace(tmp_path):
    plan = make_plan(make_state().intake)
    plan.required_case_files = ['system/controlDict', 'system/blockMeshDict', 'system/snappyHexMeshDict', 'system/surfaceFeaturesDict', 'system/topoSetDict']
    ws = CaseWorkspace(tmp_path)
    for path in plan.required_case_files:
        ws.write_text(path, foam_header(path) + 'testValue 1;\n')
    return plan, ws


@pytest.mark.parametrize('changed,expected', [
    ('system/blockMeshDict', ['blockMesh', 'snappyHexMesh', 'checkMesh']),
    ('system/surfaceFeaturesDict', ['surfaceFeatures', 'snappyHexMesh', 'checkMesh']),
])
def test_v521_delta_rebuilds_transitive_consumers(tmp_path, changed, expected):
    plan, ws = mesh_workspace(tmp_path)
    graph = compile_case_delta_graph(ws, plan, replacements=[(changed, ws.read_text(changed).replace('1;', '2;'))])
    assert graph.valid, graph.failures
    assert [item.command for item in graph.native_pipeline] == expected


def test_v521_inferred_mesh_precedes_explicit_toposet(tmp_path):
    plan, ws = mesh_workspace(tmp_path)
    files = [CaseBundleFile(path=path, content=ws.read_text(path)) for path in plan.required_case_files]
    action = ExecuteCasePlanAction(type='execute_case_plan', goal='mesh', plan=plan, files=files,
                                  native_pipeline=[NativeOpenFOAMCommand(command='topoSet', role='mesh')])
    graph = compile_case_build_graph(action, {item.path: item.content for item in files})
    assert graph.valid, graph.failures
    commands = [item.command for item in graph.native_pipeline]
    assert commands.index('blockMesh') < commands.index('snappyHexMesh') < commands.index('topoSet')


def test_v521_explicit_dict_does_not_add_default_second_invocation(tmp_path):
    plan, ws = mesh_workspace(tmp_path)
    custom = 'system/customBlockMeshDict'
    files = [CaseBundleFile(path=path, content=ws.read_text(path)) for path in plan.required_case_files]
    files.append(CaseBundleFile(path=custom, content=foam_header(custom)))
    action = ExecuteCasePlanAction(type='execute_case_plan', goal='mesh', plan=plan, files=files,
                                  native_pipeline=[NativeOpenFOAMCommand(command='blockMesh', arguments=['-dict', custom], role='mesh')])
    graph = compile_case_build_graph(action, {item.path: item.content for item in files})
    assert graph.valid, graph.failures
    blocks = [item for item in graph.native_pipeline if item.command == 'blockMesh']
    assert len(blocks) == 1 and blocks[0].arguments == ['-dict', custom]


@pytest.mark.parametrize('success', [False, None, 'false'])
def test_v521_runtime_failure_or_nonboolean_success_is_not_repair_success(success):
    spec = BenchmarkTrialSpec(trial_id='repair', category='runtime', prompt='repair', expected_outcome='repair_then_accept')
    result = evaluate_report(spec, 'ours', {'final_state': 'SOLVE_READY', 'runtime_report': {'success': success},
                             'reliability_observables': {'runtime_repair_attempts': 1}})
    assert result.repair_success is False


def test_v521_presolve_repair_stage_must_be_declared():
    spec = BenchmarkTrialSpec(trial_id='repair', category='pre_solve', prompt='repair',
                             expected_outcome='repair_then_accept', repair_success_stage='pre_solve')
    result = evaluate_report(spec, 'ours', {'final_state': 'SOLVE_READY',
                             'reliability_observables': {'runtime_repair_attempts': 1}})
    assert result.repair_success is True


def test_v521_second_repair_candidate_can_be_checked_twice(tmp_path):
    ws, plan, approval = _approved(tmp_path)
    first = ws.read_text('system/controlDict').replace('deltaT 0.01;', 'deltaT 0.005;')
    approval.check_repair_delta(ws, {'system/controlDict': first})
    ws.write_text('system/controlDict', first)
    second = first.replace('deltaT 0.005;', 'deltaT 0.002;')
    approval.check_repair_delta(ws, {'system/controlDict': second})
    approval.check_repair_delta(ws, {'system/controlDict': second})
    ws.write_text('system/controlDict', second)
    approval.check_repair_files(ws.seal(plan))


@pytest.mark.parametrize('action_kind', ['runtime', 'legacy'])
def test_v521_both_runtime_action_routes_block_before_commit(tmp_path, graph_path, action_kind):
    from conftest import ScriptedLLM
    from openfoam_agent.engineering import CFDEngineeringAgent
    from openfoam_agent.engineering.phases.repair_controller import execute_runtime_repair_plan
    from openfoam_agent.schemas.engineering import RuntimeCaseRepairAction, RepairCasePlanAction
    ws, plan, approval = _approved(tmp_path)
    plan.required_case_files = list(ws.list_authored())
    approval = ExecutionApproval.issue(plan, ws.seal(plan))
    state = make_state()
    state.engineering_plan = plan
    state.execution_approval = approval
    agent = CFDEngineeringAgent(ScriptedLLM([]), workspace=tmp_path, capability_db=graph_path)
    # Use the same tracked authored files, not another empty workspace view.
    agent.workspace = ws
    original = ws.read_text('system/controlDict')
    replacements = [CaseBundleFile(path='system/controlDict', content=original.replace('endTime 10;', 'endTime 100;'))]
    if action_kind == 'runtime':
        action = RuntimeCaseRepairAction(type='repair_runtime_case', diagnosis='repair', replacement_files=replacements)
    else:
        action = RepairCasePlanAction(type='repair_case_plan', diagnosis='repair', replacement_files=replacements)
    outcome = execute_runtime_repair_plan(agent, state, action, approved_solver=plan.solver,
                                         llm_step=1, native_execution=False, runtime_event_start=0)
    assert outcome is not None and not outcome.retry
    assert ws.read_text('system/controlDict') == original
    assert not state.revalidation_records


def test_v521_typed_unit_conversion_uses_controller_scale(tmp_path):
    from openfoam_agent.contracts.models import PhysicalQuantity
    ws = _workspace(tmp_path)
    ws.write_text('constant/geometryProperties', foam_header('constant/geometryProperties') + 'length [0 1 0 0 0 0 0] 0.002;\n')
    intake = _typed_intake()
    fact = intake.fact('boundary.inlet_velocity')
    fact.category = 'geometry'
    fact.value = '2'
    fact.unit = 'mm'
    fact.quantity = PhysicalQuantity(quantity='length', value=2, unit='mm', dimensions=(0,1,0,0,0,0,0))
    binding = ConfirmedFactBinding(fact_id=fact.id, numeric_relation=NumericRelationAssertion(
        numerator=[NumericEvidenceTerm(path='constant/geometryProperties', entry_path='length')]))
    from openfoam_agent.schemas.engineering import EngineeringPlan
    data = _plan(intake).model_dump()
    data['required_case_files'].append('constant/geometryProperties')
    data['confirmed_fact_bindings'] = [binding]
    plan = EngineeringPlan.model_validate(data)
    result = DeterministicSafetyGate(FakeOpenFOAMTools(), ws).validate_plan(plan, intake)
    assert result.valid, result.failures


def test_v521_dimension_mismatch_cannot_be_hidden_by_equal_numbers(tmp_path):
    ws = velocity_workspace(tmp_path, 2)
    ws.write_text('0/U', ws.read_text('0/U').replace('[0 1 -1 0 0 0 0]', '[0 2 -2 0 0 0 0]'))
    binding = ConfirmedFactBinding(fact_id='boundary.inlet_velocity', numeric_relation=NumericRelationAssertion(
        numerator=[NumericEvidenceTerm(path='0/U', entry_path='boundaryField.inlet.value')]))
    assert not check_velocity(ws, binding).valid


def test_v521_vector_speed_uses_magnitude_not_one_favorable_component(tmp_path):
    ws = velocity_workspace(tmp_path, 2)
    ws.write_text('0/U', ws.read_text('0/U').replace('value uniform (2 0 0)', 'value uniform (2 2 0)'))
    binding = ConfirmedFactBinding(fact_id='boundary.inlet_velocity', numeric_relation=NumericRelationAssertion(
        numerator=[NumericEvidenceTerm(path='0/U', entry_path='boundaryField.inlet.value')]))
    assert not check_velocity(ws, binding).valid


def test_v521_summary_text_alone_does_not_count_as_repair():
    spec = BenchmarkTrialSpec(trial_id='repair', category='runtime', prompt='repair', expected_outcome='repair_then_accept')
    result = evaluate_report(spec, 'ours', {'final_state': 'COMPLETE', 'runtime_report': {'success': True},
                             'engineering_events': [{'success': True, 'summary': 'No repair was needed.'}]})
    assert not result.repair_attempted and result.repair_success is False


def test_v521_missing_results_have_explicit_expected_denominator(tmp_path):
    from openfoam_agent.qualification.reliability import BenchmarkManifest, evaluate_results_directory
    spec = BenchmarkTrialSpec(trial_id='case', category='runtime', prompt='run', expected_outcome='accept')
    manifest = BenchmarkManifest(release='5.2.1', variants=['ours'], trials=[spec])
    result = evaluate_results_directory(manifest, tmp_path)
    assert result['summary_by_variant']['ours']['missing'] == 1
    assert result['summary_by_variant']['ours']['coverage_rate'] == 0
    assert result['summary_by_variant']['ours']['execution_success_over_expected'] == 0
    assert not result['complete']


def test_v521_release_metadata_and_docker_copy_sources():
    import tomllib
    import openfoam_agent
    root = Path(__file__).resolve().parents[1]
    assert openfoam_agent.__version__ == tomllib.loads((root / 'pyproject.toml').read_text())['project']['version']
    assert (root / 'V5_2_1_RELEASE.md').is_file()
    for line in (root / 'docker/Dockerfile').read_text().splitlines():
        if line.startswith('COPY '):
            for source in line.split()[1:-1]:
                assert (root / source).exists(), source


@pytest.mark.parametrize('mutation', ['missing_phase', 'duplicate_report', 'changed_source', 'missing_node'])
def test_v521_evidence_verifier_rejects_incomplete_or_stale_results(tmp_path, mutation):
    import hashlib
    import runpy
    root = Path(__file__).resolve().parents[1]
    verify = runpy.run_path(str(root / 'scripts/verify_regression_evidence.py'))['verify']
    source = tmp_path / 'src' / 'sample.py'
    source.parent.mkdir()
    source.write_text('value = 1\n')
    node = 'tests/test_sample.py::test_sample'
    reports = [{'nodeid': node, 'phase': phase, 'outcome': 'passed'} for phase in ['setup','call','teardown']]
    data = {'release': '5.2.1', 'collected': [node], 'reports': reports, 'collection_errors': [],
            'exit_code': 0, 'source_hashes': {'src/sample.py': hashlib.sha256(source.read_bytes()).hexdigest()}}
    assert verify(data, tmp_path)['all_passed']
    if mutation == 'missing_phase':
        reports.pop()
    elif mutation == 'duplicate_report':
        reports.append(reports[-1].copy())
    elif mutation == 'changed_source':
        source.write_text('value = 2\n')
    else:
        data['collected'].append('tests/test_sample.py::test_other')
    assert not verify(data, tmp_path)['all_passed']
