from openfoam_agent.engineering.repair_context import diagnostic_literal_candidates, native_supported_alternatives
from openfoam_agent.schemas.engineering import RepairTurn

D = """Unknown transport type constIso\nSupported transport types:\n5\n(\nconstIsoSolid\npolynomialSolid\nexponentialSolid\nconstAnisoSolid\ntabulatedSolid\n)\nFOAM exiting\n"""

def test_native_alternatives_are_projected_not_chosen():
    assert diagnostic_literal_candidates(D) == ["constIso"]
    assert native_supported_alternatives(D) == ["constIsoSolid", "polynomialSolid", "exponentialSolid", "constAnisoSolid", "tabulatedSolid"]

def test_prepare_repair_schema_stays_compact():
    schema = str(RepairTurn.model_json_schema())
    assert "updated_plan" not in schema
    assert "plan_patch" not in schema
