# OpenFOAM Agent 5.2.1 변경·검증 인수인계

첨부받은 v5.2.0 전체 프로젝트를 복사한 뒤 직접 수정한 패치 버전이다.
원본 ZIP SHA-256: `19fdea58031e571777d2555ff1df6f13d549c45bd707e8cde656c91f930cd74b`.

## 적용한 수정

| 영역 | 수정 내용 | 주요 파일 |
|---|---|---|
| 수치 요구 검증 | 사용자 확정 목표와 실제 값을 비교. anchor만 있는 증거, 임의 expected_value/배율로 통과시키는 경로 차단 | verification/safety.py |
| 단위·대상 | SI 변환은 controller 단위표 사용. field 차원과 확정 region/patch 확인. 고정 경계 속도는 벡터 크기로 비교 | verification/safety.py |
| Mesh 후속 작업 | 변경 산출물까지 따라가며 후속 소비자를 추가. blockMesh 재생성 후 snappy 누락, feature dictionary 수정 후 추출 누락 수정 | contracts/execution_scopes.py, engineering/case_delta_graph.py |
| 초기 순서 | 기본 mesh·snappy 이후 topoSet 등 mesh 소비자 배치. 명시한 -dict가 있으면 기본 명령을 중복 추가하지 않음 | engineering/case_build_graph.py, data/native_tool_contracts.json |
| Runtime 수정 | 양수·유한 수치·감소 방향·최초 상한 검사. fvSolution/fvSchemes는 보호 leaf만 비교. 다중 파일 승인 결과 일괄 반영 | contracts/execution.py |
| 수정 전 차단 | 새 runtime 요청과 legacy repair 요청 모두 commit 전에 같은 검사 적용. 승인된 파일 hash와 적용 결과 일치 확인 | engineering/phases/repair_controller.py |
| 평가 | runtime 복구 성공에는 실제 runtime 성공 필요. pre-solve 성공 단계는 별도 명시. 누락 보고서와 전체 지정 사례 분모 노출 | qualification/reliability.py |
| 설치·회귀 | 없는 Docker COPY 파일 제거. 과거 33개 audit 파일에 의존하지 않는 전체 pytest 실행·증거 검사기 추가 | docker/Dockerfile, scripts/ |

소스 경로는 `src/openfoam_agent/` 기준이다. 특정 문제에 solver/BC를 강제하는 레시피는 추가하지 않았다.
도구의 입출력 관계, 물리 차원, 허용한 수치 변경 규칙은 명시적 실행 계약으로 관리한다.

## 확인한 결과

- 신규 5.2.1 회귀: 38개 통과.
- 이번 수정과 직접 관련된 기존 테스트를 포함한 집중 검사: 62개 통과.
- 전체 pytest: 712개 수집, **615개 통과 / 97개 실패 / skip 0개**.
- 같은 환경의 v5.2.0: 571개 통과 / 103개 실패.
- 이번 전체 실행의 실패 97개는 모두 이전 실패 목록에 있던 항목이다.
- wheel 빌드, 의존성 설치를 생략한 별도 경로 wheel 설치, 설치된 패키지의 CLI `--help` 통과.
- Python 3.12.14 / pytest 9.1.1 / Pydantic 2.13.5에서 확인.
- **OpenFOAM, MPI, Docker build, 실제 LLM 호출 및 물리 결과 정확도는 미검증**이다.

전체가 green인 릴리스는 아니다. 남은 실패는 `KNOWN_REGRESSION_FAILURES.md`에 모두 기재했다.
실패를 일괄 skip/xfail 처리하거나 테스트를 삭제해서 통과 수를 늘리지 않았다.
6개 기존 실패는 현재 버전 일치, surfaceFeatures 허용, 필수 수치 요구 검사,
실제 미사용 import와 오래된 주석 등 변경된 계약·정리 사항을 반영해 해결했다.
수치 검증의 기존 positive/drift fixture는 문자 조각과 임의 배율 대신 명시적 entry와 차원을 사용하도록 갱신했다.

## 재현 명령

프로젝트 루트에서 실행한다.

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -e '.[dev]'
sh scripts/run_regression.sh
```

현재 알려진 실패 때문에 마지막 명령의 종료 코드는 0이 아니다. 결과는 `verification-local/`에 남는다.
동봉한 실제 실행 결과는 `verification-local/v5.2.1/` 아래에 있다.
`regression.json`에는 수집·실행 node ID, 각 단계 결과, 코드/테스트/스크립트/config hash가 들어 있다.
`summary.json`의 `all_passed=false`, `native_cfd_qualified=false`를 유지한다.

새 반례만 빠르게 확인하려면:

```bash
python3 -m pytest -q tests/test_v521_reliability_fixes.py
```

벤치마크는 여전히 보고서 집계기이며 자동 CFD 실행기나 독립 기준해 판정기가 아니다.
새 manifest는 `research/V5_2_1_RELIABILITY_BENCHMARK.json`이고 NOT_RUN 상태다.
기존 v5.2.0 manifest는 이력을 위해 보존했다.

## 호환성·한계

1. 구 checkpoint는 새 필드를 기본값으로 읽는다. 단, 이미 수정됐지만 승인 hash가 없는 수치 파일에 대해
   과거 권한을 새로 만들어 주지는 않는다. 그런 상태는 검토·재검증 후 다시 seal해야 한다.
2. 기존 anchor 기반 수치 증거는 정확한 entry/차원으로 보완이 필요할 수 있다.
   동적 dictionary, 시간 의존 수치, fixedValue가 아닌 경계조건의 값 증명,
   임의 복합식은 지원 검증기가 없으면 원인을 표시하고 통과시키지 않는다.
3. Re 검사는 dimensioned U × reference length / nu의 입력 산술 검사다.
   **기준 길이가 실제 형상의 올바른 길이인지, 결과가 물리적으로 정확한지는 별도의 검증이 필요하다.**
   이 패치가 독립 geometry/CFD oracle를 완성한 것은 아니다.
4. 도구 registry가 설명하는 후속 관계를 개선했으며, 모든 native utility나 반복적인 in-place 변형의
   임의 조합을 완전히 지원한다고 주장하지 않는다. 복잡한 새 pipeline은 별도 native 검증이 필요하다.
5. 이번에는 97개 과거 회귀 실패 전체를 재설계하지 않았다. 우선 실패를 현재 계약과 유효한 동작 회귀로 분류하고,
   실제 OpenFOAM 단순/feature/named-region 케이스를 연결하는 작업이 다음 순서다.

소스·tests·config·examples·기존 문서가 포함된 전체 프로젝트 ZIP이며, 수정 파일만 묶은 패치는 아니다.
