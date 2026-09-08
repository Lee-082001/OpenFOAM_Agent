# OpenFOAM Agent v4 구현·검증 보고서

**배포 버전: 4.0.0rc1 · 작성일: 2026년 9월 7일 · 판정: native CFD 검증 전 후보판**

## 1. 결론과 배포 범위

v3.6.0 원본을 보존한 별도 작업 사본에서 v4 계약·실행·검증 코드를 수정하고, 사용자 감사 원문의 **33개 항목을 빠짐없이 구현 경로와 실제 통과한 테스트에 대조**했다. 그러나 **33개 전체 완료 또는 실환경 검증 완료로 판정하지 않는다.** 재현된 소프트웨어 결함 6개는 아래에 한정한 검사 범위에서 수정 확인으로 분류했고, 나머지 27개는 부분 구현 또는 추가 검증이 필요하다. 이 숫자는 CFD 성공률이 아니다.

기준본은 `OpenFOAM_Agent_v3.6.0_bounded_staged_context.zip`이며 SHA256은 다음과 같다.

```text
3d9d99c8b186db5ed429ac36a6c49104052ae2b0c5c4b25b73e61a3b7eefb9ac
```

감사 원문은 `붙여넣은 텍스트 (1)(8).txt`이다. 원문에 적힌 WSL 커밋 `c11e4c4`와 현재 저장소가 같다고 가정하지 않았다. **패치 기준은 위 ZIP의 실제 바이트**다. `docs/AUDIT_SOURCE_33_KO.txt`에 감사 원문을 그대로 보존하고, `docs/V4_PROVENANCE.json`에 원본 해시와 작업 출처를 기록했다.

접근 가능한 작업 폴더와 Library의 명칭·내용 검색에서는 이전 v4 수정본을 찾지 못했다. 이를 모든 외부 저장소에 v4가 없다는 뜻으로 확대하지 않는다. 원본 파일은 변경하지 않았다. 작업 중 만든 `v4_work_before_final_verification.zip`은 이번 작업의 중간 백업이며 이전 대화에서 복구한 v4가 아니다.

## 2. 실제 실행한 검사

| 검사 | 관측 결과 | 보장하지 않는 것 |
|---|---|---|
| 수정 전 v3.6.0 기준본 | 298 passed | 실제 OpenFOAM 해석 성공 |
| v4 기존 회귀 묶음 | 298 passed, 25.33초 | 원래 기대값을 전혀 바꾸지 않았다는 주장 |
| v4 신규 감사 경계 묶음 | 61 passed, 7.91초 | 실제 MPI/CFD/backend 연동 |
| 전체 수집 ID와 두 묶음의 합집합 | 359개 정확히 일치, 중복·누락·실패·skip 0 | 한 번의 pytest 실행이 끝났다는 주장 |
| Python compileall | 통과 | 모든 타입·실행 경로의 증명 |
| 합성 실행 파일을 이용한 실제 subprocess | 승인, 실행 수 예산, timeout, CPU 상한, 로그 상한·원문 보존 검사 통과 | OpenFOAM 바이너리 실행, 메모리·전체 디스크·MPI 합산 quota |
| wheel 빌드 및 별도 venv 설치 | 소스 밖 CLI `--help`, 13/14 config 로딩 확인 | 깨끗한 의존성 설치 및 live backend |
| 패치·ZIP 검증 | 별도 `OpenFOAM_Agent_v4_package_verification.json`과 SHA256SUMS 참조 | 사용자의 수정된 Git worktree에 무충돌 적용 |

모든 테스트를 한 번에 실행한 시도는 도구 실행 제한으로 끝까지 완료되지 않았다. 최종 성공 근거는 **두 번의 분할 실행 JUnit XML**이다. `verification/test_summary.json`에서 수집한 전체 ID와 XML 결과를 대조했고, `verification/all_tests.xml`은 두 성공 결과를 합친 기록이다. 미완료 단일 실행 로그를 전체 통과 근거로 사용하지 않았다.

검사 환경은 Linux x86_64, Python 3.13.5다. `foamRun`, `foamMultiRun`, `blockMesh`, `checkMesh`, `foamDictionary`, `foamPostProcess`, `mpirun`, `mpiexec`, Codex/Claude/Ollama CLI가 발견되지 않았다. 실제 CFD, 실제 병렬 계산, 실제 모델 API 호출은 **0회**다. 테스트 이름에 native가 있거나 실행 파일 이름이 `foamRun`/`checkMesh`여도 이번 신규 실행 경계 검사는 합성 fixture일 수 있으므로 native CFD 검증으로 세지 않는다.

wheel은 `--no-deps`로 설치하고 이미 있는 부모 환경의 의존성을 `.pth`로 재사용했다. 해당 환경에는 OpenAI SDK도 없었다. 따라서 이번 설치 검사는 패키지 구성·CLI·resource smoke에 한정된다. Python 3.12, Windows/WSL, 새 환경에서의 의존성 해결 및 실제 인증·과금 경로는 검증하지 않았다.

### 기존 테스트를 바꾼 이유와 검토 범위

기존 테스트 파일 21개를 수정하고 신규 파일 3개를 추가했다. 기존 298개라는 수만 유지했다고 해서 수정 전과 기대값이 같다는 뜻은 아니다. v4에서는 성공 fixture에도 명시적 승인, 완료 계약, 목표 시각, 필요한 결과 파일을 넣었다. 무승인 실행을 허용하던 기대값은 거부로 바꿨으며, 설치 소스 발견을 실제 runtime 검증으로 표시하던 기대값도 낮췄다.

Codex fixture는 인증 종류·JSON 이벤트·usage 계약을 반영했다. staged 작성 fixture에는 실제 읽은 문법 근거를 넣었다. 필수 계약을 보존하면서 컨텍스트가 늘어 기존 토큰 기준 일부를 조정했다. `test_v360_bounded_staged_context.py`의 관련 한도는 15,000→18,000으로, `test_v210_token_optimization.py`의 관련 비율은 0.5→0.6으로 바뀌었다. **v4가 모든 경로에서 기존보다 토큰을 적게 쓴다고 주장하지 않는다.** 정확한 테스트 수정 내용은 배포 패치와 `verification/test_migration_manifest.json`에서 확인할 수 있다.

## 3. 주요 동작 변경

실행 승인은 더 이상 solver 이름 하나에 대한 허가가 아니다. 확정 intake, driver/module/영역 배치, argv, MPI 규모, 물리 구현, QoI, 자원 한도와 입력 seal을 묶는다. 설계·문법 수리를 이유로 다른 물리 문제나 다른 규모의 계산을 자동 허가하지 않는다. 검증 action 수와 실제 subprocess 수를 분리하고 캐시 재사용은 실제 입력과 환경 근거에 연결한다.

정상 종료와 계산 완료도 분리한다. `Time = 0`과 `End`만 있는 로그, 요청한 시간보다 짧은 계산, 이번 실행에서 갱신되지 않은 기존 결과 파일은 완료 근거가 아니다. 결과 파일 확인에 통과해도 수치 품질과 물리 목표 달성은 별도다. `/accept`는 미완료 계산을 완료로 바꾸는 우회 수단이 아니다.

확정 조건과 필수 문법 근거는 비용 절감을 위해 잘라내지 않는다. 필수 정보만으로 컨텍스트 한도를 넘으면 중단하며 자동으로 파일/영역별 계획을 재분할하는 기능은 아직 없다. 자산과 체크포인트도 추정으로 진행하지 않는다. 사용자 소유 자산 변경이나 불명확한 실행 재개는 입력을 보존한 채 검토 대상으로 남긴다.

## 4. 33개 항목 요약

**수정 확인**은 재현된 소프트웨어 결함에 한정한 판정이다. **부분**은 코드와 통과한 테스트가 있어도 원문의 전체 요구 또는 실환경 검증이 남았다는 뜻이다. 아래 모든 항목의 native/live-model 검증은 미수행이다.

| 번호 | 감사 주제 | 판정 | 실행한 연결 테스트 수 |
|---:|---|---|---:|
| 1 | 모든 native 경로의 실행 승인 | 수정 확인(범위 한정) | 4 |
| 2 | 전체 execution spec과 자동 수리 승인 범위 | 부분·추가 검증 필요 | 2 |
| 3 | 인라인·중첩·주석 libs 검사 | 수정 확인(범위 한정) | 5 |
| 4 | 파일 정책과 프로세스 정책 분리 | 부분·추가 검증 필요 | 2 |
| 5 | 실제 subprocess 예산과 문법 검사 캐시 | 수정 확인(범위 한정) | 3 |
| 6 | 영역별 case layout과 pre-solve | 부분·추가 검증 필요 | 3 |
| 7 | 영역별 mesh 검증 무효화 | 부분·추가 검증 필요 | 1 |
| 8 | 사용자 geometry·데이터 AssetRegistry | 부분·추가 검증 필요 | 2 |
| 9 | 체크포인트와 보수적 재개 | 부분·추가 검증 필요 | 5 |
| 10 | 구조화된 로컬 MPI | 부분·추가 검증 필요 | 1 |
| 11 | 숫자 override·취소·현재 유효 요구사항 | 부분·추가 검증 필요 | 2 |
| 12 | 물리량·단위·대상 연결 | 부분·추가 검증 필요 | 1 |
| 13 | 결과에 큰 영향을 주는 모호성 | 부분·추가 검증 필요 | 1 |
| 14 | 작성 단계의 실제 문법 근거 | 부분·추가 검증 필요 | 1 |
| 15 | 확정 조건을 자르지 않는 컨텍스트 | 부분·추가 검증 필요 | 1 |
| 16 | 파일 작성 전 설계 불변 조건 검사 | 부분·추가 검증 필요 | 1 |
| 17 | 검색 횟수 대신 필요한 근거 충족 여부 | 부분·추가 검증 필요 | 3 |
| 18 | 발견한 function object의 catalog 연결 | 부분·추가 검증 필요 | 1 |
| 19 | 가용성 근거의 강도 구분 | 부분·추가 검증 필요 | 1 |
| 20 | 검색 scope·언어·편향 | 부분·추가 검증 필요 | 1 |
| 21 | 프로세스 종료와 계산 완료 분리 | 부분·추가 검증 필요 | 8 |
| 22 | timeout의 구조화 및 부분 로그 | 부분·추가 검증 필요 | 1 |
| 23 | 긴 로그의 bounded 메모리 | 부분·추가 검증 필요 | 3 |
| 24 | execution·region 기반 후처리 context | 부분·추가 검증 필요 | 1 |
| 25 | 범용 QuantityOfInterest | 부분·추가 검증 필요 | 1 |
| 26 | 비균일 시각·재시작 통계 | 부분·추가 검증 필요 | 1 |
| 27 | one-shot --solve의 SOLVE_READY 전이 | 수정 확인(범위 한정) | 2 |
| 28 | SOLVE_READY에서 /feedback | 수정 확인(범위 한정) | 1 |
| 29 | Codex ChatGPT/API-key 인증 구분 | 수정 확인(범위 한정) | 4 |
| 30 | Codex transport 환경·이벤트·usage | 부분·추가 검증 필요 | 1 |
| 31 | 중심 agent의 책임 분리 | 부분·추가 검증 필요 | 1 |
| 32 | 깨끗한 배포와 설치 resource | 부분·추가 검증 필요 | 1 |
| 33 | 테스트 수준과 CFD 성공률 분리 | 부분·추가 검증 필요 | 359 |

하나의 테스트가 여러 항목을 검증할 수 있으므로 표의 테스트 수를 합산하지 않는다. 33번은 전체 회귀 묶음이다. 모든 정확한 pytest node ID는 `docs/V4_AUDIT_33.json`에 기록했다.

## 5. 항목별 구현·검증 대조

### 01. /solve 승인 검사가 일반 native 실행 경로에는 없다

**구현 위치:** `src/openfoam_agent/tools/execution_policy.py` · `src/openfoam_agent/tools/safe_runner.py` · `src/openfoam_agent/contracts/execution.py` · `src/openfoam_agent/engineering/agent.py`

**반영 내용:** 명령 효과를 runner가 판정한다. LLM의 role 문자열을 권한으로 쓰지 않는다. 준비·runtime repair·사용자 revision의 일반 native 경로에서도 승인 없는 본 계산은 프로세스 생성 전에 차단한다. 실제 argv, 실행 위치, 승인 계약과 case 입력을 대조한다.

**실행한 검증:** 세 engineering phase에서 모의 실행기까지 도달하지 않는지 검사했다. 실제 SafeRunner에서도 미승인 호출을 막고, 승인된 합성 실행 파일만 시작하는 경계를 검사했다.

**연결 테스트:** `test_a01_unapproved_generic_native_cannot_reach_even_fake_runner`; `test_a01_real_runner_requires_contract_before_spawn`

**미완료·미검증·제약:** 검사한 Python 진입점의 정책 경계에 대한 확인이다. 실제 OpenFOAM 실행, 악의적인 설치 바이너리, OS 권한 격리까지 검증한 것은 아니다.

### 02. 자동 수리의 승인 범위가 plan.solver 문자열에 치우쳐 있다

**구현 위치:** `src/openfoam_agent/contracts/execution.py` · `src/openfoam_agent/runtime/orchestrator.py` · `src/openfoam_agent/engineering/agent.py`

**반영 내용:** driver, module, 영역별 배치, arguments, MPI 계약, 물리 구현, QoI, 자원 제한을 승인 문서에 묶는다. 자동 수리는 제한된 fvSchemes/fvSolution 파일 변경만 허용하고 물리·자산·실행 계약 변경은 재승인 대상으로 다룬다. mesh 수정은 seal과 실행 승인을 무효화한다.

**실행한 검증:** argv 불일치, 승인 후 목표 변경, 승인 자원 상한, 허용·불허 수리 경로를 비교했다.

**연결 테스트:** `test_a02_actual_argv_and_repair_scope_are_bound`; `test_a02_approval_binds_postprocessing_goals_and_resource_limits`

**미완료·미검증·제약:** 문법과 물리 의미를 완전하게 구별하는 자동 판별기는 아니다. 실제 OpenFOAM 수리 재시도와 multi-region module 교체 전체 흐름은 native 미검증이다.

### 03. libs 검사가 줄 배치에 따라 우회된다

**구현 위치:** `src/openfoam_agent/tools/dictionary_policy.py` · `src/openfoam_agent/tools/workspace.py`

**반영 내용:** 줄 시작 정규식 대신 token 단위로 libs를 검사한다. 인라인, 중첩, quoted key/value, 여러 줄, 공백 없는 주석 삽입에도 같은 allowlist를 적용한다. 경로 지정 및 미허용 라이브러리는 거부한다.

**실행한 검증:** 감사 재현 형태와 libs/*comment*/(...) 우회, 주석에만 있는 문자열을 구분하는 회귀 검사를 수행했다.

**연결 테스트:** `test_a03_libs_inline_nested_comments_cannot_bypass`; `test_a03_comment_only_unknown_library_is_not_loaded`

**미완료·미검증·제약:** 검사 대상은 지원하는 dictionary 문법 범위이며 native 라이브러리를 실제 로드하거나 임의 OpenFOAM 문법 전체를 형식 검증한 것은 아니다.

### 04. 안전 정책이 한편으로는 좁고, 다른 한편으로는 실행 효과를 충분히 제한하지 않는다

**구현 위치:** `src/openfoam_agent/tools/dictionary_policy.py` · `src/openfoam_agent/tools/execution_policy.py` · `src/openfoam_agent/tools/exec_guard.py`

**반영 내용:** 문서 검사와 명령 효과·승인을 분리했다. literal local include를 명시적으로 처리할 수 있는 helper에 해시·경로·순환·크기 제한을 추가했다. 동적 코드와 위험 명령/옵션은 거부하고 POSIX CPU·주소 공간·개별 파일 제한을 지원한다.

**실행한 검증:** include 정상 해시, 순환, 외부 경로 거부를 검사했다. 합성 CPU 부하 subprocess에서 CPU 상한 종료를 관찰했다.

**연결 테스트:** `test_a04_local_include_hash_cycle_and_escape`; `test_a04_actual_cpu_limit_is_enforced_on_synthetic_process`

**미완료·미검증·제약:** include helper는 모든 기존 case 가져오기/작성 경로에 자동 연결되어 있지 않다. 메모리 상한 스트레스 검증, 전체 디스크 quota, MPI 전체 합산 CPU·메모리, filesystem/network namespace 격리는 미완성 또는 미검증이다. 서비스용 보안 sandbox로 보증하지 않는다.

### 05. native budget이 실제 subprocess 수를 세지 않는다

**구현 위치:** `src/openfoam_agent/tools/safe_runner.py` · `src/openfoam_agent/tools/openfoam.py` · `src/openfoam_agent/engineering/agent.py` · `src/openfoam_agent/cli.py`

**반영 내용:** 실제 Popen 진입에서 spawn intent와 pid를 기록하고 예산을 소모한다. 여러 dictionary를 확인하는 한 action도 각각의 프로세스가 계수된다. 파일 해시·실행 파일·환경·옵션 기반 캐시를 적용하고 승인된 한도와 runner 한도 중 더 엄격한 값을 따른다. CLI는 legacy event 수와 실제 spawn 수를 구분한다.

**실행한 검증:** budget 1에서 두 번째 실제 프로세스 차단, cache hit의 무차감, 파일 변경 시 재검사, 승인 한도의 우선 적용을 확인했다.

**연결 테스트:** `test_a05_budget_is_actual_spawns_and_cache_hits_do_not_spend`; `test_a05_file_hash_invalidates_dictionary_cache`; `test_a05_approval_budget_is_stricter_than_runner_budget`

**미완료·미검증·제약:** 직접 시작한 native subprocess 수이며 MPI 내부 rank/손자 프로세스 개수를 같은 계수로 세지 않는다. rank 수는 별도 계약이다. 원장 동시 접근은 단일 agent 프로세스 전제로, 다중 프로세스 workspace lock은 없다.

### 06. multi-region은 실행 명세에 들어왔지만 pre-solve는 아직 단일 영역 중심이다

**구현 위치:** `src/openfoam_agent/contracts/models.py` · `src/openfoam_agent/contracts/regions.py` · `src/openfoam_agent/verification/presolve.py`

**반영 내용:** RegionCaseLayout을 공통 계약으로 두고 각 영역의 mesh, 초기 필드, fvSchemes/fvSolution과 interface patch를 검사한다. 다중 영역에 dummy root 수치 dictionary를 요구하지 않는다.

**실행한 검증:** fluid/solid fixture에서 루트 fv 파일 없이 성공하고 solid 필드 누락은 fluid 필드로 대체되지 않음을 확인했다. interface의 존재와 연결 대상을 검사했다.

**연결 테스트:** `test_a06_two_regions_need_no_dummy_root_fv_files`; `test_a06_missing_solid_field_is_not_satisfied_by_fluid_field`; `test_a06_interface_checks_membership_not_flux_truth`

**미완료·미검증·제약:** 실제 conjugate heat transfer solver 실행, 영역 사이 열유속 연속성·보존성, 모든 필수 물성 model 호환성은 검증하지 않았다. 정적 layout 통과는 물리적으로 올바른 CHT case의 보증이 아니다.

### 07. region별 mesh 변경이 mesh 검증 무효화에 반영되지 않는다

**구현 위치:** `src/openfoam_agent/contracts/regions.py` · `src/openfoam_agent/tools/workspace.py` · `src/openfoam_agent/engineering/agent.py` · `src/openfoam_agent/verification/safety.py`

**반영 내용:** 영역별 mesh evidence와 입력 digest를 사용한다. region polyMesh·region mesh dictionary·공유 geometry 변경을 반영하고 native mesh 변경 후 seal/approval을 무효화한다. checkMesh 근거는 제한된 UI tail 대신 해시가 연결된 전체 디스크 로그에서 읽는다.

**실행한 검증:** 영역별 독립 변경, 공유 geometry의 공통 무효화, 긴 합성 checkMesh 로그의 원문 처리와 경계 검사를 수행했다.

**연결 테스트:** `test_a07_mesh_digest_separates_regions_and_tracks_shared_geometry`

**미완료·미검증·제약:** 보수적인 경로 기반 의존성 모델이다. 모든 임의 native 도구의 입력·출력 관계를 학습하는 완전한 dependency DAG는 아니다. 실제 OpenFOAM checkMesh는 실행하지 않았다.

### 08. 실제 geometry·데이터 입력 연결이 부족하다

**구현 위치:** `src/openfoam_agent/tools/assets.py` · `src/openfoam_agent/tools/workspace.py` · `src/openfoam_agent/schemas/request.py` · `src/openfoam_agent/cli.py`

**반영 내용:** 명시적으로 승인한 원본만 case에 복사하고 원본/사본 해시, 형식, 크기 및 선언된 단위·좌표계·연결 정보를 기록한다. LLM 작성 경로와 사용자 자산을 분리해 덮어쓰기와 symlink/경로 이탈을 막는다. binary STL의 좌표 유한성과 삼각형 구조, CSV 기본 형식을 검사한다.

**실행한 검증:** binary 자산의 내용 보존·승인·불변성, symlink와 case 외부 경로 거부를 확인했다.

**연결 테스트:** `test_a08_binary_asset_hash_approval_and_immutability`; `test_a08_asset_symlink_and_outside_case_rejected`

**미완료·미검증·제약:** STL manifold/교차/법선 품질 및 실제 meshing은 미검증이다. 단위·표면·BC 연결은 선언 정보이며 자동 의미 검증이 아니다. 기존 case 가져오기는 입력 중심으로, 계산 시간 디렉터리의 정식 restart를 지원한 것으로 보지 않는다. import 실패 시 부분 사본 자동 rollback은 없다.

### 09. checkpoint와 재개가 일급 기능이 아니다

**구현 위치:** `src/openfoam_agent/workflow/checkpoint.py` · `src/openfoam_agent/workflow/state.py` · `src/openfoam_agent/engineering/agent.py` · `src/openfoam_agent/cli.py`

**반영 내용:** 확정 state뿐 아니라 staged draft, pending authoring, evidence, 실행 원장을 versioned/checksummed atomic checkpoint에 저장한다. 입력 해시와 환경을 대조하고, 불명확한 action/transaction/실행 중 프로세스는 자동 replay하지 않는다. 재개 시 실행 승인은 폐기한다.

**실행한 검증:** draft·예산 복원, 파일 변경, 환경 변경, pending action, 중단된 transaction을 주입해 보존·거부 동작을 검사했다.

**연결 테스트:** `test_a09_draft_and_budget_checkpoint_restore`; `test_a09_uncertain_or_changed_state_is_preserved_not_replayed`

**미완료·미검증·제약:** WSL 재시작, 실제 장기 solver 중단 후 process attach/restart, schema migration, fork CLI, 여러 agent의 동시 접근은 검증·완성되지 않았다. 환경 fingerprint는 전체 shared-library closure를 해시하는 방식이 아니다.

### 10. 병렬/HPC 실행은 명시적 계약이 필요하다

**구현 위치:** `src/openfoam_agent/contracts/models.py` · `src/openfoam_agent/runtime/parallel.py` · `src/openfoam_agent/tools/safe_runner.py` · `src/openfoam_agent/runtime/orchestrator.py`

**반영 내용:** serial/local MPI, rank 상한, launcher, decomposition method, 재구성 여부를 계약화했다. decomposePar 입력과 rank별 산출물 해시를 확인하고 승인된 argv만 실행한다. 기존 processor 산출물은 강제 삭제하지 않으며 마지막 시각 확인 후 reconstructPar로 연결한다.

**실행한 검증:** 모의 transport로 분할 계약, rank별 파일, 기존 산출물 보존과 오류 경로를 확인했다.

**연결 테스트:** `test_a10_mpi_structure_and_existing_outputs_preserved`

**미완료·미검증·제약:** 실제 mpirun/분할/병렬 계산/재구성은 모두 미실행이다. collated/distributed I/O, hostfile, scheduler/Slurm, 병렬 restart와 모든 rank 일괄 종료는 미지원 또는 미검증이다. MPI 완료로 표시하지 않는다.

### 11. 사용자 숫자를 보존하려다 정상적인 수정 요청을 거부한다

**구현 위치:** `src/openfoam_agent/contracts/requirements.py` · `src/openfoam_agent/agents/intake.py` · `src/openfoam_agent/schemas/intake.py`

**반영 내용:** 대화의 명시적 대상별 scalar assignment를 시간순으로 다루어 과거 숫자 전체를 최종 fact에 요구하지 않는다. 알려진 Re alias와 명시적 fact ID를 정규화하고 active/superseded/cancelled 이력을 보존한다. 질문이나 다른 대상 숫자를 자동 override로 취급하지 않는다.

**실행한 검증:** Re 1000→2000, 다른 압력 대상, 질문형 발화, 취소, 파일명/근거 구분 관련 회귀를 수행했다.

**연결 테스트:** `test_a11_numeric_override_preserves_latest_not_historical`; `test_a11_different_targets_questions_and_cancellation_are_not_silent_overrides`

**미완료·미검증·제약:** 일반 자연어 모든 표현을 이해하는 deterministic 요구사항 해석기는 아니다. 지원하는 명시적 scalar 표현 밖에서는 모델 해석 또는 사용자 재확인이 필요하다. live intake 모델 정확도는 미검증이다.

### 12. 숫자·단위·대상 간의 의미 연결이 더 필요하다

**구현 위치:** `src/openfoam_agent/contracts/models.py` · `src/openfoam_agent/contracts/quantities.py` · `src/openfoam_agent/agents/intake.py`

**반영 내용:** quantity/value/unit/dimensions와 region/patch/material/reference를 표현하고 제한된 SI 변환 및 차원·대상 일관성을 검증한다. 숫자만 같아도 대상이 다르면 같은 조건으로 인정하지 않는다.

**실행한 검증:** 단위 변환 및 대상 불일치 거부를 확인했다.

**연결 테스트:** `test_a12_units_and_targets_must_both_be_preserved`

**미완료·미검증·제약:** 단위 표는 제한적이며 모든 물성·시간 함수·비선형 관계를 지원하지 않는다. 모든 typed quantity를 실제 native dictionary 값과 끝까지 대조하는 연결은 불완전하다.

### 13. 모호한 문장을 바로 물리 경계조건으로 확정하지 않도록 해야 한다

**구현 위치:** `src/openfoam_agent/schemas/intake.py` · `src/openfoam_agent/agents/intake.py` · `src/openfoam_agent/workflow/state.py`

**반영 내용:** critical ambiguity를 계약에 두고 해결되지 않은 영향도 높은 해석은 review-ready/confirm 경계를 통과하지 못하게 한다. 사용자 근거를 통해 해소하도록 한다.

**실행한 검증:** 물리적으로 중요한 미해결 ambiguity가 남으면 review-ready 생성이 거부되는지 검사했다.

**연결 테스트:** `test_a13_unresolved_physics_ambiguity_cannot_be_review_ready`

**미완료·미검증·제약:** 벽/외곽, gauge/absolute, 총열량/체적열원 등의 모호성을 모델이 얼마나 잘 발견하는지는 실제 모델 평가를 하지 않았다. 구조적 차단 장치와 발견 능력은 별개다.

### 14. 파일 작성 단계에서 앞서 찾은 문법 근거가 빠진다

**구현 위치:** `src/openfoam_agent/contracts/evidence.py` · `src/openfoam_agent/engineering/agent.py` · `src/openfoam_agent/llm/context.py`

**반영 내용:** staged authoring에 ImplementationEvidencePack을 넣는다. 실제 읽은 reference 본문, 출처·해시, 대상 파일 연결을 전달하며 검색 결과 요약이나 reference ID만으로 문법 근거를 대체하지 않는다.

**실행한 검증:** 실제 read 본문이 authoring payload에 남고 검색 요약만으로는 같은 증거를 만들지 못하는지 확인했다.

**연결 테스트:** `test_a14_read_syntax_is_projected_with_hash_not_just_search_summary`

**미완료·미검증·제약:** staged DesignCaseAction 경로 중심으로 강제된다. legacy one-shot author 경로 전부를 같은 수준으로 강화하지는 않았다. 근거의 내용이 선택 model과 실제로 호환되는지에 대한 native 검증도 없다.

### 15. 일반 컨텍스트 축약기가 확정 요구사항까지 자른다

**구현 위치:** `src/openfoam_agent/llm/context.py` · `src/openfoam_agent/engineering/agent.py`

**반영 내용:** 확정 facts, 실행 계약, 필수 파일 연결, 자산과 문법 pack을 보호한다. 선택적 검색·로그만 축약하며 필수 정보 자체가 예산을 넘으면 조용히 자르지 않고 호출 전에 실패한다.

**실행한 검증:** 200개 확정 항목을 포함한 mandatory context 보존과 예산 초과의 fail-closed 경로를 검사했다.

**연결 테스트:** `test_a15_mandatory_context_is_not_truncated`

**미완료·미검증·제약:** 자동 파일/영역 단위 작업 분할은 구현하지 않았다. 큰 요청은 예산 조정 또는 명시적 분할이 필요하다. 토큰 비용이 항상 감소한다는 주장은 하지 않는다.

### 16. design 단계에서 검사할 수 있는 오류를 너무 늦게 검사한다

**구현 위치:** `src/openfoam_agent/contracts/regions.py` · `src/openfoam_agent/contracts/evidence.py` · `src/openfoam_agent/engineering/agent.py`

**반영 내용:** staged design 단계에서 확정 fact 대응, 미선언 binding, 중복 region, 필수 파일·문법 근거 연결 등 파일 없이 가능한 검사를 앞당겼다.

**실행한 검증:** 누락·미선언 binding과 중복 region을 설계 단계에서 거부하는지 확인했다.

**연결 테스트:** `test_a16_early_design_rejects_undeclared_binding_and_duplicate_regions`

**미완료·미검증·제약:** 물리 모델의 완전한 호환성, 모든 legacy author 경로, 실제 생성 dictionary 의미 검사까지 early design 단계로 이동한 것은 아니다.

### 17. 고정된 검색 횟수보다 “근거가 충분한가”를 더 잘 판단해야 한다

**구현 위치:** `src/openfoam_agent/contracts/evidence.py` · `src/openfoam_agent/engineering/agent.py`

**반영 내용:** staged 설계/작성의 파일별 근거 연결과 실제 읽기 증거를 요구한다. 검색 상한과 반복 억제는 유지하되 횟수 소진 자체를 근거 충족으로 취급하지 않는다.

**실행한 검증:** bounded retrieval 및 staged 전환의 기존 회귀에 더해 작성 근거 연결을 확인했다.

**연결 테스트:** `test_a14_read_syntax_is_projected_with_hash_not_just_search_summary`; `test_two_retrieval_cycles_stay_bounded_and_switch_to_small_decision_schema`; `test_staged_cli_style_flow_reaches_solve_ready_with_two_small_contracts`

**미완료·미검증·제약:** 모든 물리 조합의 availability/model/interface 호환성을 포괄하는 범용 evidence sufficiency planner는 미완성이다. 하드 상한에 도달했을 때 자동으로 다른 탐색 전략을 구성한다고 보장하지 않는다.

### 18. function object를 발견해 놓고 catalog로 연결하지 않는다

**구현 위치:** `src/openfoam_agent/tools/installation.py` · `src/openfoam_agent/tools/capability_catalog.py`

**반영 내용:** 설치 소스에서 발견한 function object 등 component를 catalog와 모델 노출 경로로 연결하고 탐색 근거를 유지한다.

**실행한 검증:** source-discovered function object의 catalog 등록과 노출을 fixture로 확인했다.

**연결 테스트:** `test_a18_19_source_discovery_is_not_runtime_registration`

**미완료·미검증·제약:** boundary condition, thermophysical/momentum transport model 등 모든 runtime selectable category의 완전한 catalog 통합 및 실제 사용 성공 검증은 남아 있다.

### 19. installed 근거가 실제 생성 가능성을 과하게 암시할 수 있다

**구현 위치:** `src/openfoam_agent/tools/installation.py` · `src/openfoam_agent/tools/capability_catalog.py` · `src/openfoam_agent/schemas/capability.py`

**반영 내용:** source_discovered와 binary_present 등을 구분하고 소스 이름 발견을 runtime load/실행 검증으로 보고하지 않는다.

**실행한 검증:** 소스 발견 provider가 실제 등록·native 실행 성공으로 승격되지 않는지 확인했다.

**연결 테스트:** `test_a18_19_source_discovery_is_not_runtime_registration`

**미완료·미검증·제약:** foamToC 또는 작은 실제 생성 실행에 의한 runtime table 등록·loadability probe는 구현 완료/검증으로 볼 수 없다. 소스가 있다는 사실만 확인한 경우 그대로 제한적으로 표시한다.

### 20. reference 검색에 순서 편향과 언어 한계가 있다

**구현 위치:** `src/openfoam_agent/tools/references.py` · `src/openfoam_agent/tools/capability_catalog.py`

**반영 내용:** Unicode/NFKC 토큰과 제한된 한국어 용어 alias, source/tutorials/etc/modules의 균형 탐색, bounded cache 및 실제 read hash를 추가했다. 잘림·scope 소진을 결과에 명시한다.

**실행한 검증:** 한국어 query, FOAM_MODULES, scope별 자료 접근과 실제 read 해시를 fixture로 확인했다.

**연결 테스트:** `test_a20_unicode_search_fair_scopes_and_actual_read_hash`

**미완료·미검증·제약:** 일반 다국어 의미 검색, 모든 query의 최적 자료 순위, 대규모 persistent 설치 인덱스는 구현·평가하지 않았다. 제한된 용어 정규화 개선이다.

### 21. 현재 runtime 완료 판정은 너무 약하다

**구현 위치:** `src/openfoam_agent/runtime/completion.py` · `src/openfoam_agent/tools/parsers.py` · `src/openfoam_agent/runtime/orchestrator.py` · `src/openfoam_agent/workflow/state.py`

**반영 내용:** 정상 exit, 실제 진행, transient 목표 구간 또는 steady 수렴 조건, 결과 파일 확인을 분리한다. literal controlDict 목표와 완료 계약을 대조한다. 결과의 존재·유한성·해시 및 실행 전후 변경 여부를 검사하고 미완료 결과는 /accept로 COMPLETE가 되지 않는다.

**실행한 검증:** Time=0/End 및 짧은 계산 거부, steady outer residual 기준, 목표 축소 거부, 결과 누락·비유한·기존 stale 파일, 미완료 accept 거부를 검사했다.

**연결 테스트:** `test_a21_exit_is_not_requested_completion`; `test_a21_contract_progress_and_bounded_parser`; `test_a21_steady_requires_outer_not_inner_residual_convergence`; `test_a21_completion_cannot_shorten_literal_control_target`; `test_a21_outputs_must_be_fresh_present_and_finite`; `test_a21_accept_cannot_turn_incomplete_calculation_into_complete`

**미완료·미검증·제약:** native CFD 정확성·보존성·사용자 물리 목표는 검증하지 않았다. 결과 parser는 제한된 ASCII/gzip 검증으로 모든 OpenFOAM binary field/차원/원소 개수를 보증하지 않는다. restart startFrom 지원도 제한적이며 불명확하면 차단한다.

### 22. timeout이 실행 결과로 정리되지 않는 경로가 있다

**구현 위치:** `src/openfoam_agent/tools/safe_runner.py` · `src/openfoam_agent/schemas/common.py` · `src/openfoam_agent/runtime/orchestrator.py`

**반영 내용:** timeout을 종료 이유, 부분 디스크 로그, 마지막 관측치와 프로세스 정보가 남는 ToolResult로 반환한다. POSIX process group 정리 경로와 승인 timeout 한도를 연결했다.

**실행한 검증:** 실제 합성 subprocess를 timeout으로 종료하고 부분 로그와 하나의 spawn 원장을 확인했다.

**연결 테스트:** `test_a22_timeout_returns_partial_disk_log_and_counts_one_spawn`

**미완료·미검증·제약:** 실제 MPI/복잡한 자식 프로세스 트리의 모든 종료 및 장기 solver restart 가능성은 검증하지 않았다. runner의 구조화된 실패 결과가 자동으로 재시작 안전성을 보장하지는 않는다.

### 23. 긴 계산의 로그 메모리 사용이 커질 수 있다

**구현 위치:** `src/openfoam_agent/tools/safe_runner.py` · `src/openfoam_agent/tools/parsers.py` · `src/openfoam_agent/verification/safety.py`

**반영 내용:** 원문은 디스크로 streaming 저장하고 해시를 계산한다. 메모리 tail/queue/부분 줄과 parser 시계열을 제한한다. 출력 상한 초과는 명시적 종료 사유이며 checkMesh는 원문 파일에서 제한된 검사로 근거를 수집한다.

**실행한 검증:** 1MB 이상 합성 출력, bounded tail과 전체 로그, 출력 예산 중단, tail에서 사라진 checkMesh 결과의 전체 로그 처리를 확인했다.

**연결 테스트:** `test_a23_large_log_is_bounded_in_memory_and_complete_on_disk`; `test_a23_output_budget_stops_process`; `test_a23_checkmesh_reads_full_bounded_disk_log_not_tail`

**미완료·미검증·제약:** 모든 코드 경로가 O(1) 메모리라는 보증은 아니다. 특히 Codex subprocess.run의 stdout는 JSON parse 상한 검사 전에 capture된다. 실제 장기 native 계산, 디스크 부족, 네트워크 filesystem 장애는 미검증이다.

### 24. post-processing 실행도 새 execution 구조를 따라가지 못한다

**구현 위치:** `src/openfoam_agent/postprocessing/context.py` · `src/openfoam_agent/postprocessing/agent.py` · `src/openfoam_agent/tools/openfoam.py`

**반영 내용:** 후처리에는 driver 문자열 대신 선택한 영역의 solver module을 전달한다. region이 모호하거나 direct application의 module context가 확인되지 않으면 거부한다.

**실행한 검증:** 다중 영역의 올바른 module 선택, driver와 module 구분 및 모호한 선택의 거부를 확인했다.

**연결 테스트:** `test_a24_postprocessing_uses_region_module_not_driver`

**미완료·미검증·제약:** 실제 foamPostProcess와 solver-context function object 실행은 미검증이다. 지원하지 않는 direct application이나 병렬 결과 형태를 자동 추정하여 실행하지 않는다.

### 25. 정량 후처리가 force coefficient 중심이다

**구현 위치:** `src/openfoam_agent/contracts/models.py` · `src/openfoam_agent/postprocessing/quantities.py` · `src/openfoam_agent/postprocessing/agent.py` · `src/openfoam_agent/workflow/state.py`

**반영 내용:** 물리량·region/selection·단위·시간 구간·연산·입력 파일 계약을 추가했다. scalar CSV/텍스트의 평균·RMS·적분·극값·차이/수지 산술과 원문 hash를 기록하고 승인된 QoI만 실행한다. 산술 검증과 물리 의미 검증을 분리해 보고한다.

**실행한 검증:** 서로 다른 scalar quantity, source hash, 선언된 단위/영역 및 비유한 입력 경계를 검사했다.

**연결 테스트:** `test_a25_generic_scalar_quantities_provenance_and_declared_semantics`

**미완료·미검증·제약:** 공간 적분/patch flux/energy balance의 native 생성과 물리 의미 검증은 하지 않는다. 차이/수지 연산 성공이 질량·에너지 보존 증명은 아니다. physical_semantics_verified=False를 유지한다.

### 26. 비균일 시간 간격의 통계 처리가 필요하다

**구현 위치:** `src/openfoam_agent/postprocessing/quantities.py` · `src/openfoam_agent/postprocessing/analysis.py`

**반영 내용:** piecewise-linear 시간 가중 평균·적분·RMS, 중복 시각과 latest restart segment 정리, 시간 구간 보간 및 관측 길이 기반 주파수 해상도 정보를 추가했다.

**실행한 검증:** 비균일 간격에서 샘플 평균과 시간 평균을 구분하고 재시작·중복 데이터를 정리하는 산술 검사를 수행했다.

**연결 테스트:** `test_a26_nonuniform_time_weighting_and_restart_reconciliation`

**미완료·미검증·제약:** 적분은 명시된 보간 가정에 따른 값이다. 자동 정상성 판정·과도 구간 선택과 엄밀한 주파수 신뢰구간/다중 창 불확실성 분석은 완료하지 않았다.

### 27. one-shot --solve의 상태 조건이 오래된 상태를 본다

**구현 위치:** `src/openfoam_agent/workflow/states.py` · `src/openfoam_agent/cli.py` · `src/openfoam_agent/workflow/state.py`

**반영 내용:** interactive와 one-shot이 공통 solve 승인 상태 집합과 계약 생성 경로를 사용한다. SOLVE_READY를 승인 가능한 상태에 포함하고 runtime policy를 승인 전에 준비한다.

**실행한 검증:** one-shot --solve에서 workflow가 승인 후 재진입하는지, interactive가 같은 승인 계약을 만드는지 검사했다.

**연결 테스트:** `test_a27_one_shot_solve_ready_reenters_workflow_after_common_approval`; `test_a27_interactive_solve_uses_same_approval_contract`

**미완료·미검증·제약:** 상태 전이는 모의 workflow로 검증했다. CLI를 통한 실제 OpenFOAM solver 실행은 별도 미검증이다.

### 28. SOLVE_READY에서 안내하는 /feedback이 거부될 수 있다

**구현 위치:** `src/openfoam_agent/cli.py` · `src/openfoam_agent/review/agent.py` · `src/openfoam_agent/workflow/states.py`

**반영 내용:** 공통 feedback 허용 상태에 SOLVE_READY를 포함하고 실제 review agent까지 연결한다. 오류 시 승인/검토 상태를 안전하게 유지한다.

**실행한 검증:** 명령 안내만 바꾸는 검사가 아니라 SOLVE_READY에서 실제 review 경로에 도달함을 확인했다.

**연결 테스트:** `test_a28_solve_ready_feedback_reaches_actual_review_agent`

**미완료·미검증·제약:** 사용자의 자연어 결과 피드백을 모델이 정확하게 반영하고 native case를 올바르게 수정하는 능력은 평가하지 않았다.

### 29. Codex backend가 구독 인증 사용을 확실히 판별하지 않는다

**구현 위치:** `src/openfoam_agent/llm/codex_transport.py` · `src/openfoam_agent/llm/codex_client.py`

**반영 내용:** ChatGPT 로그인임이 명확한 status만 허용한다. API-key 또는 판별 불가 응답은 거부하고 강제 ChatGPT 로그인 설정을 전달한다. 환경변수 제거만으로 인증 경로를 보장하지 않는다.

**실행한 검증:** ChatGPT, API-key, 단순 Logged in, Not logged in status fixture 및 기존 adapter 회귀를 검사했다.

**연결 테스트:** `test_a29_auth_kind_is_not_any_logged_in_message`

**미완료·미검증·제약:** 실제 사용자의 Codex 로그인 상태, CLI 버전별 모든 출력, 구독 사용량/과금 경로는 확인하지 않았다. 모의 판별 검사 통과를 구독 사용 보장이라고 표시하지 않는다.

### 30. Codex adapter의 “모델 전용 transport” 의도를 더 강하게 보장해야 한다

**구현 위치:** `src/openfoam_agent/llm/codex_transport.py` · `src/openfoam_agent/llm/codex_client.py`

**반영 내용:** 환경변수 allowlist와 --json/--ignore-user-config 기능 계약을 사용한다. 완료 이벤트와 응답을 대조하고 도구 사용/미지원 이벤트가 관측되면 결과를 채택하지 않는다. usage가 없으면 unknown으로 남긴다.

**실행한 검증:** 환경 필터, JSONL 완료/도구 이벤트, usage 유무와 auth 경계를 fixture로 검사했다.

**연결 테스트:** `test_a30_environment_allowlist_and_event_usage`

**미완료·미검증·제약:** 이벤트 검사는 사후 감지이며 도구 실행을 사전에 봉쇄하는 격리가 아니다. stdout capture의 선제 메모리 제한과 모든 backend 공통 계약은 미완성이다. 실제 Codex/Claude/Ollama/OpenAI 호출은 실행하지 않았다.

### 31. 중심 agent 파일에 책임이 너무 많이 모여 있다

**구현 위치:** `src/openfoam_agent/engineering/agent.py` · `src/openfoam_agent/contracts/execution.py` · `src/openfoam_agent/contracts/evidence.py` · `src/openfoam_agent/runtime/completion.py` · `src/openfoam_agent/workflow/checkpoint.py`

**반영 내용:** 승인·영역·요구사항·근거·자산·실행·완료·체크포인트·후처리 context를 별도 계약/모듈로 분리했다. 공통 상태 전이와 보고 필드를 일부 중앙화했다.

**실행한 검증:** 이동한 기능의 경계 테스트와 기존 298개 전체 회귀를 수행했다.

**연결 테스트:** `test_staged_cli_style_flow_reaches_solve_ready_with_two_small_contracts`

**미완료·미검증·제약:** engineering/agent.py는 여전히 큰 조정자이며 책임·의존성의 전면 재설계는 끝나지 않았다. 테스트 통과가 유지보수 구조 완성도를 증명하지 않는다.

### 32. 생성물이 Git에 많이 추적되어 있다

**구현 위치:** `.gitignore` · `pyproject.toml` · `src/openfoam_agent/cli.py` · `src/openfoam_agent/data/openfoam13_capability_graph.json` · `src/openfoam_agent/data/openfoam14_capability_graph.json`

**반영 내용:** 4.0.0rc1로 버전 구분하고 13/14 config를 package resource로 포함했다. 최종 배포에서 cache/pyc/egg-info/build/.git를 제외한다. 정확한 원본 ZIP 기준으로 patch를 만들고 재적용하여 파일 단위 일치를 검사한다.

**실행한 검증:** wheel 빌드, 소스 밖 별도 venv의 설치 CLI --help, 13/14 resource JSON 및 해시를 확인했다. package/source config 동일성 테스트를 수행했다.

**연결 테스트:** `test_a32_packaged_profiles_match_source_bytes`

**미완료·미검증·제약:** wheel은 --no-deps로 설치하고 기존 의존성을 재사용했다. 깨끗한 의존성 resolve, 실제 OpenAI SDK/backend, Python 3.12, Windows/WSL 설치 검증은 하지 않았다. 감사 원문의 WSL tracked pyc 111개는 이 기준 ZIP에는 없었다.

### 33. 298개 테스트와 “범용 CFD 성공률” 사이를 연결해야 한다

**구현 위치:** `tests/test_v400_execution_contracts.py` · `tests/test_v400_audit_domains.py` · `tests/test_v400_release_edges.py`

**반영 내용:** 기존 회귀와 감사 경계 검사를 분리하고 JUnit 로그 및 테스트 ID를 보존한다. 실패 없이 통과한 두 partition의 합집합이 수집한 전체 359개와 정확히 일치함을 확인했다. native/live-model qualification은 별도 미실행 목록으로 남긴다.

**실행한 검증:** 기준본 298개 통과. 수정본 기존 298개와 신규 61개가 각각 통과했으며 누락/중복/skip=0을 XML과 collection으로 대조했다.

**연결 테스트:** 전체 359개. `verification/test_summary.json`의 `passed_nodeids` 및 `verification/all_tests.xml`.

**미완료·미검증·제약:** 실제 CFD native 통합 검사 및 미공개 문제/live-model 평가를 수행하지 않았다. 범용 CFD 성공률, 수치 정확도, 사용자 목표 달성률은 산출할 수 없다. 원문이 제안한 3단계 평가 중 빠른 회귀 층만 수행했다.

## 6. 사용 전 알아야 할 운영 제약

이 버전은 하나의 workspace에 한 agent 프로세스를 사용하는 로컬 연구용 후보판이다. 다중 사용자 서비스용 OS 격리를 제공하지 않는다. `shell=False`, trusted executable, 경로 검사와 resource limit은 filesystem namespace나 네트워크 차단과 같지 않다. 악성 설치 바이너리를 안전하게 실행하는 sandbox로 사용하면 안 된다.

프로세스 예산은 runner가 직접 생성하는 native 프로세스 수다. MPI rank 수는 별도 계약이다. CPU/메모리 제한을 모든 MPI rank의 합산 자원 보장으로 해석하지 않는다. 개별 파일 크기 상한은 case 전체의 디스크 quota가 아니다. 체크포인트의 SHA256은 무결성 비교이며 비밀키 서명이나 권한 상승 공격 방어를 대신하지 않는다.

resume은 저장된 state를 안전하게 비교·복원하는 기능이지 어떤 중단 상태에서도 solver를 알아서 이어 돌리는 기능이 아니다. pending action, 진행 중 원장, 환경·파일 변화, 미완료 transaction은 자동 replay하지 않고 차단한다. 기존 processor 디렉터리를 무시하거나 `-force`로 지우며 재분할하지 않는다.

Codex transport에서 도구 이벤트를 발견하면 응답을 거부하지만 이는 **사후 감지**다. 그 도구가 이미 실행되지 않았음을 보증하지 않는다. 또 usage 미확인은 0 token이 아니다. 기능 플래그나 인증 종류를 판별할 수 없으면 fail-closed 동작할 수 있다.

generic QoI는 이미 추출된 scalar 데이터와 선언된 위치·단위의 산술을 검증한다. 모델이 잘못된 patch 데이터를 골랐는데도 숫자가 유한하다는 이유만으로 물리적으로 올바르다고 판정하지 않는다. 공간 적분, 열유속, 질량·에너지 보존과 수치 정확도는 추가 native/물리 검증이 필요하다.

## 7. 남아 있는 실환경 qualification

`research/V4_QUALIFICATION_PLAN.json`은 **미실행 평가 계획**이며 완료된 자동 native 검증 harness가 아니다. 아래 군은 모두 NOT_RUN이다.

| ID | 평가군 | 필요한 확인 |
|---|---|---|
| N01 | 단일 영역 정상 유동 | 실제 dictionary → blockMesh/checkMesh → steady solver → 수렴·필드 검사 |
| N02 | 비정상 외부 유동 | 실제 transient 목표 시간·안정성·와류 주파수/관측 기간 검사 |
| N03 | 열전달 | 실제 물성·열원·에너지 수지 및 온도 결과 검증 |
| N04 | 유체–고체 복합 열전달 | fluid/solid layout·interface 연결·연속 열유속 검증 |
| N05 | 다상 문제 | 상별 필드·solver/module 호환성·질량 보존 검증 |
| N06 | 실제 STL 자산 | 사용자 자산 해시 → surface quality → meshing → 해석 |
| N07 | 기존 case·중단·재개 | 프로세스 중단/OS 재시작·checkpoint 검증·명시적 재승인·restart |
| N08 | 로컬 MPI | 실제 decomposePar → mpirun → rank 종료·출력 → reconstructPar |
| M01 | 미공개 자연어 문제 | 튜토리얼 그대로가 아닌 조건·표현 변경 문제의 live-model 평가 |
| M02 | 모호하거나 불충분한 요구 | 영역 경계·단위·gauge·열원 의미의 질문/차단 정확도 |
| B01 | 실제 backend | Codex 구독 로그인·JSON 기능, OpenAI/Claude/Ollama 연동·취소·usage |

성공률은 intake, design, case 생성, mesh, solver 기동, 목표 구간/수렴, 결과 유효성, 수치 품질, 물리 목표, 비용, 사람 개입을 분리해 기록해야 한다. 실제 시도에서 실패한 run을 분모에서 빼면 안 된다. 반대로 아직 실행하지 않은 군을 실패율 0% 또는 성공률 100%로 표시해서도 안 된다.

가장 먼저 남은 확인은 실제 OpenFOAM 13의 작은 serial case에서 작성→dictionary→mesh→승인→solver→결과→후처리까지의 연결이다. 다음은 fluid/solid 영역과 local MPI, 실제 자산, 중단·재개 순서다. 이 검증 결과가 생기기 전에는 4.0.0 최종판이나 범용 CFD 성능이 검증된 배포본으로 승격하지 않는다.

## 8. 설치·패치 적용·회귀 재현

### 깨끗한 ZIP 사본 사용

기존 작업 폴더를 덮어쓰지 말고 새 폴더로 압축을 해제한다. 프로젝트는 Python 3.12 이상을 요구하지만 이번 실행은 3.13.5에서만 확인했다. 다음은 설치/회귀 재현 명령이며 실제 모델이나 solver를 자동 실행하지 않는다.

```bash
cd OpenFOAM_Agent_v4.0.0rc1
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[dev]"
openfoam-agent --help
sh scripts/run_v4_regression.sh
```

새 의존성 설치는 이번 환경에서 재현하지 않았으므로 그 단계의 실패를 테스트 완료로 간주하지 않는다. OpenFOAM 및 backend는 별도로 설치·활성화하고 native qualification을 수행해야 한다. 진행 중인 계산/민감한 파일이 있는 원래 workspace를 첫 시험에 사용하지 않는다.

### v3.6.0 → v4 패치

패치는 이름이 비슷한 최신 Git branch가 아니라 위 SHA256의 원본 ZIP 내용에 대해 생성된다. 로컬 수정본은 먼저 별도 복사/commit으로 보존한다. 깨끗한 기준본의 **프로젝트 루트**에서 실행한다.

```bash
git apply --check ../OpenFOAM_Agent_v3.6.0_to_v4.0.0rc1.patch
git apply ../OpenFOAM_Agent_v3.6.0_to_v4.0.0rc1.patch
```

`--check`가 실패하면 강제 적용하지 않는다. 기존 checkpoint는 새 schema와 입력·환경 검사 때문에 자동 이관이 거부될 수 있다. v3.6의 승인 상태를 v4의 실행 승인으로 간주하지 않는다.

### 검증 결과 파일

`verification/baseline_pytest.txt`, `legacy_pytest.txt/.xml`, `v4_pytest.txt/.xml`, `all_tests.xml`, `test_summary.json`, `collected_tests.txt`, `compileall.txt`, `environment.json`, `wheel_build.txt`, `wheel_install.txt`, `installed_cli_help.txt`, `wheel_smoke.json`이 실행 근거다. 파일/패키지 검증 결과는 외부 `OpenFOAM_Agent_v4_package_verification.json`에 기록하며 최종 산출물의 해시는 `SHA256SUMS.txt`로 확인한다.

```bash
sha256sum -c SHA256SUMS.txt
```

## 9. 근거와 문서 범위

요구사항의 권위 있는 원문은 보존한 33개 감사 텍스트다. 구현·검증 사실의 근거는 배포 소스, 해당 테스트와 실제 실행 로그다. 기존 README/ARCHITECTURE/SECURITY와 V2/V3 변경 이력은 역사 기록으로 남겼다. 과거 표현과 충돌하면 이 v4 보고서의 제한 및 실제 코드 계약을 따른다.

OpenFOAM 명령 구조와 Codex 기능 검토에 참고한 공식 자료는 다음과 같다. 문서 확인은 실제 설치/실행 검증을 대신하지 않는다.

- OpenFOAM v13 parallel running: https://doc.cfd.direct/openfoam/user-guide-v13/running-applications-parallel
- OpenFOAM v13 command-line post-processing: https://doc.cfd.direct/openfoam/user-guide-v13/post-processing-cli
- OpenFOAM v13 decomposePar source: https://cpp.openfoam.org/v13/decomposePar_8C_source.html
- OpenFOAM v13 reconstructPar source: https://cpp.openfoam.org/v13/reconstructPar_8C_source.html
- Codex authentication: https://developers.openai.com/codex/auth
- Codex non-interactive mode: https://developers.openai.com/codex/noninteractive

**최종 판정:** 소프트웨어 경계의 수정 및 359개 Python 회귀는 확인했다. 실제 OpenFOAM·MPI·live-model 평가가 남아 있는 **4.0.0rc1**이다. 자산, 체크포인트, 지역별 검증, 병렬·범용 후처리의 코드가 있다는 이유만으로 해당 기능의 실환경 완료를 선언하지 않는다.
