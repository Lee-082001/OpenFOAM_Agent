# OpenFOAM Agent v4.0.1 구현·검증 보고서

**작성일: 2026년 9월 7일 · 기준: v4.0.0rc1 · 판정: 소프트웨어 회귀 검사 통과, 실제 CFD·MPI·격리 환경 미검증 후보판**

## 1. 이번 수정의 결론

사용자가 지정한 여섯 가지 제한을 단순 문서 항목으로 남기지 않고 실행 경로에 연결했다. 자동 파일별 작성 분할, 모든 생성·수리 경로의 문법 근거 검사, 전이 의존 관계 기반 mesh 근거 무효화, 선택형 Linux 격리와 합산 예산, 보존형 로컬 MPI 재개, 저장된 필드에 기반한 물리량·보존식 검사를 추가했다.

**전체 447개 테스트가 한 번의 pytest 실행에서 통과했다. 실패·오류·skip은 0개다.** rc1의 359개 테스트 ID를 모두 유지했고 신규 88개를 추가했다. 다만 이는 실제 OpenFOAM 해석 성공률이나 운영용 보안 인증을 의미하지 않는다. 이번 환경에는 OpenFOAM, MPI, Bubblewrap 및 모델 CLI가 없었으며 실제 CFD·MPI·격리 커널 실행·live LLM 호출은 모두 미수행이다.

기존 v4.0.0rc1 ZIP은 별도로 보존했다. 기준 해시는 다음과 같다.

```text
v4.0.0rc1 ZIP
 e4c3ae9c0bda1eeb07e586a413233b8d6c7cf4f5abd84b70cf3671582fe075ae
v3.6.0 ZIP
 3d9d99c8b186db5ed429ac36a6c49104052ae2b0c5c4b25b73e61a3b7eefb9ac
```

패치 기준은 위 ZIP의 실제 파일이다. 사용자의 별도 Git 변경이나 rc1 체크포인트를 자동 병합·이전했다고 주장하지 않는다. 이전 33개 감사 전체를 종결한 배포도 아니다. `docs/V4_0_1_AUDIT_UPDATE.json`은 이번 여섯 영역을 원래 감사 번호에 연결하며, `all_33_completed`는 계속 `false`다.

## 2. 구현과 검증 수준 요약

| 영역 | 이번에 연결한 동작 | 실제 확인한 범위 | 아직 확인하지 않은 것 |
|---|---|---|---|
| 자동 컨텍스트 분할 | 파일별 작업 배정, 확정 fact·의존 조건·문법 본문 투영, 전체 조립 전 반영 금지, 큐 체크포인트 | 자동 분할을 유발하는 실제 controller 호출, 작업 ID·파일 누락·중복·조기 native 차단 | live 모델의 대규모 작성 품질 |
| 모든 작성 경로의 문법 근거 | bundle과 primitive write/patch에 공통 gate, legacy·revision·runtime repair 연결 | 각 phase의 무근거 쓰기 거부, 실제 reference read 후 허용, 분리된 read window 보존 | 실제 OpenFOAM 문법·선택 모델의 호환성 |
| mesh 의존 DAG | 작업의 입력·출력·부모·영역 등록, 전이 closure, include·추가·삭제·변경 반영 | 자산→변환→mesh 전이, 영역 독립성, 공유 자료 변경, 그래프 변조 거부 | 임의 외부 바이너리의 실제 파일 접근 추적 |
| OS 격리·합산 quota | 선택형 Bubblewrap/cgroup v2, 제한된 writable volume, 합산 CPU·메모리·프로세스·시간·출력 예산 | 합성 subprocess의 실제 출력·디스크·시간 제한, 잠금, 설정·명령 생성, 부적합 환경 거부 | 실제 namespace/cgroup 격리·OOM·MPI 전체 종료 |
| 병렬 restart | 공통 완전 시각 선택, 후속 불완전 결과 보관, 입력 해시, 재승인, 재분할 생략 | 다중 rank fixture, 손상·차원·개수·별칭 거부, 실제 CLI의 모델 없는 준비, 복원 상태 | 실제 decomposePar/mpirun/reconstructPar 실행 |
| 물리 의미·보존성 | native field 차원·종류·영역·개수·기하 검증, 공간/시간 연산, 구간별 수지·양측 interface 검사 | 해석적으로 값이 알려진 synthetic cube/필드, 질량·에너지 수지, 잘못된 부호·단위·오래된 결과 차단 | 실제 해석의 PDE 완전성, 모든 물리 모델의 정확성 |

표의 ‘실제 controller/CLI’는 Python 코드 경로를 실행했다는 뜻이다. OpenFOAM 이름을 가진 fixture나 모의 transport를 진짜 OpenFOAM 실행으로 세지 않았다.

## 3. 항목별 구현 내용

### 3.1 자동 컨텍스트 작업 분할

`engineering/authoring_tasks.py`와 `engineering/agent.py`가 staged authoring의 문자 예산 초과를 처리한다. 컨트롤러는 전체 확정 계획을 보유한 채, 파일을 정확히 한 작업에 배정한다. 해당 파일의 확정 fact와 전이 의존 fact를 포함하고, 파일로 매핑되지 않은 조건·실행 topology·영역 interface는 전역 조건으로 보존한다. 문법 근거는 그 파일에 필요한 실제 읽기 본문과 해시를 전달한다.

기존 intake 투영에서 context fact를 제외하면서 남을 수 있던 `depends_on`의 끊어진 연결도 보완했다. 필요한 확정 provenance fact만 `provenance_dependencies`로 따로 전달하며 원시 대화 전체를 내려보내지 않는다.

작업 ID나 파일 목록이 다르거나, 중간 작업이 native 실행을 시도하거나, 작업 중 확정 계획이 바뀌면 채택하지 않는다. **모든 작업 응답이 모이기 전에는 case 파일 반영과 native 실행을 하지 않는다.** 전체 조립 이후 기존 bundle 사전 검증·transaction 경로로 넘긴다. 부분 응답은 체크포인트에 보존한다.

자동 분할도 기존 LLM/action 예산을 사용하며 상한을 몰래 늘리지 않는다. 한 파일 자체 또는 반드시 반복해야 하는 전역 계약만으로 예산을 초과하면 여전히 호출 전에 차단한다. 설계 단계 자체의 임의 분할이나 단일 파일 본문의 조각 생성은 구현 범위가 아니다. CLI에서는 staged 경로가 켜져 있으며, 라이브러리 사용자는 `compact_phase_schemas=True`, `staged_case_authoring=True`를 명시해야 한다.

### 3.2 legacy를 포함한 동일한 문법 근거 강제

`contracts/evidence.py`의 `require_authoring_evidence`를 전체 bundle의 반영 전 검사와 중앙 write/patch dispatcher에 연결했다. staged, legacy execute/sequence, 사용자 revision, runtime repair가 같은 검사를 거친다. 실행 승인이나 문법 근거를 우회하는 ‘허용’ 스위치는 추가하지 않았다.

`ReadReferenceAction.target_case_files`로 대상 파일을 연결하거나 primitive 작성의 `evidence_ids`로 이미 읽은 자료를 명시할 수 있다. 검색 snippet이나 존재하지 않는 ID만으로는 통과하지 않는다. 파일을 아직 지정하지 않은 실제 read도 legacy 작성 프롬프트에 본문을 포함하므로, 다음 stateless 호출이 ID만 보고 문법을 기억한다고 가정하지 않는다. 같은 reference의 서로 다른 읽기 구간도 긴 하나로 덮어 버리지 않고 각 구간의 hash·record ID·범위를 유지한다.

이 검사는 **실제로 관찰한 문법 자료가 작성 호출에 전달되었는지**를 보장하는 계약이다. 자료의 버전·모델 적합성과 생성 파일의 native 문법 성공까지 같은 의미로 표시하지 않는다. 사용자 승인 자산 가져오기는 별도 소유권·해시 경로이며 LLM 쓰기 우회 수단이 아니다.

### 3.3 의존 관계 기반 mesh 무효화

`contracts/mesh_dependencies.py`는 `mesh-dependencies.json`에 작업별 입력·출력·영역·명령과 이전 생산 작업을 기록한다. 동일 파일을 재작성하는 작업은 시간 순서가 있는 노드로 표현해 순환 그래프를 만들지 않는다. 검증 digest는 조상 closure, 실제 파일 내용, 디렉터리에 추가·삭제된 입력, case 내부 literal include 및 누락 파일 sentinel을 반영한다. 체크포인트는 그래프 자체의 해시도 확인한다.

중앙 파일 변경 및 native mesh 변경 경로에서 영향받은 영역의 checkMesh 근거를 폐기하고 seal·실행 승인을 무효화한다. 입력 관계를 확실히 아는 제한된 blockMesh 호출 외에는 **전체 case 입력을 의존성으로 취급**한다. LLM이 좁은 읽기 목록을 주장해서 검증을 재사용할 수 없다.

따라서 지원 입력 범위 내 전이 무효화는 연결했지만, 임의 바이너리의 외부 파일 읽기를 자동 추론하는 시스템은 아니다. 모르는 도구는 검증을 더 수행하는 방향으로 보수적으로 처리한다. include의 의존성을 읽는 것과 기존 보안 정책상 금지된 include를 LLM이 새로 작성하도록 허용하는 것은 별개다.

### 3.4 Linux 격리와 전체 프로세스 집합 예산

`tools/linux_isolation.py`, `safe_runner.py`, `exec_guard.py`에 선택형 strict backend를 추가했다. `--isolation-policy`가 지정되면 비특권 사용자, 설치된 Bubblewrap, cpu/memory/pids가 위임된 cgroup v2, `cgroup.kill`, 별도 용량 제한 파일시스템을 요구한다. 요건이 없으면 격리 없는 실행으로 바꾸지 않고 거부한다.

namespace는 mount·PID·user·IPC·UTS·network·cgroup 보기를 제한한다. 공개 runtime 경로와 운영자가 지정한 OpenFOAM 경로만 읽기 전용으로 노출하고, case와 임시 파일·공유 메모리는 제한된 writable volume에 둔다. 실행 wrapper는 exec 전에 cgroup에 들어가므로 MPI launcher와 자식 rank도 같은 합산 한도를 적용받도록 구성했다.

메모리·swap·프로세스 수·CPU 사용률은 cgroup 설정에 연결하고, 누적 CPU 시간·wall time·출력량을 원장에 기록한다. 개별 파일 제한만으로 전체 디스크 quota라고 부르지 않는다. 별도 파일시스템의 총 용량을 확인하고 여러 작은 파일도 검사한다. 동일 workspace의 native 실행에는 잠금을 사용한다. 과거 합산 CPU 사용량을 알 수 없으면 0으로 간주하지 않는다.

**기본 로컬 모드는 여전히 OS sandbox가 아니다.** 실제 보고서에는 요청한 mode와 관찰된 process mode를 구분한다. strict 모드의 실환경 검증은 수행하지 않았다. 현재 실행 환경은 UID 0이고 Bubblewrap이 없으며 cgroup에 쓰기 권한이 없다. 이런 상태를 성공적인 격리 검사로 표시하지 않는다.

CPU/wall 모니터링은 polling 기반이므로 작은 초과 시간이 가능하다. 디스크는 현재 저장 용량 제한이지 누적 I/O 대역폭 제한이 아니다. GPU·scheduler·I/O rate 등의 모든 자원을 망라하는 운영 플랫폼을 구현했다고 주장하지 않는다. 학교 서버 권한·네트워크 설정을 변경하거나 sudo 작업을 자동 실행하지 않았다.

### 3.5 로컬 MPI 재시작

`runtime/restart.py`가 승인된 rank 수의 `processorN` 디렉터리를 확인하고 모든 rank에 공통으로 존재하는 완전한 저장 시각을 선택한다. static mesh의 연결·개수·patch, 필드의 class·차원·원소 수·유한성, 초기 sealed 필드와의 일치, decomposition·module·arguments·intake 해시를 대조한다.

더 최신 시각이 불완전하면 그 이전의 완전한 공통 시각을 선택하고 이후 결과는 `restart-history/`로 이동해 보존한다. 변경 전에 intent journal을 기록하므로 중간 중단은 자동 재실행하지 않는다. controlDict는 top-level `startFrom`·`startTime`만 바꾸며 원래 요청한 시작·종료 조건은 완료 계약에 보존한다. 선택한 snapshot receipt를 검증한 뒤에만 실제 진행 구간의 시작점으로 사용한다.

준비 명령은 모델 인증 없이 실행되며 solver를 시작하지 않는다. 이후 상태는 `SOLVE_READY`이고 기존 승인은 폐기한다. 재승인한 실행에서는 입력을 다시 대조하고 decomposePar를 건너뛴다. 살아 있거나 재사용된 PID, 불명확한 프로세스 상태, 비어 있지 않은 cgroup은 임의로 종료하거나 완료로 간주하지 않는다.

재구성은 최종 시각 외에도 native 물리량·보존식에 필요한 양의 저장 시각을 포함한다. 원래 입력인 `0/`는 덮어쓰지 않는다. 시각 0의 분석에는 원래 저장 필드가 있어야 하고, rank별 요청 시각이 누락되면 재구성을 시작하지 않는다.

지원 범위는 로컬 MPI·static mesh·uncollated processorN·지원 class의 bounded ASCII/gzip 필드다. binary/collated/distributed I/O, 이동 topology, 숨겨진 사용자 정의 restart 상태, Slurm, rc1 체크포인트 자동 migration은 지원 완료로 표시하지 않는다. 실제 MPI 실행은 미수행이다.

### 3.6 후처리의 물리 의미와 보존식 검사

`postprocessing/native_fields.py`는 실제 저장된 필드와 mesh를 읽는다. 7개 기본 SI 차원, field class, literal 영역·patch, 원소 개수, 유한값을 확인하며, 면적·부피 연산에 필요한 mesh의 양의 방향·기하학적/위상학적 닫힘을 검사한다. 공간 연산은 signed patch sum, 면적/부피 평균·적분 및 cell 극값을 지원한다. 시간 연산은 외삽 없는 piecewise-linear 가정에 따른 평균·RMS·적분이다.

질량 유량과 체적 유량, Pa와 운동학적 압력을 이름만 보고 같다고 취급하지 않는다. 원본 필드·mesh 해시와 물리량 계약을 결과에 연결해, 분석 후 파일이나 정의가 바뀌면 기존 결과를 보고서에서 제거한다. 일반 CSV의 산술은 계속 `physical_semantics_verified=False`로 남긴다.

`postprocessing/conservation.py`는 다음 식을 각 저장 시각 구간에서 계산한다.

```text
d(storage)/dt + sum(outward boundary flux) - integrated source = 0
```

질량·에너지·체적 수지는 각각 맞는 차원을 요구한다. 모든 경계 patch의 outward flux를 포함하며 저장량 density와 source density를 부피 적분한다. steady storage와 zero source는 명시적으로 승인된 가정이어야 한다. 영역 interface는 양측 바깥 방향 flux의 합을 각 시각에서 확인한다. **서로 반대 부호의 큰 오차가 전체 평균에서 상쇄되어 통과하지 않게 모든 구간·interface가 기준을 만족해야 한다.**

요청한 물리량·보존식이 누락되거나 stale이거나 기준을 위반하면 보고서는 성공하지 못하고 높은 신뢰도로 표시되지 않는다. 에너지 저장량 증가와 열유량의 평형, 질량 저장·유입·source 관계, 잘못된 단위·부호·누락된 상대 영역을 synthetic fixture로 검사했다.

`physical_semantics_verified=True`는 **선언한 관측량의 field/차원/선택/기하 연산과 실제 저장 데이터의 일치** 범위다. 모든 PDE source 항의 완전성, 모델 선택의 정확성, pointwise interface 연속성 또는 저장하지 않은 모든 timestep의 보존을 증명하지 않는다. `governing_equation_completeness_verified=False`는 유지한다. 지원 geometry는 static reconstructed ASCII/gzip이며 planar convex face 등 검사 가능한 부분집합 밖에서는 추정 계산 대신 거부한다.

## 4. 실행한 테스트와 재현 자료

| 검사 | 결과 |
|---|---:|
| 수정 전 rc1 기준 회귀 | 359 passed |
| 수정 후 유지된 rc1 테스트 ID | 359개 |
| 신규 context·문법 근거·DAG | 27 passed |
| 신규 실행 제한·격리 계약·MPI 재개 | 27 passed |
| 신규 물리량·보존식 | 34 passed |
| 단일 전체 pytest 실행 | **447 passed / 0 failed / 0 error / 0 skipped** |
| collection과 JUnit 테스트 ID 대조 | 누락·중복 없음 |
| compileall 및 JSON 예제 schema 검사 | 통과 |
| wheel 설치 후 소스 밖 CLI·13/14 resource | 제한된 환경에서 통과 |

`verification/v4_0_1/pytest.xml`, `pytest.log`, `collected_tests.txt`, `test_summary.json`, `tested_inputs_sha256.json`에 근거를 보존했다. 신규 테스트와 원래 33개 감사의 연결은 `docs/V4_0_1_AUDIT_UPDATE.json`에 있다. 이전 버전의 `verification/` 루트 파일은 rc1 이력이며 rc2 결과로 합산하지 않는다.

기존 테스트 파일 **7개**를 수정했다. 주된 변경은 positive fixture에 실제 synthetic 문법 읽기 근거를 제공하고, 무근거 거부 테스트에서는 이를 명시적으로 제외한 것이다. 버전 기대값과 커진 구조화 schema의 예산 기대값도 변경했다. `test_v360_bounded_staged_context.py`의 design/legacy 비율 상한은 0.75→0.76, 해당 approximate token 상한은 18,000→20,000으로 바뀌었다. **기존 테스트가 변경 없이 통과했다거나 모든 경로에서 토큰이 줄었다고 주장하지 않는다.** `test_migration.json`과 패치에서 정확한 변경을 확인할 수 있다.

새 테스트는 문법 gate 자체를 비활성화하지 않는다. 문법 fixture는 테스트용 파일의 실제 바이트를 읽은 근거이며 OpenFOAM 공식 튜토리얼 또는 native 문법 검증으로 표시하지 않는다. 자동 분할 통합 테스트는 controller가 최종 transaction 호출을 한 번만 하는 경계를 관찰하며 그 테스트에서 CFD를 실행하지 않는다.

wheel은 `--no-deps`로 설치했다. 처음의 별도 venv 실행은 pydantic이 없어 실패했고, 기존 환경 의존성 경로를 `.pth`로 재사용한 후 CLI와 package resource를 검사했다. wheel에서 import된 경로도 확인했다. OpenAI SDK는 없었으며 새 환경의 의존성 해결, Python 3.12·Windows/WSL 설치, 실제 인증은 검증하지 않았다. 초기 실패와 성공 기록을 모두 `verification/v4_0_1/`에 보존한다.

## 5. 적용 및 사용

새 폴더에 전체 ZIP을 푸는 방법을 권장한다. 기존 작업 폴더나 결과를 덮어쓰지 않는다. 깨끗한 rc1 프로젝트 루트에서는 다음 패치를 사용할 수 있다.

```bash
git apply --check ../OpenFOAM_Agent_v4.0.0rc1_to_v4.0.1.patch
git apply ../OpenFOAM_Agent_v4.0.0rc1_to_v4.0.1.patch
```

깨끗한 v3.6.0에는 누적 패치 `OpenFOAM_Agent_v3.6.0_to_v4.0.1.patch`를 적용한다. 두 패치를 같은 기준본에 연속 적용하는 방식이 아니다. `--check`가 실패하면 강제 적용하지 않는다. 배포 ZIP·패치·보고서·검증 JSON의 해시는 `OpenFOAM_Agent_v4.0.1_SHA256SUMS.txt`에 있다. 두 기준본에 패치를 재적용한 파일 내용·실행 권한 대조 결과는 별도 `OpenFOAM_Agent_v4.0.1_package_verification.json`을 기준으로 확인한다.

병렬 재개 준비는 다음과 같이 수행한다.

```bash
openfoam-agent --resume /path/to/run --prepare-parallel-restart latest --capability-db config/openfoam13_capability_graph.json --json
```

준비 결과와 `parallel-restart.json`, 보관된 후속 결과를 검토한 뒤에만 새로 승인한다.

```bash
openfoam-agent --resume /path/to/run --solve --backend codex --capability-db config/openfoam13_capability_graph.json
```

`--prepare-parallel-restart`와 `--solve`를 한 호출에 함께 주지 않는다. rc1 소스 업데이트는 rc1 실행 중 체크포인트가 무조건 호환된다는 뜻이 아니다. 기존 checkpoint·case·로그는 반드시 보존한다.

OS 격리를 사용하려면 운영자가 먼저 준비한 정책을 지정한다.

```bash
openfoam-agent --interactive --backend codex --capability-db config/openfoam13_capability_graph.json --workspace /mnt/ofa-bounded-volume/runs --isolation-policy /path/to/isolation.policy.json
```

`examples/v4rc2/`에 schema 검사를 통과한 격리 정책·native 온도 물리량·질량 보존식 예제를 넣었다. 이 예제 경로는 서버에 자동 생성되지 않으며 실제 field 이름·저장 시각·영역·자원 상한에 맞게 설정해야 한다. `rhoPhi` 예시는 파일의 존재나 차원을 보장하지 않으므로 실제 데이터가 없으면 검사는 거부한다. 상세 운영 계약은 `docs/V4_0_1_OPERATIONS.md`에 정리했다.

## 6. 실제 환경에서 별도로 통과해야 할 검증

배포 사용 전에는 실제 OpenFOAM 13에서 dictionary 작성→mesh→checkMesh→solver→결과 필드 검사를 수행해야 한다. 이어서 로컬 MPI의 의도적 중단·공통 시각 재개·요청 시각 재구성, 실제 geometry의 공간 적분, 다중 영역 열유량 수지, 지원 밖 형식의 거부를 확인해야 한다.

strict Linux 모드는 비특권 계정·실제 위임 cgroup·용량 제한 volume에서 host 비공개 파일/네트워크 접근 차단, aggregate OOM·pids·CPU·저장량 한도와 전체 자식 종료를 검사해야 한다. 이번의 설정/명령 검사나 로컬 합성 subprocess 결과로 이 qualification을 대체하지 않았다.

## 7. 구현 시 확인한 공식 근거

아래 자료는 기능 계약과 형식 확인에 사용했다. 이를 읽었다는 사실은 native 실행 테스트가 아니다.

- Linux cgroup v2: https://www.kernel.org/doc/html/latest/admin-guide/cgroup-v2.html
- Bubblewrap 옵션: https://raw.githubusercontent.com/containers/bubblewrap/main/bwrap.xml
- OpenFOAM 13 병렬 실행: https://doc.cfd.direct/openfoam/user-guide-v13/running-applications-parallel
- controlDict: https://doc.cfd.direct/openfoam/user-guide-v13/controldict
- mesh 설명과 파일: https://doc.cfd.direct/openfoam/user-guide-v13/mesh-description · https://doc.cfd.direct/openfoam/user-guide-v13/mesh-files
- 필드 형식과 차원: https://doc.cfd.direct/openfoam/user-guide-v13/basic-file-format
- reconstructPar / timeSelector: https://cpp.openfoam.org/v13/reconstructPar_8C_source.html · https://cpp.openfoam.org/v13/timeSelector_8H_source.html