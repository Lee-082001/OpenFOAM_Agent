# OpenFOAM Agent v4.0.3 Codex transport 및 evidence projection 수정 보고서

## 1. 재현된 문제

v4.0.2에서 `prepare_design` context partition은 정상적으로 동작했지만, 두 번째
Codex structured-output 호출에서 Codex가 `command_execution`, `web_search`, MCP/action,
`file_change` 또는 새로운 item event를 생성하면 Python transport가 호출 후 이를 발견하고
`StructuredOutputError`로 전체 ENGINEERING을 종료했다.

이 문제는 CFD 설계나 OpenFOAM 검증 실패가 아니라 Codex를 "model-only transport"로 사용하려는
OpenFOAM Agent의 의도와 Codex CLI가 제공하는 tool surface 사이의 계약 불일치였다.

## 2. v4.0.3 구현

### 2.1 Codex tool surface 사전 비활성화 요청

각 `codex exec` 호출에 `--disable <feature>`를 반복 전달한다. shell/unified execution,
code mode, web/search, apps/plugins, browser/computer-use, image generation, skill search,
multi-agent 계열을 model-only 호출에서 비활성화하도록 요청한다. 기존 `--ephemeral`,
`--ignore-user-config`, `--sandbox read-only`도 유지하며 `approval_policy="never"`를 강제한다.

이 설정은 사전 예방을 **요청**하는 계약이다. 서버 측 실제 tool 목록을 응답 JSON만으로 독립적으로
검증할 수는 없으므로, 성공했다고 해서 `pre_execution_tool_prevention_verified=True`로 과장하지 않는다.

### 2.2 사후 JSON event guard 유지

CLI 또는 모델 회귀로 tool event가 다시 나타나는 경우에는 계속 fail-closed로 차단한다. 다만 기존의
모호한 메시지 대신 다음 정보를 노출한다.

- event type (`item.started`, `item.completed` 등)
- item type (`command_execution`, `web_search`, `mcp_tool_call`, `file_change` 등)
- 가능한 경우 bounded command/query/tool/path diagnostic

따라서 다음 재현에서는 "tool/action or unknown"이 아니라 정확히 어떤 Codex item이 발생했는지를
바로 확인할 수 있다.

### 2.3 evidence batch 승격 상한

한 번의 evidence gap retrieval이 20~30개 이상의 후보를 찾더라도 모든 후보를 다음 LLM prompt에
전문으로 넣지 않는다. 검색 결과 전체는 `EngineeringEvidenceRecord`와 gap ledger에 보존한다.
그중 기본 6개만 relevance/fairness 정책으로 다음 model context에 승격한다.

우선순위는 대략 다음과 같다.

1. runtime/native 수준의 강한 capability 근거
2. 실제 reference read로 확보한 본문 excerpt
3. installed/source 수준 capability 근거
4. 검색 snippet 또는 discovery-only 근거

두 개 이상의 gap이 동시에 요청된 경우 한 gap이 전체 승격 예산을 독점하지 않도록 round-robin 방식으로
분배한다. 사용자는 `--engineering-new-evidence-items`로 1~12 사이에서 조정할 수 있다.

## 3. 검증 결과

- 전체 pytest: **455 passed**
- Codex command construction에서 model-only disable profile 포함 여부 확인
- command execution event의 정확한 command diagnostic 확인
- web search event의 정확한 query diagnostic 확인
- 정상 event stream에서 model-only profile 요청 상태와 사후 guard 상태 확인
- 100개 evidence retrieval에서 durable store 100개 보존 + 즉시 승격 6개 확인
- 승격되지 않은 retrieval candidate가 같은 bounded model turn에 fallback으로 재유입되지 않음을 확인

## 4. 미검증 범위

현재 빌드 환경에는 사용자의 ChatGPT/Codex 로그인과 실제 Codex CLI 0.153.0 세션이 없으므로,
실제 live call에서 tool surface가 완전히 제거되는지는 직접 실행하지 않았다. 따라서 다음 사용자 재실행에서
`StructuredOutputError`가 사라지는지 확인해야 한다. 만약 Codex가 disable profile을 무시한다면 새 오류는
정확한 `event`와 `item.type`을 출력하므로 다음 수정의 원인을 바로 식별할 수 있다.

실제 OpenFOAM solver, MPI, post-processing 역시 이번 transport 수정 검증의 범위가 아니다.

## 5. 권장 재검증

기존 실행 명령을 그대로 사용한다.

```bash
openfoam-agent --interactive \
  --backend codex \
  --confirm-api-calls \
  --skip-postprocess \
  --mode easy
```

이전 bypass flow 입력을 다시 사용했을 때 evidence retrieval 로그는 예를 들어 다음처럼 바뀌는 것이 정상이다.

```text
Evidence-gap batch completed: 28 retrieved; 6 promoted to model context (limit=6); stagnant=0.
```

두 번째 Codex 호출이 정상 structured JSON을 반환하면 engineering이 계속 진행해야 한다. tool event가 다시
발생할 경우에는 그 종류와 command/query가 새 diagnostic에 나타나야 한다.
