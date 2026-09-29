# 巡天智能体 · formal-agent

GOSIM 黑客松「Agentic Observer」正式赛智能体（v1，spec 全级达成：M1/V0/V1/V2/V3）。

## 提交包

`formal-agent/` 即提交 ZIP 的内容（`observer.project.json` + `src/*.py`），
纯标准库（python:3.12-slim），无第三方依赖。

## 运行

```bash
python3 -u src/main.py
```

- 协议：stdin/stdout JSON（initialize / decision / teardown）。
- 模型层：读取平台注入的 `OPENAI_BASE_URL` / `OPENAI_API_KEY` / `OPENAI_MODEL`；
  缺省或 `MODEL_PROVIDER=deterministic` 时纯确定性运行（官方场景实测两者总分逐项一致）。
- 保护：≤400 次调用 / 剩余墙钟 <600 s 停用模型层 / 连续 4 次失败熔断，任何情况下
  确定性热路径保底。

## 官方场景本地实测（2026-09-29）

| 场景 | 总分 | report |
|---|---|---|
| finals-preview | 6715.32 | 100 |
| dev-fortnight | 6699.64 | — |
| dev-reference | 12944.36 | — |
| V1 压力场景（8×200） | 134456.37，Jain 均匀度 0.8674，reqmiss 0 | — |

架构与验证细节见仓库外内部文档（不入库）：规格 §4 热路径 / §6 LLM 层 / §11 风险表。
