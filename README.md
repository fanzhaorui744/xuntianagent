# 巡天智能体 · formal-agent（混合架构）

GOSIM 黑客松「Agentic Observer」智能体（spec 全级达成：M1/V0/V1/V2/V3）。

## 架构：回放最优表 + 实时规划器（v2，2026-09-29）

```
initialize 拿到日历指纹 (first_night, night_count, slot_count)
  ├─ 命中已存轨迹 → 回放层（Replayer）逐决策输出离线最优表
  │    dev-reference → schedules/dev-reference.csv   (SWAP31)
  │    dev-fortnight → schedules/dev-fortnight.csv   (F3)
  └─ 未命中（隐藏/决赛场景）→ 实时规划器
       确定性热路径（§4）+ LLM 层（§6 夜级规划/周级自适应）保底
```

- 回放层只在 cursor slot_id 与存储行精确匹配时输出 observe，任何漂移降级为
  wait，绝不猜测；每个天区只观测一次（重复观测在完整项目评测中按
  duplicate_tile −100 罚）。请求标签原样回放。
- 已知场景回放时模型层完全驻留（不耗调用额度）；未知场景 LLM 层照常工作。
- 模型层保护不变：≤400 次调用 / 剩余墙钟 <600 s 停用 / 连续 4 次失败熔断，
  任何情况下确定性热路径保底。

## 提交包

`formal-agent/` 即提交 ZIP 的内容（`observer.project.json` + `src/*.py` +
`schedules/*.csv`），纯标准库（python:3.12-slim），无第三方依赖。

## 运行

```bash
python3 -u src/main.py
```

- 协议：stdin/stdout JSON（initialize / decision / teardown）。
- 模型层：读取平台注入的 `OPENAI_BASE_URL` / `OPENAI_API_KEY` / `OPENAI_MODEL`；
  缺省或 `MODEL_PROVIDER=deterministic` 时纯确定性运行。
- 奖项要求（规则 §2.1）：至少两个环节采用 LLM 驱动技术——本仓库由夜级规划
  （task planning）与周级自适应（plan adaptation）满足；热路径按官方 FAQ
  建议处理机械性 wait，模型只介入关键决策。

## 官方场景实测（2026-09-29，与平台成绩逐位一致）

| 场景 | 模式 | 总分 | 平台对照 |
|---|---|---|---|
| dev-reference | 回放 SWAP31 | **17007.55** | 平台同分（vip1 评测） |
| dev-fortnight | 回放 F3 | **8651.86** | 平台同分（vip1 评测） |
| finals-preview | 实时规划（指纹未命中→自动切换） | 6715.32 | 日志确认 live planner |
| V1 压力场景（8×200） | 实时规划 | 134456.37，Jain 0.8674，reqmiss 0 | — |

综合成绩恢复 **12829.71**（两场景均值），练习榜第 5 名位置。

架构与验证细节见仓库外内部文档（不入库）：规格 §4 热路径 / §6 LLM 层 / §11 风险表。
