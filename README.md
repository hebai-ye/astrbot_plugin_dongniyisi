# 懂你意思（astrbot_plugin_dongniyisi）

让 AI **听懂梗、会回梗、能通过视频学习新梗与人设**的 AstrBot 拟人增强插件。
不改 AstrBot / NapCat 任何源码，走插件公开 API；所有「学到的东西」先入候选库，**确认后才用**。

## 它解决什么

- 用户说的话里有梗 / 黑话 / 抽象表达时，AI 能理解「具体意思 + 潜台词 + 想表达的情绪」，而不是按字面回。
- AI 的回应有**分寸**：有时接梗、有时不接、有时反问，像真人一样，不会每句都甩梗。
- 把视频转发给 AI：抽帧看画面 + 语音转写听内容 + 提炼「梗候选」和「人物人设候选」，入库待审核，可慢慢充实 AI。

## 怎么装

1. 把本目录放到 AstrBot 插件目录：`data/plugins/astrbot_plugin_dongniyisi/`
2. 在 AstrBot 管理面板「插件」页重载/启用「懂你意思」。
3. 第一次加载会自动建梗库：`data/plugin_data/astrbot_plugin_dongniyisi/meme_library.db`
4. 视频学习需要 `ffmpeg`（推荐放便携版，把 `video.ffmpeg_path` 指到 `ffmpeg.exe` 或所在 bin 目录）。
5. 语音转写需要 AstrBot 里已配置 STT Provider（例如 SenseVoice / Whisper 类）；没配也能用，只是没有转写文本。

## 命令

| 命令 | 说明 |
| --- | --- |
| `/懂你意思` | 功能总览 |
| `/梗库` | 梗库统计（总数 / 确认 / 候选 / 回梗次数） |
| `/找梗 关键词` | 关键词/别名/释义检索 |
| `/梗详情 id` | 看一条梗的释义、用法、别名 |
| `/学梗 标题\|意思\|怎么用` | 手动教一个梗（直接确认，后两项可省） |
| `/回梗概率 0~0.95` 或 `auto` | 当前会话临时调接梗频率 |
| `/待审核` | 看视频学到的梗/人设候选（管理员） |
| `/确认梗 id` / `/拒绝梗 id` | 审核梗候选（管理员） |

管理员判定：AstrBot 管理员，或 `integration.admin_ids` 里配置的 QQ 号（默认 `3535085149`）。

## 工作原理

1. **理解**：`on_llm_request`（priority=120000）里，用当前会话的模型把用户这句话识别成结构化 JSON：
   `is_meme / memes[] / emotion / tone / intent / subtext / should_meme`。
   识别失败、超时、超频时自动回退「关键词 + 标点 + 梗库命中」的离线启发式，**绝不影响正常聊天**。
2. **决策**：回梗概率 = 基础概率 × 情绪系数 × 响应意愿 × 冷却 × 同梗衰减；再叠加硬规则：
   严肃/难过/求助不接梗、10 条内最多回 N 次、高风险梗只理解不模仿、刚接过梗要收一收。
   最终给出 `meme(接梗) / tease(反向玩梗) / followup(追问) / normal(正常回)` 四种处理。
3. **注入**：把很短的一段「这句怎么理解 + 相关梗 + 这次怎么回 + 注意事项」追加进 system_prompt，
   带 `【懂你意思·内部参考】` 标记，并要求模型不要复述这段；不覆盖其它插件注入的内容。
4. **视频学习**：收到视频消息 → 后台抽帧（≤8 张）+ 抽 16k 单声道音频 →
   视觉模型描述画面、STT 转写人声 → 文本模型输出「梗候选 + 人设候选 JSON」→ 入库候选 →
   QQ 里回一句「整理好了，用 /待审核 查看」。

## WebUI 管理页

插件自带页面 `pages/memes/index.html`：在 AstrBot 面板「插件 → 懂你意思」的页面入口打开，
可看统计、梗列表、候选审核（确认/拒绝）、人设池，以及删除。
页面调用插件 web_api：`memes/list`、`memes/review`、`memes/update`、`memes/delete`、
`personas/list`、`personas/review`、`stats`（均由面板登录态保护）。

## 配置要点

- `reply.base_probability`：接梗基础概率（默认 0.30，真人感建议 0.2~0.4）。
- `reply.max_per_10` / `cooldown_sec` / `repeat_decay`：控制「别每句都接梗、别把同一个梗用烂」。
- `detector.rate_limit_per_min`：识别调用限流（防烧 token）。
- `learning.auto_learn`：自动收录候选（默认开；不会直接用于聊天，要确认）。
- `video.max_frames_vision` / `max_per_hour`：视频学习成本与频率上限。
- `persona.mode`：`off` 只存不注入 / `reference` 参考表达 / `absorb` 确认后吸收（默认 reference）。

## 隐私与安全

- 插件只读「触发它的消息」和视频抽帧/音频，不主动上传聊天记录；梗库存本机 SQLite。
- 注入段是内部参考，要求模型不复述；若发现模型偶尔复述，可在 `_build_injection` 里把说明写得更硬或调小 `inject_max_chars`。
- 高风险梗（risk=high）只理解不模仿；不接入任何违规内容生成。
- 视频学习默认「只入库、手动确认」，避免一条视频把人设带跑偏。

## 调试

日志关键字 `[MemeSense]`：
- `inject umo=... mode=... p=... emotion=...`：每次注入的决策
- `视频学习入库 umo=... memes=N persona=True/False`
- `视频学习失败` / `STT 失败` / `视觉描述失败`：失败原因

## 版本

- 0.2.0：LLM 结构化识别 + 四模式回梗决策 + 梗库/人设池 + 视频学习（抽帧/转写/提炼）+ 命令 + WebUI 页
- 0.1.0：骨架
