# 每日论文推荐工具

该工具每天抓取最新论文，先按感知算法方向、热点信号和质量信号做候选召回，再调用本地 Codex 判断候选论文是否值得推荐，并生成中文业务推荐理由，最后写入飞书多维表格后推送飞书群。当前方向覆盖检测/开放词汇检测、跟踪/ReID、关键点/姿态、自动标注/数据清洗/数据引擎、图像分类/场景识别/人脸属性分类、小模型/部署/剪枝/量化、视频理解/视频感知。

## 运行前准备

1. 安装依赖：

```bash
python3 -m pip install -r requirements.txt
```

如果希望阅读报告尽量基于 PDF 正文，推荐安装系统级 `pdftotext`。未安装时会尝试使用 Python 依赖 `pypdf`，仍失败则自动降级为摘要版报告。

2. 配置环境变量：

```bash
cp .env.example .env
```

填写：

- `FEISHU_APP_ID`
- `FEISHU_APP_SECRET`
- `FEISHU_BITABLE_APP_TOKEN`
- `FEISHU_BITABLE_WIKI_TOKEN`，当多维表格链接是 `feishu.cn/wiki/...` 时填写 wiki token
- `FEISHU_BITABLE_TABLE_ID`
- `FEISHU_BITABLE_URL`
- `FEISHU_WEBHOOK_URL`
- `FEISHU_SIGN_SECRET`，如自定义机器人启用签名
- `SEMANTIC_SCHOLAR_API_KEY`，可选
- `PAPER_REPORT_BASE_URL`，GitHub Pages 根地址，例如 `https://<owner>.github.io/<repo>`。飞书群卡片和多维表格中的“阅读报告”链接会基于它拼出 HTML 页面地址。
- `GITHUB_PAGES_REPOSITORY` / `GITHUB_PAGES_TOKEN`，用于把报告 bundle 更新到 GitHub Pages 仓库。也可在 `config.yaml` 中配置 `reports.github_repository`。

3. 创建飞书多维表格字段。

如果使用 `FEISHU_BITABLE_WIKI_TOKEN`，飞书自建应用还需要开通任一 wiki 节点读取权限，例如 `wiki:node:read`、`wiki:wiki:readonly` 或 `wiki:wiki`。

推荐字段均用文本字段即可，`推荐分` 可用数字字段：

```text
推荐日期、首次发现时间、最近更新时间、arXiv ID、DOI、标题、作者、机构、
论文链接、代码链接、类别、关键词、推荐分、推荐等级、一句话结论、核心贡献、
技术要点、实验亮点、落地价值、风险限制、建议动作、是否已推送
```

可选追溯字段：`展示标题`、`推荐钩子`、`热点信号`、`质量信号`、`为什么现在看`、`质量判断`、`热点判断`、`可验证依据`、`推荐原因`、`主任务方向`、`业务相关度`、`工程落地性`、`命中关键词`、`Codex推荐判断`、`阅读报告链接`、`报告生成时间`、`报告状态`、`是否读取全文`。如需写入这些字段，需要先在多维表格中创建列，并在 `config.yaml` 的 `feishu.field_names` 中增加映射。

## 本地命令

只跑候选抓取、规则排序和兜底摘要，不写飞书：

```bash
python3 tools/paper_recommender/run_daily.py --dry-run --no-codex
```

无外网时演练后半段流程：

```bash
python3 tools/paper_recommender/run_daily.py --dry-run --no-codex --sample
```

只验证飞书 token、wiki 链接转换和表格读取权限：

```bash
python3 tools/paper_recommender/run_daily.py --check-feishu
```

跑完整 Codex 业务评审和推荐生成，但不写飞书：

```bash
python3 tools/paper_recommender/run_daily.py --dry-run
```

写入飞书多维表格但不推群：

```bash
python3 tools/paper_recommender/run_daily.py --no-send
```

强制重新生成 HTML 报告、写入追溯表格，但不推送飞书群消息：

```bash
python3 tools/paper_recommender/run_daily.py --force-reports --no-send
```

只发布当前 `reports/` 目录到配置的报告发布目标，不重新抓论文、不写飞书、不推群：

```bash
python3 tools/paper_recommender/run_daily.py --publish-existing-reports
```

完整每日流程：

```bash
python3 tools/paper_recommender/run_daily.py
```

正式推送飞书群之前，脚本会先生成论文报告。默认配置下会调用本地 `paper-craft-skills` 的 `paper-analyzer` 子技能生成 HTML 报告，并把 `reports/` 打包成 `reports_bundle.tgz.b64` 更新到 GitHub 仓库 `main` 分支。GitHub Pages workflow 会解包 bundle 并发布 plain HTML 静态站点；GitHub Pages 发布或 URL 校验失败时会中断流程，不发送飞书群消息，避免群卡片里的“阅读报告”链接指向不存在的文件。默认流程不再提交或推送 GitLab `paper-reports` 分支。

生成报告会输出到：

```text
reports/README.md
reports/data/reports.json
reports/data/state.json
reports/index.html
reports/reports/<arxiv_id>/index.html
reports/assets/<arxiv_id>/figure-*.png
```

飞书群卡片里的“阅读报告”按钮指向 GitHub Pages HTML 页面，例如：

```text
${PAPER_REPORT_BASE_URL}/reports/<arxiv_id>/index.html
```

报告会优先下载 PDF 全文，使用带页码的全文文本交给 `paper-craft-skills` 生成深度 HTML 长文；如果安装了 `PyMuPDF` 或系统 `pdfimages`，会 best-effort 从 PDF 中抽取图像到 `reports/assets/<arxiv_id>/`，并把可用于 HTML 的相对路径传给技能。报告发布前会检查内部打分信号泄露、页码证据和图表路径；如果 `paper-craft-skills` 输出不是完整 HTML，脚本会回退到内置 HTML 模板并标记为“需复核”。

强制重新生成已有报告：

```bash
python3 tools/paper_recommender/run_daily.py --force-reports
```

## 定时任务

北京时间每天 09:30：

```cron
30 9 * * * cd /mnt/d/faceid/code/paper-share && python3 tools/paper_recommender/run_daily.py >> logs/paper_recommender.log 2>&1
```

## 调整研究方向

编辑 `tools/paper_recommender/config.yaml`：

- `arxiv.categories`：arXiv 类别。
- `sources.huggingface_papers`：Hugging Face Daily/Trending Papers 热点源，会补充 upvote、GitHub stars、代码链接和项目页；失败时自动降级，不中断主流程。
- `sources.cvf_openaccess`：CVF Open Access 会议论文源，默认接入 CVPR 2026；可继续添加 ICCV 2025、ECCV 2026、WACV 2026 等会议页面，并用 `max_results` 控制每天进入候选池的数量。
- `sources.cvf_openaccess.detail_limit`：从 CVF 详情页补摘要的数量；CVF 页面较慢，默认只补前 8 篇，其余先用标题和 venue 信号参与排序。
- `ranking.keyword_tiers.A`：A 档强相关关键词，优先推荐。
- `ranking.keyword_tiers.B`：B 档高相关关键词，联合命中时优先。
- `ranking.keyword_tiers.C`：C 档专项关注关键词，用于扩大召回。
- `ranking.tier_weights`：标题、摘要、元信息的分层权重。
- `ranking.negative_keyword_weights`：不相关方向降权词和权重。
- `ranking.final_limit`：每日推荐数量。
- `ranking.min_score`：入选最低分。
- `ranking.hotness_cap` / `ranking.quality_cap`：热点分与质量分上限。
- `prestige_venues.enabled` / `prestige_venues.quality_bonus`：开启重点会议/期刊召回，Semantic Scholar venue 命中 CVPR/ICCV/ECCV/NeurIPS/ICLR/ICML/TPAMI/IJCV 等默认白名单时，会增加质量分并优先进入 Codex 评审。
- `prestige_venues.candidate_bucket_limit`：每天最多优先送入 Codex 评审的重点会议/期刊候选数量。
- `codex.judge_candidate_limit`：交给 Codex 评审的规则入围候选数量。
- `codex.judge_min_rule_score`：进入 Codex 评审的最低规则分。
- `codex.judge_min_hotness_score` / `codex.judge_min_quality_score`：即使规则分不高，也会进入 Codex 评审的热点/质量候选阈值。
- `codex.judge_min_codex_score`：Codex 判定推荐后的最低业务分。
- `history.enabled` / `history.mode`：跨天历史过滤，默认 `pushed_forever`，已成功推送过的 arXiv ID 不再进入每日推荐；临时回放可用 `--allow-repeat`。
- `ranking.hard_negative_keywords`：强制过滤的无关方向关键词，例如 AI skill/agent 构建类论文。
- `reports.publisher`：主报告发布渠道，默认 `file`，不创建飞书文档。
- `reports.generator` / `reports.paper_craft_style`：报告内容生成器与 paper-craft paper-analyzer 风格；默认使用 `paper_craft` + `academic` 生成 HTML 精读长文。
- `reports.paper_craft_skill_path`：本机 paper-craft paper-analyzer 子技能路径，默认 `/home/wjw/.codex/skills/paper-analyzer`。
- `reports.archive_markdown`：是否保留 Markdown 归档，默认开启。
- `reports.quality_gate`：是否启用报告质量闸门，默认开启。
- `reports.format`：报告格式，默认 `html`。
- `reports.output_dir`：报告输出目录，默认 `reports`。
- `reports.base_url`：飞书按钮使用的 GitHub Pages 报告根地址；推荐用 `PAPER_REPORT_BASE_URL` 环境变量配置。
- `reports.read_pdf`：是否尝试下载 PDF 并抽取全文正文。
- `reports.pdf_text_chars`：传给 Codex 生成阅读报告的全文上限，默认 `80000`；超长论文会标记为“PDF全文截断与摘要”。
- `reports.extract_figures`：是否从 PDF 中抽取图片候选。
- `reports.max_figures`：每篇报告最多嵌入的图片数量。
- `reports.publish_target`：报告发布目标，默认 `github_pages`，即更新 GitHub Pages 仓库中的报告 bundle。
- `reports.publish_before_send`：发飞书群之前是否发布报告，默认开启。
- `reports.publish_on_no_send`：使用 `--no-send` 跳过飞书群消息时是否仍发布报告，默认开启。
- `reports.github_repository` / `reports.github_branch` / `reports.github_bundle_path`：GitHub Pages 发布目标，默认 `Jevin0924/paper-share`、`main`、`reports_bundle.tgz.b64`。

## 流程容错

- Semantic Scholar 请求失败只影响补充信息，不中断主流程。
- Hugging Face 热点源请求或解析失败只影响热点信号，不中断主流程。
- Codex 业务评审失败时回退到规则排序，并尽量使用 Codex 摘要；摘要失败时使用规则摘要兜底。
- PDF 下载或正文抽取失败时，阅读报告自动降级为“基于摘要与评审信息生成”。
- paper-craft HTML 生成失败或输出不完整时，阅读报告自动回退到内置 HTML 模板并标记为“需复核”。
- GitLab 报告分支推送失败时不写入飞书多维表、不推送群消息。
- 飞书多维表格写入失败时不推送群消息，避免群消息和追溯库不一致。
- 日志不会打印飞书密钥、webhook 或 app secret。
