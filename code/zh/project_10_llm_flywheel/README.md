# Project 10: LLM Flywheel

这个目录是 `project_10_llm_flywheel` 的 GitHub 精简版，保留了：

- `src/` 核心源码
- 环境配置文件

原始数据、处理产物、缓存和大文件已从发布版中移除。如需完整运行，请在本项目基础上按你的本地实验流程补充数据输入与生成产物目录。


## Repository Smoke Test

From the repository root, run this project's smoke check through the unified runner:

```bash
python scripts/run_all_project_smoke_tests.py --project P10
```

Expected output: a `P10: PASS` or `P10: FAIL` line plus a report in `smoke_reports/` with the failing command and stderr when a check cannot complete.
