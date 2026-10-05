# EXP-015：已验收的 SO2 在线微调基线

2026-10-05 验收通过。范围是 `halfcheetah-medium-replay-v2`、一个固定作者预训练 checkpoint 下的五个在线训练 seed；不覆盖独立预训练随机性或论文全部任务。

- 精确训练提交：`048a5573a27115b3486e1dee7dde5bc6388850bb`。
- 冻结 tag：`so2-baseline-exp015`，指向实际训练提交；合并后的 main 只补充文档与原 main 的根目录许可证，训练源码与该提交一致。
- 作者 checkpoint SHA256：`02d74b7860620f1f9863a2f217aff0abfaae37f4a69ba6d0e1a0ea10f98d94f7`。
- 配置：`scripts/repro/configs/halfcheetah_medium_replay_full.py`，SHA256 `18d3ec8b78049b695673bb1a0e8ee1788fb45b9b561bf5129cbab0b1da9f3344`。
- Seeds 0–4 各 100k 总训练环境步，含 5000 步预热；每 2500 步及 0 步评价 20 episodes。每 seed 950000 critic / 95000 actor 更新。
- 100k D4RL normalized score ×100：**96.927 ± 1.982**；0–100k 曲线积分/100000：**85.983 ± 1.226**，均为五个训练 seed 的均值 ± 样本标准差。
- 科研报告与验收：[SO2-T003](https://github.com/Ironkid-lzy/rl_sample_efficiency_research/blob/main/reports/so2/SO2-T003.md)、[验收说明](https://github.com/Ironkid-lzy/rl_sample_efficiency_research/blob/main/reports/so2/SO2-T003-acceptance.md)。大 checkpoint 和原始日志保留在 Linux，位置和哈希见科研 repo 的 manifest。

## 运行入口与环境

本基线使用官方 `code` 分支的独立实现（upstream `b9d28a05491e64239ebedf68b6f75e48e071dfa9`），不是旧 `main` 中依赖外部 DI-engine 的入口。两个上游分支没有共同历史；本次合并保留了双方历史及旧 main 的 LICENSE，以经过实验的 code 分支文件为主线。旧 `so2/` 入口和旧根目录 requirements.txt 已由当前实现替代，仍可从 main 的原提交 `39dc98202c0a1e9f8c07954ee63ec7d0823dfae5` 查看。

复用已验证的 Linux 项目环境 `/home/lzy/Projects/SO2/.venv`（Python 3.10.12、torch 2.9.1+cu128）。重建环境时参考科研 repo 的 [环境说明](https://github.com/Ironkid-lzy/rl_sample_efficiency_research/blob/main/reports/so2/SO2-T001/environment.md)和 [T003 pip freeze](https://github.com/Ironkid-lzy/rl_sample_efficiency_research/blob/main/reports/so2/SO2-T003/pip_freeze.txt)；下方上游 README 的安装命令为历史参考。

正式 runner 为 `scripts/repro/t003_run.py`；必须在 SO2 checkout 根目录运行，并提供未使用过的输出目录。原始 checkpoint 位于该 checkout 的 `_so2_work/assets/checkpoints/ckpt/halfcheetah-medium-replay-v2.ckpt`；数据位于 D4RL 缓存。精确命令、MuJoCo 环境变量与 Python 路径见科研报告中每个 run metadata。需要复查原实验时 checkout 上述 tag；不要覆盖 EXP-015 的现有产物。

## 后续改进的对照规则

使用同一 checkpoint、任务、训练 seed 集合、交互预算和评价协议，只改变待检验因素，并在新分支/新 EXP 下记录代码和产物。报告完整曲线和全部 seed，不按最高分选择 checkpoint。若改动 critic 数量或网络结构，需要事先定义权重迁移与初始化规则；只写“同一 checkpoint”不能自动保证这种架构变化下的初始化公平。

本次分数高于论文对应表格值；checkpoint 差异可能是原因之一，但尚未证明。这个不确定性不阻止当前版本作为固定初始化下的后续对照，也不构成超越论文算法的结论。
