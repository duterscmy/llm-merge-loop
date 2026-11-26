# CoMe + Loop experiment handoff

研究问题：用 CoMe 合并相邻层后，让合并层共享权重并重复执行，再用未压缩的 dense 模型做恢复性蒸馏。CoMe 原论文是 Wang et al., *Layer as Puzzle Pieces*, NeurIPS 2025。[论文](https://arxiv.org/abs/2510.15304) · [CoMe 上游](https://github.com/WangFei-2019/CoMe)

## 已完成的实验

| Llama-2-7B 方法 | WikiText-2 PPL ↓ | C4 PPL ↓ | 6 项 Commonsense 平均 ↑ |
|---|---:|---:|---:|
| Dense | 5.4683 | 7.2667 | 64.8312 |
| CoMe 32→29 | 6.2328 | 8.5494 | 62.5668 |
| CoMe + Loop，未蒸馏 | 7.0656 | 9.6489 | 61.8538 |
| 组末端蒸馏 + Loop，10k | 6.5150 | 8.7577 | 62.3763 |
| 逐次蒸馏 + Loop，10k | 6.5239 | 8.7643 | 62.4665 |
| 组末端蒸馏 + Loop，长训已评估 checkpoint | 6.4826 | 8.7092 | 62.5231 |

蒸馏把 Loop 表现拉回到接近 CoMe merged。已评估的 Loop checkpoint 在两项 PPL 上仍比 CoMe merged 高，长训相对 10k 的增益较小。10k 公平对照的末段曲线都趋平，组末端方案略优。长训外部评估使用 step 201,400，训练日志中最低 dev PPL 出现在 step 241,700。

- [完整历史结果](results/COMPLETE_EXPERIMENT_RESULTS.md)，含每项 Commonsense、随机/均匀 Loop、Mistral 和收敛分析。
- [组末端曲线数据](results/distillation-convergence/endpoint-10k.json) · [逐次曲线数据](results/distillation-convergence/aggressive-10k.json)。
- [Qwen 迁移 TODO](TODO.md)。

Arrhenius 的 Qwen 作业 `5782869` 已取消。Qwen3-8B-Base 下载、36→35 单次试合并和 Dense 前向均已完成。取消时正在重载 merged checkpoint，Loop 前向、36→33 合并及恢复训练均待验证。单次试合并的映射在 `results/qwen3-paused/`。

仓库包含代码、脚本、结果、历史配置和带本地补丁的 CoMe 源码。上游基点为 `7bb695adba4b69df45020e281348e99739824c39`。`archive/arrhenius/` 中的 Slurm 脚本与说明保留原站点配置，供审计复现。模型权重、训练状态、数据缓存和集群日志不在交接包内。Llama-2 的历史蒸馏 checkpoint 之前已清理，重跑时需要从 dense 和 CoMe 合并重新开始。历史 JSON 含原机器路径，公开前请审阅。

## 在新服务器接续

先准备 Python 3.11 和适配新机的 CUDA PyTorch。Qwen 路径在旧服务器使用 PyTorch 2.6.0、Transformers 4.51.3、Tokenizers 0.21.4。GPU 相关安装及运行在新服务器的 GPU 环境中执行。

```bash
python -m pip install -r requirements-qwen3.txt
bash scripts/qwen3_smoke.sh
bash scripts/qwen3_pilot.sh prepare
bash scripts/qwen3_pilot.sh loop-c4
bash scripts/qwen3_pilot.sh loop-mixed
bash scripts/qwen3_pilot.sh merge-mixed
```

上面是顺序执行示例。用 Slurm 时，把每条命令放入有 GPU allocation 的脚本，并使用新站点的 account、partition 和环境激活方式。三个训练组都依赖 `prepare` 成功，之后可以各用一张 GPU 独立跑。恢复同一组时重新运行同名模式，保留该组 `state/last.pt` 及原始配置。
