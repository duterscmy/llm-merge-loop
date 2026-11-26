# CoMe merge + loop：完整实验结果

更新时间：2026-10-23

## 1. 统一评估口径

- 主模型：Llama-2-7B，32 层；CoMe 按论文配置压缩为 29 个物理层，参数量由 6.738B 降至 6.131B（-9.01%）。
- CoMe 合并组（0-based）：`[23,24]`、`[25,26]`、`[27,28]`。论文对齐的 merge 不改；本项目只额外增加 loop。
- 外部 PPL：WikiText-2 test（166 个 2048-token block）与 C4 validation（固定 seed=10，256 个 2048-token block）。数值越低越好。
- Commonsense：ARC-Challenge、ARC-Easy、HellaSwag、OpenBookQA、PIQA 使用 length-normalized accuracy，WinoGrande 使用 accuracy；六项简单平均。全部 zero-shot，数值为百分比，越高越好。
- MMLU 不属于本轮主评估，Llama-2 表中不包含 MMLU。
- `29/29` 表示 29 个物理层、29 次逻辑执行；`29/32` 表示 29 个物理层通过参数共享 loop 执行 32 次。

## 2. Llama-2-7B：PPL 与 Commonsense 总表

| 方法 | 物理/逻辑层 | 蒸馏数据量 | Wiki2 PPL ↓ | C4 PPL ↓ | Commonsense Avg ↑ |
|---|---:|---:|---:|---:|---:|
| Dense baseline | 32/32 | — | **5.4683** | **7.2667** | **64.8312** |
| CoMe merged（论文配置，无 loop） | 29/29 | — | 6.2328 | 8.5494 | 62.5668 |
| CoMe merge-aware loop（未蒸馏） | 29/32 | — | 7.0656 | 9.6489 | 61.8536 |
| Uniform loop control | 29/32 | — | 7.0789 | 9.6452 | 58.4371 |
| Random loop control | 29/32 | — | 7.0537 | 9.7201 | 61.2743 |
| Endpoint hidden MSE 1k（merged 推理） | 29/29 | 0.512M tokens | 6.2439 | 8.5264 | 62.5465 |
| Endpoint hidden MSE 1k（loop 推理） | 29/32 | 0.512M tokens | 6.8634 | 9.3533 | 62.1483 |
| 主方案 10k：forward KL + group-endpoint normalized hidden MSE（merged） | 29/29 | 5.12M tokens | 6.2775 | 8.6046 | 62.5662 |
| 主方案 10k：forward KL + group-endpoint normalized hidden MSE（loop） | 29/32 | 5.12M tokens | 6.5150 | 8.7577 | 62.3763 |
| 激进方案 10k：forward KL + per-iteration normalized hidden MSE（merged） | 29/29 | 5.12M tokens | 6.2793 | 8.6086 | 62.5123 |
| 激进方案 10k：forward KL + per-iteration normalized hidden MSE（loop） | 29/32 | 5.12M tokens | 6.5239 | 8.7643 | 62.4665 |
| 主方案长训 checkpoint（10k 起点 + 长训 step 201,400，loop） | 29/32 | 108.24M cumulative tokens | **6.4826** | **8.7092** | **62.5231** |

加粗只表示同类中的参照或当前最佳，不表示统计显著性。

## 3. Llama-2-7B：六个 Commonsense 任务明细

| 方法 | ARC-C | ARC-E | HellaSwag | OBQA | PIQA | WinoGrande | Avg |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense baseline | 45.73 | 74.03 | 76.13 | 44.40 | 79.00 | 69.69 | 64.83 |
| CoMe merged | 43.34 | 70.33 | 74.23 | 42.00 | 77.15 | 68.35 | 62.57 |
| Untrained merge-aware loop | 42.49 | 69.61 | 74.07 | 41.40 | 76.22 | 67.32 | 61.85 |
| Uniform loop | 36.95 | 60.48 | 70.89 | 40.40 | 75.84 | 66.06 | 58.44 |
| Random loop | 41.04 | 68.35 | 73.48 | 41.20 | 76.17 | 67.40 | 61.27 |
| Endpoint MSE 1k merged | 43.86 | 70.33 | 74.14 | 41.60 | 76.77 | 68.59 | 62.55 |
| Endpoint MSE 1k loop | 43.00 | 69.65 | 73.98 | 41.80 | 76.50 | 67.96 | 62.15 |
| 主方案 10k merged | 43.52 | 70.12 | 73.90 | 42.00 | 77.04 | 68.82 | 62.57 |
| 主方案 10k loop | 42.83 | 70.41 | 74.06 | 41.80 | 77.04 | 68.11 | 62.38 |
| 激进方案 10k merged | 43.69 | 70.08 | 73.83 | 41.80 | 77.09 | 68.59 | 62.51 |
| 激进方案 10k loop | 43.00 | 70.20 | 74.09 | 42.00 | 77.31 | 68.19 | 62.47 |
| 主方案长训 step 201,400 loop | 43.17 | 70.41 | 74.23 | 42.20 | 76.93 | 68.19 | 62.52 |

## 4. 两种正式蒸馏方案

两种方法都固定：C4、sequence length 512、10k steps、5.12M tokens、LR 1e-5、500-step warmup、cosine schedule、temperature 1、forward KL 为主、normalized hidden MSE 为辅、动态等量平衡；只训练三个合并层，共 607.15M trainable parameters。

1. **组末端（主方案）**：每个共享层执行两次后，只让最终输出拟合 dense 组末端，即 `a*2 → dense layer 24`。
2. **逐次/激进方案**：共享层第一次执行拟合 dense 前一层、第二次执行拟合 dense 后一层，即 `a → layer 23`，`a*2 → layer 24`；另外两组合并层同理。

在相同 10k 预算下，主方案 loop 的 Wiki2/C4 PPL 为 6.5150/8.7577，激进方案为 6.5239/8.7643。差距只有 0.0089/0.0066；激进方案 Commonsense 高 0.0902 个点。这个量级不足以支持“激进方案更好”，PPL 上主方案略优。

## 5. 收敛判断

- 公平的 10k 收敛曲线使用同一个固定 C4 dev split（64 × 512 tokens），每 1,000 steps 评估一次真实 next-token PPL。该曲线仅用于训练过程横向比较；因为 block 长度和采样集不同，不能与第 2 节的外部 C4 PPL 直接比较。
- 长训曲线也使用固定 64 × 512-token dev，但其缓存样本与本次 10k 公平对照不是同一批 token；因此长训曲线只用于判断自身平台期，不能与 10k 曲线的绝对 PPL 首尾拼接。
- 主方案固定 dev PPL 从 10.7735 降至 9.9273（-0.8462，-7.85%）；激进方案从 10.7735 降至 9.9431（-0.8304，-7.71%）。主方案最终低 0.0158。
- 主方案最后 3k steps 只再下降 0.00118；激进方案在 step 7,000 达到最低 9.9421，最终反而高 0.00102。两条曲线都已进入明确平台期；在当前 cosine LR 于 10k 降至 0 的训练预算下，可以视为已收敛。这个结论不等价于“换新 schedule/重启学习率也不可能再改善”。
- 主方案长训从已完成的 10k checkpoint 继续。固定 dev PPL 从 9.3933（长训 step 0）下降到 9.3434（step 241,700，日志中的最低观测），step 261,700 回升到 9.3462。
- 长训前 40k steps 下降 0.0375；40k 到 120.9k 只再下降 0.0079；120.9k 到最低点只再下降 0.0046。后半段变化已接近 dev 噪声，说明主方案基本进入平台期，但不是数学意义上的严格收敛。
- 已完成外部 benchmark 的冻结 checkpoint 是长训 step 201,400（dev PPL 9.3452），不是日志最低的 step 241,700。两者必须区分。

## 6. 主要结论

1. **merge 本身是最强压缩基线**：CoMe merged 为 6.2328/8.5494/62.5668；直接加未训练 loop 会退化到 7.0656/9.6489/61.8536。
2. **蒸馏确实让 loop 恢复**：主方案长训相对未蒸馏 loop，Wiki2 PPL 降 0.5831，C4 PPL 降 0.9397，Commonsense 提升 0.6695 点。
3. **但 loop 尚未超过无 loop 的 CoMe merged**：当前最佳已评估 loop 相比 CoMe merged，Wiki2/C4 PPL 仍高 0.2498/0.1598；Commonsense 低 0.0437 点，几乎持平。
4. **延长主方案训练的边际收益很小**：相对主方案 10k，长训 checkpoint 只改善 Wiki2 0.0324、C4 0.0485，Commonsense 提升 0.1468 点。
5. **激进逐次蒸馏没有显示出稳定优势**：10k 外部 PPL 略差，Commonsense 的小幅提升低于单次评估应谨慎解读的范围。
6. 因此当前最稳妥的论文表述是：**merge-aware loop 通过蒸馏恢复了大部分因重复共享层造成的退化，并达到接近 merged baseline 的水平；尚无证据表明它优于原始 merged 模型。**

## 7. Mistral-7B-v0.1 CoMe 复现（补充）

该组实验是前期 CoMe 论文复现，用于确认 merge 与 loop 基础实现；C4 PPL 当时未评估。

| 方法 | 物理/逻辑层 | Wiki2 PPL ↓ | Commonsense Avg ↑ |
|---|---:|---:|---:|
| Dense | 32/32 | 5.2521 | 68.9782 |
| CoMe merged | 29/29 | 6.5997 | 66.0840 |
| Merge-aware loop | 29/32 | 7.9125 | 65.0494 |
| Uniform loop | 29/32 | 7.4008 | 64.3889 |
| Random loop | 29/32 | 8.1200 | 64.6142 |

## 8. 结果与模型位置

- Llama-2 原始 CoMe 与 loop/control：`results/come-llama2-9pct-paper/`
- Endpoint-only 1k：`results/come-llama2-endpoint-kd-c4-1k/`
- 主方案 10k：`results/come-llama2-kl-endpoint-c4-10k/`
- 激进方案 10k：`results/come-llama2-aggressive-kl-hidden-c4-10k/`
- 主方案长训已评估 checkpoint：`results/come-llama2-kl-endpoint-c4-24h-best-step-201400/`
- 两种方法的严格 10k 收敛曲线数据：`results/distillation-convergence/`
- 主方案 10k 权重与长训训练状态已从旧服务器清理。上面的结果和收敛数据仍保留。
