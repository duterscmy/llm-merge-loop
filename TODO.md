# Qwen3 恢复实验 TODO

- [ ] 在新服务器跑 `qwen3_smoke.sh`：检查 36→35 合并权重重载、WikiText-2 两个 block 的 Dense、Merge、Merge + Loop PPL，并确认数值有限。
- [ ] 用 WikiText-2 train 重新校准 CoMe，生成 36→33 checkpoint、重要性以及真正的 Qwen 合并组。沿用 BI、256 条校准样本、每次合并两层与 `wo-repeat` 配置。之前 36→35 的合并组仅来自 8 条校准样本。
- [ ] 按 Qwen tokenizer 的 token 数准备配对语料。混合组每 5 个 block 替换 1 个为 Wiki-Rewrite，其余与 C4 对照共用。记录 Nemotron shard、revision、子集许可和过滤统计。脚本对 WikiText-2 dev/test 检查完全匹配的归一化 12 词片段，改写级重叠仍需进一步抽查。
- [ ] 跑 5 步混合输入蒸馏 smoke，确认教师和学生前向、KL、hidden loss、反传和显存。保留 dense 及两种未训练压缩基线的 WikiText-2 test PPL。
- [ ] 跑三组单卡训练：Loop + C4、Loop + C4/Wiki-Rewrite、Merge + C4/Wiki-Rewrite。固定 256 tokens/step，最多 2,000 步（512,000 tokens/组），只更新合并块。损失为 forward KL 加组末端 normalized hidden MSE。
- [ ] 在固定 WikiText-2 validation 上每 200 步评估。连续 3 次没有累计至少 0.1% 的相对改善则早停，保存 best/last 状态。如果上限处仍持续改善，先看曲线再决定是否增加预算。早停可能使各组最终消耗不同，画曲线时按 token 对齐比较共同预算。
- [ ] 对各组 best 跑完整 WikiText-2 test，汇总六个变体、dev 曲线、实际 tokens 和停因。只用 dev 选 checkpoint。这轮只做 WikiText-2 PPL。

训练输出在 `checkpoints/qwen3-come-pilot/<mode>/state/`。可打包的历史与新结果在 `results/`。报告跨模型比较时记录各自相对 dense 的变化，避免把不同 tokenizer 的绝对 PPL 当作直接对照。
