# 折内基础训练试跑

2026-09-05 已完成服务器试跑。脚本：`tools/benchmark_fold_training.py`。服务器结果目录：`/root/autodl-tmp/experiments/PARK_repro_01/results/fold_training_pilot_20260905/`。本地无标识汇总：`results/fold_training_pilot_20260905/benchmark_summary.json`。

这是为下一阶段提供工程与资源依据的单套训练试跑，使用现有 v3 的 outer fold 0 划分，不是已经冻结或完成的 v4 评估。

## 实现和验证

- 按参与者把 outer-train 划为优化训练 353 人、checkpoint 选择 76 人、校准 76 人；outer-validation 的 127 人只用于预测。四个集合互斥。
- StandardScaler 与 SMOTE 只使用优化训练数据。三个单模态网络和 UFNet 从头初始化，checkpoint 仅由 selection 决定；每个模型的逻辑回归校准器只使用 calibration 的参与者平均分数及标签。
- 递归检查 scaler → experts → fusion → calibration 的拟合依赖，确认预测参与者不出现在祖先中。合成图测试验证正确图通过、目标成员泄漏/未知祖先/循环依赖均被拒绝。这是声明的拟合依赖与调用路径检查，并非任意 Python 程序的自动数据流证明。
- 使用继承的 paired 训练函数和完整 epoch 配置、MC=30。未直接调用会预测锁定队列的 `train_one`。保存的 finger 配置为 97 epochs、UFNet 为 285 epochs；未传 epoch 覆盖参数。
- 私有成员清单、无标签预测表、模型与校准器保存在服务器。源码哈希与下载汇总匹配；受保护上游文件哈希未变化。未计算预测集疾病性能。
- 结果目录存在即拒绝运行，避免覆盖历史结果。

此实现只是基础模型阶段；尚未完成路由器层间 OOF、收益校准、场景隔离和 v3 风险重算问题的修复。最终无标签推理接口也仍需独立封装。恢复的原始特征与继承的结构/训练函数仍有上游来源局限，不应称为原始特征提取器的从零重训。

## 实测资源

GPU：RTX 2080 Ti，11 GB。

| 阶段 | 秒 |
| --- | ---: |
| Finger | 1.157 |
| Speech | 0.720 |
| Smile | 0.354 |
| UFNet | 4.526 |
| 预测与校准 | 0.066 |
| 训练、保存、预测与校准计时区间 | 6.839 |

计时在数据读取、上游模块导入、逆缩放和 scaler 拟合之后开始；不包含这些准备过程、SSH、进程启动及结束时的审计散列计算。训练阶段使用 CUDA 同步后读数。这里只测了一次、一个训练样本量，不是运行时间置信区间。

PyTorch 峰值 allocated 125.70 MiB，reserved 174 MiB；这不是整个 GPU 进程显存或主机 RAM 峰值。本次未测主机 RAM 峰值。汇总写入之前输出约 2.31 MiB。

按单次结果机械外推：105 套对应计时区间约 718 秒（11.97 分钟），315 套约 2154 秒（35.91 分钟）；旧试跑产物量级对应约 243/729 MiB。这些数值仅用于基础模型阶段排期，不包括下层分区差异、所有压力场景推理、检测器、收益模型、bootstrap 和 OOF 缓存开销。

全量开跑前仍需一个完整 outer fold 端到端试跑，记录总 wall time、全部拟合次数、RAM/GPU 峰值和缓存大小，再确定完整 105 套方案的预算。不能把约 12 分钟写成完整实验的承诺时间。

## 运行方法与下一项交付

在服务器 park 环境运行：

```bash
python tools/benchmark_fold_training.py --self-test
python tools/benchmark_fold_training.py \
  --fold 0 --seed 101 \
  --output-dir results/fold_training_pilot_NEW_RUN
```

下一项是将该基础训练单元接入分层 OOF 调度器：在每个策略验证折的训练侧独立生成下层 OOF、登记全部模型/选择/校准成员依赖，完成一个 outer fold 的端到端隔离测试。当前数据暴露状态仍为开发数据，不能用于无偏确认性结论。
