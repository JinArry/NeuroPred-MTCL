# NeuroPred-MTCL
Multi-Task Contrastive Learning with Attention Mechanisms for Neuropeptide Prediction Using ESM Representations

## About

本系统实现了基于ESM预训练模型的神经肽预测流程，包括特征提取、模型训练和评估三个主要步骤。

## 文件说明

### 1. `extract_esm_features.py` - ESM特征提取
- **功能**: 使用ESM预训练模型从蛋白质序列中提取特征
- **输入**: 训练和测试CSV文件（包含序列和标签）
- **输出**: ESM特征numpy数组文件
- **特点**: 自动处理不同长度的序列，支持批处理

### 2. `main_train_optimized_v2.py` - 模型训练
- **功能**: 基于ESM特征训练分类模型
- **特点**: 
  - 自适应ESM特征维度
  - 支持多随机种子训练
  - 自动保存最佳模型和训练参数
  - 支持命令行参数配置

### 3. `main_evaluation_optimized_v2.py` - 模型评估
- **功能**: 评估训练好的模型性能
- **特点**:
  - 自适应ESM特征维度
  - 计算多种评估指标（ACC, Precision, Recall, F1, MCC, AUROC）
  - 支持批量评估多个模型
  - 生成详细的结果报告

### 4. `model06_v2.py` - 模型架构
- **功能**: 定义神经网络模型结构
- **特点**:
  - 支持动态输入维度
  - 包含BiLSTM编码器和自注意力机制
  - 支持对比学习和知识蒸馏

## 使用方法

### 步骤1: 提取ESM特征

```bash
python extract_esm_features.py
```

这将从 `data/training.csv` 和 `data/testing.csv` 读取数据，并提取ESM特征保存到 `features/` 目录。

### 步骤2: 训练模型

```bash
# 使用默认参数训练
python main_train_optimized_v2.py

# 自定义参数训练
python main_train_optimized_v2.py --data_dir features \
                                   --output_dir checkpoints \
                                   --seeds 30,40 \
                                   --epochs 20
```

参数说明：
- `--data_dir`: ESM特征数据目录
- `--output_dir`: 模型保存目录
- `--seeds`: 随机种子范围，格式为 "start,end"
- `--epochs`: 训练轮数

### 步骤3: 评估模型

```bash
# 使用默认参数评估
python main_evaluation_optimized_v2.py

# 自定义参数评估
python main_evaluation_optimized_v2.py --data_dir features \
                                        --checkpoint_dir checkpoints \
                                        --seeds 30,40 \
                                        --model_name "Model06_v2_Attn" \
                                        --results_prefix "eval_m06"
```

参数说明：
- `--data_dir`: ESM特征数据目录
- `--checkpoint_dir`: 检查点目录
- `--seeds`: 随机种子范围
- `--model_name`: 模型名称
- `--results_prefix`: 结果文件前缀

## 输出文件

### 特征提取输出
- `train_esm_seq.npy`: 训练集ESM特征
- `train_labels_seq.npy`: 训练集标签
- `val_esm_seq.npy`: 验证集ESM特征
- `val_labels_seq.npy`: 验证集标签
- `test_esm_seq.npy`: 测试集ESM特征
- `test_labels_seq.npy`: 测试集标签

### 训练输出
- `Model06_v2_Attn_seed{seed}_best.weights.h5`: 最佳模型权重
- `Model06_v2_Attn_seed{seed}_best_params.json`: 训练参数和元数据

### 评估输出
- `eval_m06_seed{seed}_table.csv`: 单个模型结果表格
- `eval_m06_seed{seed}_table.md`: 单个模型结果Markdown表格
- `eval_m06_seed{seed}_results.json`: 单个模型详细结果
- `eval_m06_average_table.csv`: 平均结果表格
- `eval_m06_average_table.md`: 平均结果Markdown表格
- `eval_m06_summary.json`: 汇总结果

## 自适应维度特性

系统具有以下自适应特性：

1. **特征维度自适应**: 自动检测ESM特征维度，无需手动指定
2. **类别数自适应**: 自动检测标签类别数
3. **输入形状自适应**: 支持2D和3D输入格式
4. **模型结构自适应**: 根据输入维度动态调整模型结构

## 依赖要求

- Python 3.7+
- TensorFlow 2.x
- PyTorch (用于ESM模型)
- NumPy
- Pandas
- scikit-learn
- tqdm

## 注意事项

1. 确保有足够的GPU内存用于ESM特征提取
2. 训练过程中会自动保存最佳模型
3. 评估结果包含详细的性能指标和混淆矩阵
4. 支持多随机种子训练以获得稳定的结果
