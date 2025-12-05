# NeuroPred-MTCL
Multi-Task Contrastive Learning with Attention Mechanisms for Neuropeptide Prediction Using ESM Representations

## About

This system implements a neuropeptide prediction pipeline based on ESM pre-trained models, including three main steps: feature extraction, model training, and evaluation.

## File Description

### 1. `extract_esm_features.py` - ESM Feature Extraction
- **Function**: Extract features from protein sequences using ESM pre-trained models
- **Input**: Training and testing CSV files (containing sequences and labels)
- **Output**: ESM feature numpy array files
- **Features**: Automatically handles sequences of different lengths, supports batch processing

### 2. `main_train_optimized_v2.py` - Model Training
- **Function**: Train classification models based on ESM features
- **Features**: 
  - Adaptive ESM feature dimensions
  - Supports training with multiple random seeds
  - Automatically saves best models and training parameters
  - Supports command-line parameter configuration

### 3. `main_evaluation_optimized_v2.py` - Model Evaluation
- **Function**: Evaluate the performance of trained models
- **Features**:
  - Adaptive ESM feature dimensions
  - Computes multiple evaluation metrics (ACC, Precision, Recall, F1, MCC, AUROC)
  - Supports batch evaluation of multiple models
  - Generates detailed result reports

### 4. `model06_v2.py` - Model Architecture
- **Function**: Defines the neural network model structure
- **Features**:
  - Supports dynamic input dimensions
  - Includes BiLSTM encoder and self-attention mechanism
  - Supports contrastive learning and knowledge distillation

## Usage

### Step 1: Extract ESM Features

```bash
python extract_esm_features.py
```

This will read data from `data/training.csv` and `data/testing.csv`, and extract ESM features saved to the `features/` directory.

### Step 2: Train Model

```bash
# Train with default parameters
python main_train_optimized_v2.py

# Train with custom parameters
python main_train_optimized_v2.py --data_dir features \
                                   --output_dir checkpoints \
                                   --seeds 30,40 \
                                   --epochs 20
```

Parameter description:
- `--data_dir`: ESM feature data directory
- `--output_dir`: Model save directory
- `--seeds`: Random seed range, format as "start,end"
- `--epochs`: Number of training epochs

### Step 3: Evaluate Model

```bash
# Evaluate with default parameters
python main_evaluation_optimized_v2.py

# Evaluate with custom parameters
python main_evaluation_optimized_v2.py --data_dir features \
                                        --checkpoint_dir checkpoints \
                                        --seeds 30,40 \
                                        --model_name "Model06_v2_Attn" \
                                        --results_prefix "eval_m06"
```

Parameter description:
- `--data_dir`: ESM feature data directory
- `--checkpoint_dir`: Checkpoint directory
- `--seeds`: Random seed range
- `--model_name`: Model name
- `--results_prefix`: Result file prefix

## Output Files

### Feature Extraction Output
- `train_esm_seq.npy`: Training set ESM features
- `train_labels_seq.npy`: Training set labels
- `val_esm_seq.npy`: Validation set ESM features
- `val_labels_seq.npy`: Validation set labels
- `test_esm_seq.npy`: Testing set ESM features
- `test_labels_seq.npy`: Testing set labels

### Training Output
- `Model06_v2_Attn_seed{seed}_best.weights.h5`: Best model weights
- `Model06_v2_Attn_seed{seed}_best_params.json`: Training parameters and metadata

### Evaluation Output
- `eval_m06_seed{seed}_table.csv`: Single model result table
- `eval_m06_seed{seed}_table.md`: Single model result Markdown table
- `eval_m06_seed{seed}_results.json`: Single model detailed results
- `eval_m06_average_table.csv`: Average result table
- `eval_m06_average_table.md`: Average result Markdown table
- `eval_m06_summary.json`: Summary results

## Adaptive Dimension Features

The system has the following adaptive features:

1. **Feature Dimension Adaptation**: Automatically detects ESM feature dimensions without manual specification
2. **Class Number Adaptation**: Automatically detects the number of label classes
3. **Input Shape Adaptation**: Supports both 2D and 3D input formats
4. **Model Structure Adaptation**: Dynamically adjusts model structure based on input dimensions

## Requirements

- Python 3.7+
- TensorFlow 2.x
- PyTorch (for ESM models)
- NumPy
- Pandas
- scikit-learn
- tqdm

## Notes

1. Ensure sufficient GPU memory for ESM feature extraction
2. Best models are automatically saved during training
3. Evaluation results include detailed performance metrics and confusion matrices
4. Supports training with multiple random seeds for stable results
