# -*- coding: utf-8 -*-
import os, json, gc, csv, time, random
import numpy as np
import tensorflow as tf
from model06_v2 import New_Model_06_Attn  # 按你的工程路径保持一致

# ========= 配置 =========
# 绝对路径基准目录（当前脚本所在目录）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURES_DIR = os.path.join(BASE_DIR, "features")
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# GPU设置（使用GPU 1）
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

TEST_X_PATH = os.path.join(FEATURES_DIR, "val_esm_seq.npy")
TEST_Y_PATH = os.path.join(FEATURES_DIR, "val_labels_seq.npy")
MODEL_NAME = "NewModel-05-Attn"  # 表格第一列显示的名称
RESULTS_PREFIX = "eval_m05"  # 输出文件前缀


# =======================

def set_global_determinism(seed=42):
    random.seed(seed);
    np.random.seed(seed);
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1';
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'


def setup_gpu():
    """设置GPU配置并显示当前使用的GPU"""
    # 检查环境变量
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "未设置")
    print(f"🔧 CUDA_VISIBLE_DEVICES: {cuda_visible}")
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ 检测到 {len(gpus)} 个GPU设备:")
        for i, gpu in enumerate(gpus):
            print(f"   GPU {i}: {gpu.name}")
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f"      ✅ 已启用内存增长")
            except RuntimeError as e:
                print(f"      ⚠️  内存增长设置失败: {e}")
        
        # 显示TensorFlow实际使用的设备
        try:
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(f"📊 TensorFlow逻辑GPU设备: {len(logical_gpus)} 个")
            for i, lgpu in enumerate(logical_gpus):
                print(f"   逻辑GPU {i}: {lgpu.name}")
        except:
            pass
    else:
        print("⚠️  未检测到GPU设备，将使用CPU")


# ---------- metrics（纯 NumPy） ----------
def confusion_matrix_np(y_true, y_pred, C=None):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    if C is None: C = int(max(y_true.max(), y_pred.max()) + 1)
    cm = np.zeros((C, C), dtype=np.int64)
    for t, p in zip(y_true, y_pred): cm[t, p] += 1
    return cm


def binary_from_cm(cm):
    # 假设标签为 {0,1}，正类=1
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    return tp, tn, fp, fn


def acc_from_cm(cm):
    return np.diag(cm).sum() / max(cm.sum(), 1)


def precision(tp, fp, eps=1e-12):
    return tp / (tp + fp + eps)


def recall(tp, fn, eps=1e-12):  # Sensitivity
    return tp / (tp + fn + eps)


def f1_score_bin(tp, fp, fn, eps=1e-12):
    pre = precision(tp, fp, eps);
    rec = recall(tp, fn, eps)
    return 2 * pre * rec / (pre + rec + eps)


def mcc_bin(tp, tn, fp, fn, eps=1e-12):
    num = tp * tn - fp * fn
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + eps
    return num / den


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x);
    return e / np.sum(e, axis=axis, keepdims=True)


def auroc_binary(y_true, y_score):
    """等价 sklearn 的 ROC AUC（二分类），不依赖第三方"""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    # 排序（降序）
    order = np.argsort(y_score, kind="mergesort")[::-1]
    y_true_sorted = y_true[order]
    score_sorted = y_score[order]
    # 找到分数变化的断点索引
    distinct = np.where(np.diff(score_sorted))[0]
    thr_idx = np.r_[distinct, y_true_sorted.size - 1]
    tps = np.cumsum(y_true_sorted)[thr_idx]
    fps = 1 + thr_idx - tps
    P = y_true.sum();
    N = y_true.size - P
    if P == 0 or N == 0:
        return 0.0  # 退化情形
    fpr = fps / N;
    tpr = tps / P
    # 加端点
    fpr = np.r_[0.0, fpr, 1.0];
    tpr = np.r_[0.0, tpr, 1.0]
    return float(np.trapz(tpr, fpr))


# ---------- 构建与加载 ----------
def build_model_from_params(p: dict, input_shape, num_classes: int):
    kwargs = dict(
        hidden_dim=p["hidden_dim"],
        projection_dim=p["projection_dim"],
        num_heads=p["num_heads"],
        alpha=p["alpha"],
        beta=p["beta"],
    )
    try:
        model = New_Model_06_Attn(**kwargs)
    except TypeError:
        kwargs["num_classes"] = int(num_classes)
        model = New_Model_06_Attn(**kwargs)

    if hasattr(model, "temperature") and "temperature" in p:
        model.temperature = p["temperature"]

    lr = p.get("learning_rate", 1e-3)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def ensure_built_then_load(model, x_like: np.ndarray, ckpt: str):
    dummy = tf.convert_to_tensor(x_like[:1])
    try:
        _ = model(dummy, training=False)
    except Exception:
        L, D = x_like.shape[1], x_like.shape[2]
        try:
            model.build(input_shape=(None, L, D))
        except Exception:
            pass
    model.load_weights(ckpt)


# ---------- 打印/保存表格 ----------
def print_table_row(model_name, acc, pre, auroc, mcc, f1, sn, seed=None):
    if seed is not None:
        model_name_with_seed = f"{model_name}_seed{seed}"
    else:
        model_name_with_seed = model_name
    
    head = f"{'Model':<25} {'ACC':>6} {'Pre':>6} {'AUROC':>7} {'MCC':>6} {'F1':>6} {'SN':>6}"
    row = f"{model_name_with_seed:<25} {acc:>6.3f} {pre:>6.3f} {auroc:>7.3f} {mcc:>6.3f} {f1:>6.3f} {sn:>6.3f}"
    line = "-" * len(head)
    
    if seed is None:  # 汇总表格
        print(f"\n📊 Summary Results (Average ± Std)\n")
    else:  # 单个种子
        print(f"\n🎯 Seed {seed} Results\n")
    
    print(head)
    print(row)
    print(line)


def save_csv_md(model_name, acc, pre, auroc, mcc, f1, sn, prefix, seed=None):
    # 确保results目录存在
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    if seed is not None:
        filename_prefix = f"{prefix}_seed{seed}"
        model_name_with_seed = f"{model_name}_seed{seed}"
    else:
        filename_prefix = f"{prefix}_summary"
        model_name_with_seed = f"{model_name}_avg"
    
    # CSV
    csv_path = os.path.join(RESULTS_DIR, f"{filename_prefix}_table.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Model", "ACC", "Pre", "AUROC", "MCC", "F1", "SN"])
        w.writerow([model_name_with_seed, f"{acc:.3f}", f"{pre:.3f}", f"{auroc:.3f}", f"{mcc:.3f}", f"{f1:.3f}", f"{sn:.3f}"])
    
    # Markdown
    md = (
        "| Model | ACC | Pre | AUROC | MCC | F1 | SN |\n"
        "|:--|--:|--:|--:|--:|--:|--:|\n"
        f"| {model_name_with_seed} | {acc:.3f} | {pre:.3f} | {auroc:.3f} | {mcc:.3f} | {f1:.3f} | {sn:.3f} |\n"
    )
    md_path = os.path.join(RESULTS_DIR, f"{filename_prefix}_table.md")
    with open(md_path, "w", encoding="utf-8") as f: 
        f.write(md)


def save_summary_results(all_results, prefix):
    """保存汇总结果"""
    # 确保results目录存在
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # 计算统计量
    metrics = ['acc', 'precision', 'recall', 'f1', 'mcc', 'auroc']
    summary = {}
    
    for metric in metrics:
        values = [r[metric] for r in all_results]
        summary[f"{metric}_mean"] = float(np.mean(values))
        summary[f"{metric}_std"] = float(np.std(values))
        summary[f"{metric}_min"] = float(np.min(values))
        summary[f"{metric}_max"] = float(np.max(values))
    
    # 保存详细汇总
    summary_data = {
        "model_name": MODEL_NAME,
        "num_seeds": len(all_results),
        "seeds": [r["seed"] for r in all_results],
        "summary_statistics": summary,
        "individual_results": all_results
    }
    
    summary_json_path = os.path.join(RESULTS_DIR, f"{prefix}_summary.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)
    
    # 打印汇总表格
    print_table_row(
        MODEL_NAME, 
        summary["acc_mean"], summary["precision_mean"], 
        summary["auroc_mean"], summary["mcc_mean"], 
        summary["f1_mean"], summary["recall_mean"]
    )
    
    # 保存汇总CSV/MD
    save_csv_md(
        MODEL_NAME, 
        summary["acc_mean"], summary["precision_mean"], 
        summary["auroc_mean"], summary["mcc_mean"], 
        summary["f1_mean"], summary["recall_mean"], 
        prefix
    )
    
    # 打印统计信息
    print(f"\n📈 Statistics Summary:")
    print(f"   Seeds: {len(all_results)}")
    print(f"   Best ACC: {summary['acc_max']:.3f} (seed {all_results[np.argmax([r['acc'] for r in all_results])]['seed']})")
    print(f"   Best F1:  {summary['f1_max']:.3f} (seed {all_results[np.argmax([r['f1'] for r in all_results])]['seed']})")
    print(f"   Best AUROC: {summary['auroc_max']:.3f} (seed {all_results[np.argmax([r['auroc'] for r in all_results])]['seed']})")


# ---------- 主流程 ----------
def main(SEED):
    CKPT_PATH = os.path.join(CHECKPOINTS_DIR, f"Model06_v2_Attn_seed{SEED}_best.weights.h5")
    set_global_determinism(SEED);
    setup_gpu()
    assert CKPT_PATH.endswith(".weights.h5")
    params_path = CKPT_PATH.replace(".weights.h5", "_params.json")
    assert os.path.exists(params_path), f"未找到超参文件：{params_path}"

    # 读数据
    x_test = np.load(TEST_X_PATH)  # [N, L, D]
    y_test = np.load(TEST_Y_PATH)  # [N]，标签应为 0/1（二分类）
    assert x_test.shape[0] == y_test.shape[0]
    num_classes = int(np.max(y_test) + 1)

    # 读超参并建模
    with open(params_path, "r", encoding="utf-8") as f:
        p = json.load(f)["params"]
    model = build_model_from_params(p, x_test.shape, num_classes)
    ensure_built_then_load(model, x_test, CKPT_PATH)

    # 推理（批处理）
    batch = int(p.get("batch_size", 128))
    ds = tf.data.Dataset.from_tensor_slices(x_test).batch(batch).prefetch(tf.data.AUTOTUNE)
    t0 = time.time();
    y_prob = model.predict(ds, verbose=1);
    t1 = time.time()
    print(f"⏱️ {t1 - t0:.2f}s, {((t1 - t0) * 1000 / x_test.shape[0]):.2f} ms/样本")

    # 统一到 (N,C) 概率
    if y_prob.ndim == 1: y_prob = y_prob.reshape(-1, 1)
    if y_prob.shape[1] > 1:
        y_prob = softmax(y_prob, axis=1)
        y_pred = np.argmax(y_prob, axis=1)
        pos_score = y_prob[:, 1] if y_prob.shape[1] >= 2 else y_prob[:, 0]  # 二分类正类分数
    else:
        pos_score = y_prob.ravel()
        y_pred = (pos_score >= 0.5).astype(int)

    # 指标（按二分类定义）
    cm = confusion_matrix_np(y_test, y_pred, C=2)
    tp, tn, fp, fn = binary_from_cm(cm)
    acc = acc_from_cm(cm)
    pre = precision(tp, fp)
    sn = recall(tp, fn)  # Sensitivity/Recall
    f1 = f1_score_bin(tp, fp, fn)
    mcc = mcc_bin(tp, tn, fp, fn)
    try:
        auroc = auroc_binary(y_test, pos_score)
    except Exception:
        auroc = 0.0

    # 打印 & 保存表
    print_table_row(MODEL_NAME, acc, pre, auroc, mcc, f1, sn, seed=SEED)
    save_csv_md(MODEL_NAME, acc, pre, auroc, mcc, f1, sn, RESULTS_PREFIX, seed=SEED)

    # 另存一份原始结果 JSON
    result_data = {
        "seed": SEED,
        "ckpt": CKPT_PATH, 
        "params_json": params_path,
        "acc": float(acc), 
        "precision": float(pre), 
        "recall": float(sn),  # 统一字段名
        "f1": float(f1), 
        "mcc": float(mcc), 
        "auroc": float(auroc),
        "confusion_matrix": [[int(cm[0, 0]), int(cm[0, 1])],
                            [int(cm[1, 0]), int(cm[1, 1])]]
    }
    
    # 确保results目录存在
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    result_json_path = os.path.join(RESULTS_DIR, f"{RESULTS_PREFIX}_seed{SEED}_results.json")
    with open(result_json_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)

    # 清理
    del model, x_test, y_test, y_prob, y_pred
    tf.keras.backend.clear_session()
    gc.collect()
    
    # 返回结果用于汇总
    return result_data


if __name__ == "__main__":
    print("🚀 Starting evaluation for seeds 30-39...")
    print("=" * 60)
    
    all_results = []
    
    # for seed in range(30, 40):
    for seed in range(37, 38):
        print(f"\n🔄 Processing seed {seed}...")
        try:
            result = main(seed)
            all_results.append(result)
            print(f"✅ Seed {seed} completed successfully")
        except Exception as e:
            print(f"❌ Seed {seed} failed: {e}")
            continue
    
    # 生成汇总结果
    if all_results:
        print("\n" + "=" * 60)
        print("📊 Generating summary results...")
        save_summary_results(all_results, RESULTS_PREFIX)
        print(f"\n🎉 Evaluation completed! Processed {len(all_results)} seeds successfully.")
        print(f"📁 All results saved to: {RESULTS_DIR}")
        print(f"   - Individual results: {len(all_results)} files")
        print(f"   - Summary files: 3 files (CSV, MD, JSON)")
    else:
        print("\n❌ No results to summarize.")
