import os
import random
import numpy as np
import tensorflow as tf
import json
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from model06_v2 import New_Model_06_Attn

# 绝对路径基准目录（当前脚本所在目录）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURES_DIR = os.path.join(BASE_DIR, "features")
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 切换到GPU 0，避免内存冲突


def set_global_determinism(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    print(f"随机种子已设置为：{seed}")


x_train = np.load(os.path.join(FEATURES_DIR, "train_esm_seq.npy"))
y_train = np.load(os.path.join(FEATURES_DIR, "train_labels_seq.npy"))
x_val = np.load(os.path.join(FEATURES_DIR, "val_esm_seq.npy"))
y_val = np.load(os.path.join(FEATURES_DIR, "val_labels_seq.npy"))

# 独热编码
num_classes = len(np.unique(y_train))
y_train_cat = to_categorical(y_train, num_classes)
y_val_cat = to_categorical(y_val, num_classes)

params = {
    
    
    # random search
    "hidden_dim": 160,
    "projection_dim": 96,
    "alpha": 0.6, # 知识蒸馏权重0.6
    "beta": 0.03, # 对比学习权重0.03
    "temperature": 0.1,  # 模型里要接收并使用
    "learning_rate": 0.001,
    "batch_size": 64,
    "num_heads": 8,
    "dropout_rate": 0.0,  # 完全移除dropout，回到原始状态（原始模型无dropout时性能最好0.936）
    # "label_smoothing": 0.05,  # 标签平滑系数（经过测试：0.05最优ACC=0.928，0.1=0.921，0.2=0.914）
    # "grad_clip_norm": 5.0  # 梯度裁剪（经过测试：5.0最优ACC=0.931，4.0=0.929，1.0=0.927，2.0=0.926，3.0=0.917）
}

# for seed in range(30, 40):
for seed in range(37, 38):
    # 固定随机种子
    set_global_determinism(seed)

    batch = params["batch_size"]
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train_cat))
    train_ds = train_ds.shuffle(1024).batch(batch).prefetch(tf.data.AUTOTUNE)
    #train_ds = train_ds.shuffle(1024, seed=seed, reshuffle_each_iteration=True).batch(batch).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val_cat))
    val_ds = val_ds.batch(batch).prefetch(tf.data.AUTOTUNE)

    model = New_Model_06_Attn(
        hidden_dim=params["hidden_dim"],
        projection_dim=params["projection_dim"],
        num_heads=params["num_heads"],
        alpha=params["alpha"],
        beta=params["beta"],
        dropout_rate=params.get("dropout_rate", 0.1),  # 添加dropout_rate参数
        label_smoothing=params.get("label_smoothing", 0.0),  # 标签平滑系数（已禁用，设为0.0）
        grad_clip_norm=params.get("grad_clip_norm", None),  # 梯度裁剪（已禁用，设为None）
    )

    optimizer = Adam(learning_rate=params["learning_rate"])
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    ckpt_h5 = os.path.join(
        CHECKPOINTS_DIR,
        f"Model06_v2_Attn_seed{seed}_best.weights.h5"
    )
    checkpoint_cb = ModelCheckpoint(
        filepath=ckpt_h5,
        monitor='val_accuracy',  # 盯验证集准确率
        mode='max',
        save_best_only=True,  # 只在出现更优成绩时保存（覆盖）
        save_weights_only=True,  # 保存为 .h5 权重，最稳
        verbose=1
    )
    
    # 早停机制 - 防止过拟合
    early_stop_cb = EarlyStopping(
        monitor='val_accuracy',
        patience=8,  # 8个epoch没有提升就停止，从8->5
        restore_best_weights=True,  # 恢复最佳权重
        verbose=1
    )
    
    # 学习率调度 - 自动调整学习率
    reduce_lr_cb = ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.5,  # 学习率减半
        patience=3,  # 3个epoch没有提升就降低学习率
        min_lr=1e-6,  # 最小学习率
        verbose=1
    )

    # 使用 dataset 进行训练
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=50,  # 设置较大值，让早停决定何时停止
        callbacks=[checkpoint_cb, early_stop_cb, reduce_lr_cb]
    )
    val_acc_hist = history.history.get('val_accuracy', [])
    train_acc_hist = history.history.get('accuracy', [])
    train_loss_hist = history.history.get('loss', [])
    val_loss_hist = history.history.get('val_loss', [])
    
    if len(val_acc_hist) > 0:
        best_epoch = int(np.argmax(val_acc_hist) + 1)  # 1-based
        best_val = float(np.max(val_acc_hist))
        final_val = float(val_acc_hist[-1])
        
        # ========== 训练诊断报告 ==========
        print(f"\n{'='*60}")
        print(f"📊 Seed {seed} 训练诊断报告")
        print(f"{'='*60}")
        print(f"总训练轮数: {len(val_acc_hist)}")
        print(f"最佳epoch: {best_epoch}")
        print(f"最佳验证准确率: {best_val:.4f}")
        print(f"最后验证准确率: {final_val:.4f}")
        print(f"性能变化: {final_val - best_val:+.4f}")
        
        # 检查是否提前收敛
        if best_epoch < len(val_acc_hist) * 0.7:
            print(f"\n⚠️  模型在{best_epoch}个epoch就收敛了（占总轮数的{best_epoch/len(val_acc_hist)*100:.1f}%）")
            print(f"   建议: 添加早停机制，设置epochs=50+，让早停决定何时停止")
        elif best_epoch == len(val_acc_hist):
            print(f"\n✅ 模型在最后epoch达到最佳，可能需要更多epochs或学习率调度")
        else:
            print(f"\n⚠️  模型在{best_epoch}个epoch后性能下降，可能过拟合")
            print(f"   建议: 添加早停机制，恢复最佳权重")
        
        # 检查过拟合情况
        if len(train_acc_hist) > 0:
            best_train_acc = float(train_acc_hist[best_epoch - 1])
            final_train_acc = float(train_acc_hist[-1])
            best_gap = best_train_acc - best_val
            final_gap = final_train_acc - final_val
            
            print(f"\n📈 过拟合分析:")
            print(f"   最佳epoch训练准确率: {best_train_acc:.4f}")
            print(f"   最佳epoch验证准确率: {best_val:.4f}")
            print(f"   最佳epoch时训练-验证差距: {best_gap:.4f}")
            print(f"   最后epoch训练准确率: {final_train_acc:.4f}")
            print(f"   最后epoch验证准确率: {final_val:.4f}")
            print(f"   最后epoch时训练-验证差距: {final_gap:.4f}")
            
            if final_gap > best_gap + 0.05:  # 差距增加超过5%
                print(f"\n⚠️  存在过拟合趋势（差距从{best_gap:.4f}增加到{final_gap:.4f}）")
                print(f"   建议: 添加Dropout正则化防止过拟合")
            elif final_gap > 0.1:
                print(f"\n⚠️  训练-验证差距较大: {final_gap:.4f}")
                print(f"   建议: 考虑添加正则化")
            else:
                print(f"\n✅ 训练-验证差距合理，未发现明显过拟合")
        
        # Loss变化分析
        if len(train_loss_hist) > 0 and len(val_loss_hist) > 0:
            initial_train_loss = float(train_loss_hist[0])
            final_train_loss = float(train_loss_hist[-1])
            best_val_loss = float(min(val_loss_hist))
            best_val_loss_epoch = int(np.argmin(val_loss_hist) + 1)
            final_val_loss = float(val_loss_hist[-1])
            train_loss_reduction = ((initial_train_loss - final_train_loss) / initial_train_loss) * 100
            
            print(f"\n📉 Loss变化分析:")
            print(f"   初始训练Loss: {initial_train_loss:.4f}")
            print(f"   最终训练Loss: {final_train_loss:.4f}")
            print(f"   训练Loss下降: {train_loss_reduction:.2f}%")
            print(f"   最佳验证Loss: {best_val_loss:.4f} (Epoch {best_val_loss_epoch})")
            print(f"   最终验证Loss: {final_val_loss:.4f}")
            print(f"   验证Loss变化: {final_val_loss - best_val_loss:+.4f}")
            
            # 分析Loss趋势
            if final_val_loss > best_val_loss + 0.05:
                print(f"   ⚠️  验证Loss在最佳epoch后上升，可能存在过拟合")
            elif final_val_loss < best_val_loss:
                print(f"   ✅ 验证Loss持续下降，训练良好")
            else:
                print(f"   ✅ 验证Loss基本稳定")
            
            # 检查训练-验证Loss差距
            best_train_loss_at_best_epoch = float(train_loss_hist[best_val_loss_epoch - 1])
            loss_gap_at_best = best_train_loss_at_best_epoch - best_val_loss
            loss_gap_final = final_train_loss - final_val_loss
            
            print(f"\n   训练-验证Loss差距:")
            print(f"   最佳epoch时: {loss_gap_at_best:+.4f}")
            print(f"   最终epoch时: {loss_gap_final:+.4f}")
            
            if loss_gap_final > loss_gap_at_best + 0.05:
                print(f"   ⚠️  Loss差距增大，可能存在过拟合趋势")
            elif abs(loss_gap_final) < 0.1:
                print(f"   ✅ Loss差距合理，模型泛化良好")
        
        # 检查学习率变化
        if 'lr' in history.history:
            lr_hist = history.history['lr']
            print(f"\n📉 学习率分析:")
            print(f"   初始学习率: {lr_hist[0]:.6f}")
            print(f"   最终学习率: {lr_hist[-1]:.6f}")
            if lr_hist[0] == lr_hist[-1]:
                print(f"   ⚠️  学习率没有变化，建议添加学习率调度（ReduceLROnPlateau）")
            else:
                print(f"   ✅ 学习率已调整: {((lr_hist[-1] - lr_hist[0]) / lr_hist[0] * 100):.1f}%")
        else:
            print(f"\n📉 学习率分析:")
            print(f"   ⚠️  未记录学习率历史，建议添加ReduceLROnPlateau回调")
        
        print(f"{'='*60}\n")
        
        final_ckpt = ckpt_h5.replace(
            '_best.weights.h5', f'_best-epoch{best_epoch:02d}-valacc{best_val:.4f}.weights.h5'
        )

    meta_path = ckpt_h5.replace('.weights.h5', '_params.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump({
            "seed": seed,
            "best_epoch": best_epoch,
            "best_val_accuracy": best_val,
            "params": params
        }, f, ensure_ascii=False, indent=2)

print("全部循环结束")
