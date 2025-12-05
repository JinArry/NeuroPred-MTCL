import torch
import esm
import numpy as np
import pandas as pd
import os

def load_data(path="data/"):
    df_train = pd.read_csv(path + 'training.csv')
    df_val = pd.read_csv(path + 'testing.csv')
    # df_test = pd.read_csv(path + 'test.csv')   # 新增

    train_sequences = df_train["Seq"].tolist()
    train_labels = df_train["Label"].tolist()
    val_sequences = df_val["Seq"].tolist()
    val_labels = df_val["Label"].tolist()
    # test_sequences = df_test["Seq"].tolist()
    # test_labels = df_test["Label"].tolist()

    train_dict = {"text": train_sequences, 'labels': train_labels}
    val_dict = {"text": val_sequences, 'labels': val_labels}
    # test_dict = {"text": test_sequences, 'labels': test_labels}
    return train_dict, val_dict


def extract_esm_seq_features(sequences, labels, esm_model, batch_converter, device, layer=6, batch_size=32):
    all_features = []
    all_labels = []
    max_len = 0

    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i:i + batch_size]  # 当前batch的序列
            batch_labels = labels[i:i + batch_size]  # 当前batch的标签
            # 构造[("seq0", seq0), ("seq1", seq1), ...]格式，ESM要求的格式
            batch_data = [("seq%d" % idx, seq) for idx, seq in enumerate(batch_seqs)]
            batch_tokens = batch_converter(batch_data)[2].to(device)  # 转成token tensor并放到设备上
            # 前向推理，输出某一层的隐藏特征
            results = esm_model(batch_tokens, repr_layers=[layer], return_contacts=False)
            token_reps = results["representations"][layer]  # shape: [batch, seq_len+2, 768]
            for j in range(token_reps.size(0)):
                # 去掉起止标记[CLS][EOS]，只保留氨基酸token的特征
                rep = token_reps[j, 1:-1].cpu().numpy()  # shape: [seq_len, 768]
                all_features.append(rep)  # 保存特征
                all_labels.append(batch_labels[j])  # 保存标签
                if rep.shape[0] > max_len:
                    max_len = rep.shape[0]  # 动态更新最大序列长度
            print(f"Processed {i + len(batch_seqs)} / {len(sequences)}")  # 打印进度

    # 对所有样本做padding，pad到最大长度
    print("Max sequence length:", max_len)
    padded_features = []
    for feat in all_features:
        pad_len = max_len - feat.shape[0]
        if pad_len > 0:
            # 右侧补0，保证所有样本shape一致：[max_len, 768]
            pad = np.zeros((pad_len, 768))
            feat = np.concatenate([feat, pad], axis=0)
        padded_features.append(feat)
    # 堆叠成大数组 [样本数, max_seq_len, 768]
    padded_features = np.stack(padded_features)
    all_labels = np.array(all_labels)
    return padded_features, all_labels


if __name__ == "__main__":
    esm_model_name = "esm1_t6_43M_UR50S"  # 使用ESM官方的T6模型
    data_path = "features/"  # 保存npy的目录（与训练/评估脚本一致）
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. 加载ESM模型和batch_converter
    esm_model, alphabet = esm.pretrained.load_model_and_alphabet(esm_model_name)
    batch_converter = alphabet.get_batch_converter()
    esm_model.eval()  # 进入推理模式
    esm_model = esm_model.to(device)

    # 2. 加载训练和验证集数据
    # 加载训练与验证集CSV（默认从 data/ 目录读取）
    train_dict, val_dict = load_data("data/")  # 返回包含 "text" 和 "labels" 的字典

    # 3. 提取训练集序列特征，并保存
    print("Extracting training features (sequences)...")
    train_features, train_labels = extract_esm_seq_features(
        train_dict["text"], train_dict["labels"], esm_model, batch_converter, device
    )
    os.makedirs(data_path, exist_ok=True)
    np.save(os.path.join(data_path, "train_esm_seq.npy"), train_features)
    np.save(os.path.join(data_path, "train_labels_seq.npy"), train_labels)
    print(f"Saved train_esm_seq.npy shape: {train_features.shape}")

    # 4. 提取验证集序列特征，并保存
    print("Extracting validation features (sequences)...")
    val_features, val_labels = extract_esm_seq_features(
        val_dict["text"], val_dict["labels"], esm_model, batch_converter, device
    )
    np.save(os.path.join(data_path, "val_esm_seq.npy"), val_features)
    np.save(os.path.join(data_path, "val_labels_seq.npy"), val_labels)
    print(f"Saved val_esm_seq.npy shape: {val_features.shape}")
