import os
import numpy as np
import matplotlib.pyplot as plt

def plot_client_label_distribution(n, dir_path, save_dir=None, save_name="client_label_distribution.png"):
    """
    读取 n 个客户端的数据类别分布，并绘制堆叠条形图，同时可以选择保存图片。

    参数：
    n : int
        客户端数量
    dir_path : str
        包含 train_i_labels.npy 文件的文件夹路径
    save_dir : str, optional
        若不为空，则将图片保存到该目录（默认 None，即不保存）。
    save_name : str, optional
        保存的图片文件名（默认 "client_label_distribution.png"）。
    """

    # 读取所有客户端的类别数据
    label_counts = []
    
    for i in range(1, n + 1):
        file_path = os.path.join(dir_path, f"train_{i}_labels.npy")
        if not os.path.exists(file_path):
            print(f"警告: 文件 {file_path} 不存在，跳过该客户端。")
            continue

        # 加载 numpy 文件
        labels = np.load(file_path)
        
        # 统计类别分布
        unique, counts = np.unique(labels, return_counts=True)
        label_count_dict = dict(zip(unique, counts))
        label_counts.append(label_count_dict)

    # 获取所有可能的类别
    all_classes = sorted(set(key for label_count in label_counts for key in label_count.keys()))

    # 计算所有客户端的类别比例
    proportions = np.zeros((n, len(all_classes)))
    
    for client_idx, label_count in enumerate(label_counts):
        total_samples = sum(label_count.values())
        for class_idx, class_label in enumerate(all_classes):
            if class_label in label_count:
                proportions[client_idx, class_idx] = label_count[class_label] / total_samples

    # 绘制堆叠条形图
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bottom = np.zeros(n)
    
    for class_idx, class_label in enumerate(all_classes):
        ax.bar(range(n), proportions[:, class_idx], bottom=bottom, label=f"Class {class_label}")
        bottom += proportions[:, class_idx]

    ax.set_xlabel("Client Index")
    ax.set_ylabel("Class Proportion")
    ax.set_title("Client Label Distribution (Non-IID Visualization)")
    ax.legend(title="Class", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.xticks(range(n), [f"Client {i+1}" for i in range(n)], rotation=45)

    # 如果指定了保存路径，则保存图片
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)  # 确保目录存在
        save_path = os.path.join(save_dir, save_name)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"图片已保存至: {save_path}")

    plt.show()

# 示例调用
# plot_client_label_distribution(n=10, dir_path="your/dataset/path", save_dir="your/save/path")
