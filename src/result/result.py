import pickle
import matplotlib.pyplot as plt
import os
import numpy as np

def plot_accuracy_curves(accuracy_dict, save_path="round2aacurancy.png", caption = None):
    """
    参数：
    accuracy_dict: dict[str, str]
        key 是方法名 (将显示在图例中)，value 是对应的 .pkl 文件路径，文件内容应是 list 或 numpy array，表示每一轮的准确率。

    返回：
    一张显示所有方法准确率-轮数关系的折线图。
    """

    plt.figure(figsize=(10, 6))
    acc_dict = {}

    for method_name, pkl_path in accuracy_dict.items():
        if not pkl_path.endswith("Accurancy.pkl"):
            pkl_path = os.path.join(pkl_path, "res/Accurancy.pkl")
        assert os.path.exists(pkl_path)
        with open(pkl_path, 'rb') as f:
            acc_list = pickle.load(f)

        plt.plot(acc_list, label=method_name)
        acc_dict[method_name] = acc_list
    
    for name in acc_dict.keys():
        print("{} max accurancy {}".format(name, max(acc_dict[name])))

    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    plt.title(caption)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def plot_accuracy_drop(accuracy_dict, save_path, caption):
    attacker_percentages = [0, 5, 10, 15, 20, 25]
    marker_list = ['o', 's', '^', 'D', 'P', 'X', '*']
    line_styles = ['-', '--', '-.', ':']
    
    plt.figure(figsize=(7, 5))
    
    for i, (method, paths) in enumerate(accuracy_dict.items()):
        baseline_path = paths["0% attacker"]
        with open(os.path.join(baseline_path, "res/Accurancy.pkl"), "rb") as f:
            baseline_acc = pickle.load(f)
        baseline_max = max(baseline_acc)

        drops = []
        for p in attacker_percentages:
            key = f"{p}% attacker"
            with open(os.path.join(paths[key], "res/Accurancy.pkl"), "rb") as f:
                acc = pickle.load(f)
            drop = baseline_max - max(acc)
            drops.append(drop)

        plt.plot(attacker_percentages, drops, 
                 label=method, 
                 marker=marker_list[i % len(marker_list)], 
                 linestyle=line_styles[i % len(line_styles)], 
                 linewidth=2)

    plt.xlabel("Percentage of Compromised Worker Devices (%)")
    plt.ylabel("Accuracy Drop")
    plt.title(caption)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    plt.close()    

def plot_error_rate_bars(accuracy_dict, save_path="round2accuracy.png", caption=None):
    attacker_counts = ["0 attacker", "1 attacker", "2 attacker", "3 attacker"]
    a_values = list(accuracy_dict.keys())

    bar_width = 0.2
    spacing = 1.0  # space between different 'a' groups
    group_width = bar_width * len(attacker_counts)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(attacker_counts)))

    xticks = []
    xtick_labels = []

    for i, a_key in enumerate(a_values):
        base_x = i * (group_width + spacing)
        xticks.append(base_x + 1.5 * bar_width)  # Center of the group
        raw_label = a_key.split()[-1]
        xtick_labels.append(f"\u03B1={raw_label}")

        for j, attacker in enumerate(attacker_counts):
            acc_path = os.path.join(accuracy_dict[a_key][attacker], "res/Accurancy.pkl")
            with open(acc_path, "rb") as f:
                acc = pickle.load(f)
            max_acc = max(acc.values()) if isinstance(acc, dict) else max(acc)
            error_rate = 1.0 - max_acc
            bar_pos = base_x + j * bar_width
            ax.bar(bar_pos, error_rate, width=bar_width, color=colors[j], edgecolor='black')

    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels, rotation=0)
    ax.set_xlabel("Dirichlet α")
    ax.set_ylabel("Error Rate (1 - max accuracy)")
    ax.set_title(caption)
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)

    # Legend for attacker count colors
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=colors[j], edgecolor='black') for j in range(len(attacker_counts))
    ]
    ax.legend(legend_handles, [f"{j} attacker" for j in range(len(attacker_counts))], title="Attacker Count")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    plt.close()