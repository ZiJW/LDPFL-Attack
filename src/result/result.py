import pickle
import matplotlib.pyplot as plt
import os

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
