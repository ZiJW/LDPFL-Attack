from abc import ABC, abstractmethod
import torch
import logging
from tqdm import tqdm
import param
import os
import numpy as np
import pickle
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from network import load_comm
from model import load_model
from sklearn.manifold import MDS
from scipy.spatial.distance import pdist, squareform


class Base_server(ABC):
    def __init__(self, size, Model, Model_param, Epoch, comm="dist"):
        self.log_path = param.LOG_PATH + param.LOG_NAME
        os.makedirs(self.log_path, exist_ok=True)
        os.mkdir(self.log_path + "model")
        os.mkdir(self.log_path + "fig")
        os.mkdir(self.log_path + "res")
        if logging.getLogger().hasHandlers():
            logging.getLogger().handlers.clear()
        logging.basicConfig(filename=self.log_path + "/log_server.txt", format="%(asctime)s [%(levelname)s]: %(message)s", filemode="w", 
                            level=logging.INFO)
        os.system("cp param.py " + self.log_path + 
                  "param_{}client_{}rule_{}model_{}rnd_{}bcsize_{}lr_{}attacker_krum{}.py".format(param.N_NODES - 1, param.FL_RULE, 
                                                                                                  param.MODEL, param.N_ROUND, param.BATCH_SIZE_TRAIN,
                                                                                                  param.LEARNING_RATE, len(param.BAD_CLIENTS), param.MKRUM))

        self.epoch = Epoch
        self.size = size
        self.id = 0
        self.comm = load_comm(comm, self.id, self.size)
        self.model = load_model(Model, Model_param)
        self.temp_model = load_model(Model, Model_param)

    def serialize_model(self, type="concat") -> torch.Tensor:
        res = []
        for val in self.model.state_dict().values():
            res.append(val.view(-1))
        if type == "concat":
            res = torch.cat(res)
        elif type == "raw":
            pass
        else:
            raise ValueError("Invalid serialize type: {}".format(type))
        return res
    
    def unserialize_model(self, parameters: torch.Tensor):
        current_index = 0
        for val in self.model.state_dict().values():
            sz = val.numel()
            val.copy_(parameters[current_index: current_index + sz].view(val.shape))
            current_index += sz

    def unserialize_temp_model(self, parameters: torch.Tensor):
        current_index = 0
        for val in self.temp_model.state_dict().values():
            sz = val.numel()
            val.copy_(parameters[current_index: current_index + sz].view(val.shape))
            current_index += sz

    def test(self, ep: int = -1):
        """
            Test the accuracy and loss on testing dataset.
        """
        Acc, Loss = 0, 0.0
        LossList = []
        N = 0
        for data, target in tqdm(self.test_loader, desc="Test: "):
            with torch.no_grad():
                data, target = data.to(param.DEVICE), target.to(param.DEVICE)
                output = self.model(data)
                loss = self.criterion(output, target)
                _, pred = torch.max(output, dim=1)
                
                Loss += loss.item()
                LossList.append(loss.item())
                Acc += torch.sum(pred.eq(target)).item()
                N += len(target)

        # print("sum = {:.9f}, mean = {:.9f}, max = {:.9f}, min = {:.9f}".format(sum(LossList), sum(LossList) / len(self.test_loader), max(LossList), min(LossList)))
        Acc = Acc / N
        Loss = Loss / len(self.test_loader)
        logging.info('(Server) Epoch: {} Acc = {:.3f}, Loss: {:.9f}'.format(ep, Acc, Loss))
        # print('(Server) Epoch: {} Acc = {:.3f}, Loss: {:.9f}'.format(ep, Acc, Loss))
        return Acc, Loss
    
    def draw(self, data:list, ylabel:str, title:str):
        plt.figure(figsize=(10, 5))
        plt.plot(data, marker='o', linestyle='-', color='b', label='')
        plt.xlabel('Rounds')
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        figname = self.log_path + 'res/' + title + '.png'
        lstname = self.log_path + 'res/' + title + '.pkl'
        plt.savefig(figname)

        with open(lstname, "wb") as f:
            pickle.dump(data, f)
        logging.info("Success show and save {}".format(title))


    def select_multikrum(self, res_matrix, f, k):
        #通过multi-krum，从所有client模型参数中挑选k个参与聚合，
        res_matrix_tensor = torch.stack(res_matrix, dim=0)
        # 计算距离矩阵
        n = res_matrix_tensor.shape[0]
        dist_matrix = torch.zeros((n, n), device=param.DEVICE)
        # 使用广播和矩阵运算计算距离
        for i in range(n):
            dist_matrix[i] = torch.norm(res_matrix_tensor - res_matrix_tensor[i], dim=1)
        sorted_matrix, sorted_indices = torch.sort(dist_matrix, dim=1, descending=False)

        #MKrum对每个client的分数是自己和最近的n - f - 1个参数的距离之和
        MKrum_score = torch.sum(sorted_matrix[:, :n - f - 1 + 1], dim = 1)
        #因为和自己的距离是0，所以在n - f - 1上要加1
        sorted_score, indices = torch.sort(MKrum_score, dim=0, descending=False)
        selected_indices = indices[:k]
        return selected_indices

    def visualize_parameter(self, param_matrix, cap_name, save_path, mode="PCA", red_list=[], blue_list=[]):
        param_matrix_ = torch.stack(param_matrix).cpu().numpy()
        n = param_matrix_.shape[0]  # 获取模型数量

        # 生成颜色数组，默认所有点是灰色
        colors = np.array(['grey'] * n, dtype=object)  
        colors[blue_list] = 'blue'
        colors[red_list] = 'red' 
        intersection = list(set(red_list).intersection(set(blue_list)))
        colors[intersection] = 'purple'
        colors = colors.tolist()

        if mode == "PCA":
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(param_matrix_)
            plt.figure(figsize=(8,6))
            plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.7, c=colors, edgecolors='k')
            plt.xlabel("PCA Component 1")
            plt.ylabel("PCA Component 2")
            plt.title(cap_name)
            plt.savefig(save_path)
            plt.show()

        elif mode == "t-SNE":
            pca_50 = PCA(n_components=15)
            param_pca50 = pca_50.fit_transform(param_matrix_)
            tsne = TSNE(n_components=2, perplexity=15, random_state=42)
            tsne_result = tsne.fit_transform(param_pca50)
            plt.figure(figsize=(8,6))
            plt.scatter(tsne_result[:, 0], tsne_result[:, 1], alpha=0.7, c=colors, edgecolors='k')
            plt.xlabel("t-SNE Component 1")
            plt.ylabel("t-SNE Component 2")
            plt.title(cap_name)
            plt.savefig(save_path)
            plt.show()
        elif mode == "MDS" :
            def format_distance(value):
                """格式化距离矩阵中的数值，默认保留 4 位小数，过大或过小时使用科学计数法"""
                if abs(value) >= 10000 or (0 < abs(value) < 0.0001):
                    return "{:.4e}".format(value)  # 科学计数法
                else:
                    return "{:.4f}".format(value)  # 普通小数格式（4 位）
                
            """计算距离矩阵并保存 CSV优化数值格式"""
            # 计算距离矩阵
            distance_matrix = squareform(pdist(param_matrix_, metric='euclidean'))
            n = len(param_matrix_)

            # 格式化距离矩阵
            formatted_matrix = np.vectorize(format_distance)(distance_matrix)

            # 创建 DataFrame 并存储
            distance_df = pd.DataFrame(formatted_matrix, 
                                       index=[f"Point_{i}" for i in range(n)], 
                                       columns=[f"Point_{i}" for i in range(n)])
            assert save_path[-3:] == "png"
            distance_df.to_csv(save_path[:-3] + "csv")

            mds = MDS(n_components=2, dissimilarity="euclidean", random_state=42)
            mds_result = mds.fit_transform(param_matrix_)
            plt.figure(figsize=(8,6))
            plt.scatter(mds_result[:, 0], mds_result[:, 1], alpha=0.7, c=colors, edgecolors='k')

            # 在每个点上标出编号
            for i, (x, y) in enumerate(mds_result):
                plt.annotate(str(i), (x, y), textcoords="offset points", xytext=(5, 5), ha='right', fontsize=10, color='black')

            plt.xlabel("MDS Component 1")
            plt.ylabel("MDS Component 2")
            plt.title(cap_name)
            plt.savefig(save_path)
            plt.show()
        else:
            raise NotImplementedError



    @abstractmethod
    def evaluate(self):
        pass