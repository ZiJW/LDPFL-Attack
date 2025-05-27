import random
import time
import math
import logging
import torch
import queue
import threading
from tqdm import tqdm

from base_client import Base_client
from geom_median.torch import compute_geometric_median 
from util import load_dataset
import param
from param import DEVICE
    
def data_pertubation(W, c: float, r: float, eps: float, type: str = "normal"):
    sz = len(W)
    with torch.no_grad():
        # torch.clamp(W, min=c-r, max=c+r)
        coff = (math.exp(eps) + 1) / (math.exp(eps) - 1)

        Pb = (((W - c) / coff) + r)
        rnd = (torch.rand(sz) * 2.0 * r).to(DEVICE)
        cmp = torch.gt(Pb, rnd).to(DEVICE)
        if type == "bad":
            cmp = ~cmp

        res = ((cmp) * (c + r * coff)) + ((~cmp) * (c - r * coff))
    return res

class LDPFL_client(Base_client):
    """
        The client in LDP-FL
    """
    def __init__(self, id):
        super().__init__(id, param.N_NODES, param.MODEL, param.MODEL_PARAM, 
                         param.OPTIMIZER, param.LEARNING_RATE, param.CRITERION, comm=param.COMM)
        self.train_loader, = load_dataset(param.DATASET, param.FOLDER, [("train", True)], idx=self.id)
        self.epoch = param.N_EPOCH
        self.round = param.N_ROUND

        if param.CLIENTS_WEIGHTS != None:
            logging.info("Client {} has weight {}".format(self.id, param.CLIENTS_WEIGHTS[self.id]))
        if self.id in param.BAD_CLIENTS:
            logging.info("Client {} is an adversary!".format(self.id))
        self.weights_buffer = [queue.Queue(maxsize=self.size - 1) for idx in range(len(self.model_size))]

    def chose(self):
        logging.debug("Client {}: wait for invitation ...".format(self.id))
        chose = self.comm.recv(0)
        self.comm.send(0, "ACK")
        logging.debug("Client {}: receive invitation {}".format(self.id, chose))
        return chose
    
    def train(self):
        """
            Training for 1 round.
        """        
        # Download the global model parameters
        global_model, weight_range = self.comm.recv(0)
        logging.debug("Client {}: get global weights from server".format(self.id))

        self.unserialize_model(global_model)
        logging.debug("Client {}: training({}) ...".format(self.id, len(self.train_loader)))
        for ep in range(self.epoch):
            for batch_idx, (data, target) in enumerate(self.train_loader):
                # Training
                data, target = data.to(DEVICE), target.to(DEVICE)
                self.optimizer.zero_grad()
                output = self.model(data)
                if self.id in param.BAD_CLIENTS:
                    loss = -self.criterion(output, target)
                else:
                    loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
        return weight_range

    def fit_one_epoch(self, rev=False):
        for ep in range(self.epoch):
            for batch_idx, (data, target) in enumerate(self.train_loader):
                # Training
                data, target = data.to(DEVICE), target.to(DEVICE)
                self.optimizer.zero_grad()
                output = self.model(data)
                if rev:
                    loss = -self.criterion(output, target)
                else:
                    loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
        return self.serialize_model()

    def adv_train(self, global_model, weight_range, rn, mode='front'):
        if mode == 'front':
            logging.debug("Client {} generating round {} front adversarial parameter".format(self.id, rn))
            self.unserialize_model(global_model)
            self.fit_one_epoch(rev=True)
            return self.handle_weights(weight_range, param.EPS)
        elif mode == "front-total-loss":
            logging.debug("Client {} generating round {} back adversarial parameter with total loss".format(self.id, rn))
            self.unserialize_model(global_model)
            estimate_param = self.fit_one_epoch(rev=False)
            self.unserialize_model(global_model)
            param_after_agg = self.fit_one_epoch(rev=True)
            param_adv = ( param_after_agg * (self.size - 1) - estimate_param * (self.size - 1 - len(param.BAD_CLIENTS)) )  / len(param.BAD_CLIENTS)
            self.unserialize_model(param_adv)
            return self.handle_weights(weight_range, param.EPS)
        elif mode == 'back-total-loss':
            logging.debug("Client {} generating round {} back adversarial parameter with total loss".format(self.id, rn))
            self.unserialize_model(global_model)
            estimate_param = self.fit_one_epoch(rev=False)
            self.unserialize_model(global_model)
            param_after_agg = self.fit_one_epoch(rev=True)
            param_adv = ( param_after_agg * (self.size - 1) - estimate_param * (self.size - 1 - len(param.BAD_CLIENTS)) )  / len(param.BAD_CLIENTS)
            return self.clip_2val(self.flatten2layer(param_adv), weight_range)
        elif mode == 'back':
            logging.debug("Client {} generating round {} back adversarial parameter".format(self.id, rn))
            self.unserialize_model(global_model)
            param_adv = self.fit_one_epoch(rev=True)
            return self.clip_2val(self.flatten2layer(param_adv), weight_range)
        elif mode == "random-noise":
            #return self.flatten2layer(torch.zeros(global_model.shape, device=DEVICE))
            random_posneg = torch.randint(0, 2, global_model.shape).to(DEVICE) * 2 - 1 
            random_posneg = self.flatten2layer(random_posneg)
            eps = param.EPS
            for idx in range(len(random_posneg)):
                coff = (math.exp(eps) + 1) / (math.exp(eps) - 1)
                c, r = weight_range[idx]["center"], weight_range[idx]["range"]
                random_posneg[idx] = c + random_posneg[idx] * r * coff
            return random_posneg

    def receive_global_model_range(self):
        global_model, weight_range = self.comm.recv(0)
        logging.debug("Client {}: get global weights from server".format(self.id))
        return global_model, weight_range
    
    def receive_other_parameters(self):
        record_param = self.comm.recv(0)
        logging.debug("Client {} got other clients' parameter from server".format(self.id))
        param_list = [paramlist for paramlist in record_param if paramlist != []]
        assert len(param_list) == self.size - 1 - len(param.TAPPING_CLIENTS)
        #compute_geometric_median(param_list).median
        return param_list

    def get_2val_geomedian(self, weight_range, good_params):
        #定义LDPFL下的集合平均值，对于每一个参数，如果同位置的C + r多，取C+r，反之亦然，如果两者一样多，取C
        num_layers = len(good_params[0])  # 获取层数
        num_clients = len(good_params)  # 获取客户端数
        result = []
    
        for layer in range(num_layers):
            layer_params = torch.stack([client[layer] for client in good_params])  # 获取该层的所有客户端参数 (num_clients, param_size)
            unique_values = torch.unique(layer_params)  # 该层两个可能的值

            if len(unique_values) > 2:
                print(unique_values)
                raise ValueError(f"Layer {layer} does not have exactly two unique values.")

            c, r = weight_range[layer]["center"], weight_range[layer]['range']  # 获取该层的center值
            eps = param.EPS
            coff = (math.exp(eps) + 1) / (math.exp(eps) - 1)
            C_k1, C_k2 = c + r * coff, c - r * coff
            # 统计每个位置 C_k1 和 C_k2 出现的次数
            count_C_k1 = (layer_params > c).sum(dim=0)
            count_C_k2 = (layer_params < c).sum(dim=0)

            # 选择出现次数最多的值
            majority_mask = count_C_k1 > count_C_k2
            tie_mask = count_C_k1 == count_C_k2

            # 生成最终结果
            aggregated_tensor = torch.where(majority_mask, C_k1, C_k2)
            aggregated_tensor = torch.where(tie_mask, C_k1, aggregated_tensor)

            result.append(aggregated_tensor)
        return result
    
    def get_median(self, good_params):
        temp_params = [torch.cat(good_params[idx]) for idx in range(len(good_params))]
        return torch.median(torch.stack(temp_params), dim=0)[0]

    def clip_2val(self, model_param, weight_range):
        eps = param.EPS
        for idx in range(len(model_param)):
            coff = (math.exp(eps) + 1) / (math.exp(eps) - 1)
            c, r = weight_range[idx]["center"], weight_range[idx]["range"]
            model_param[idx] = torch.where(model_param[idx] > c,c + r * coff, c - r * coff)
        return model_param
    
    def flatten2layer(self, params):
        current_index = 0
        result = []
        model_keys = [name for name, _ in self.model.named_parameters()]
        for name in model_keys:
            val = self.model.state_dict()[name]
            sz = val.numel()
            result.append(params[current_index: current_index + sz].view(-1))
            current_index += sz
        return result
    def layer2flatten(self, params):
        return torch.cat(params)
    
    def generate_bad_param(self, global_model, weight_range, good_params, rn):
        #方法：取其他加上自己的平均值，反向训练，每次用二值剪裁参数差值（即自己所需要的参数）
        median = self.get_2val_geomedian(weight_range, good_params) #初始化为几何平均数
        median_flatten = torch.cat(median)
        good_num, bad_num = len(good_params), len(param.TAPPING_CLIENTS)
        all_num = (good_num + bad_num)
        good_flatten_params = [torch.cat(client_param) for client_param in good_params]
        good_flatten_sum = torch.sum(torch.stack(good_flatten_params, dim=0), dim=0)
        avg_params = (median_flatten * bad_num + good_flatten_sum) / all_num
        self.unserialize_model(avg_params)
        for i in tqdm(range(param.ADVERSARY_ITERATION), desc="client {} round {} adversarial training : ".format(self.id, rn)):
            for data, target in self.train_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                self.optimizer.zero_grad()
                output = self.model(data)

                # 计算 loss（取反方向优化）
                loss = -self.criterion(output, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                self.optimizer.step()
            avg_params = self.serialize_model()
            adv_params = (avg_params * all_num - good_flatten_sum) / bad_num
            tmp = self.clip_2val(self.flatten2layer(adv_params), weight_range)
            #添加约束，来绕过krum
            adv_params = self.layer2flatten(tmp)
            avg_params = (adv_params * bad_num + good_flatten_sum) / all_num
            self.unserialize_model(avg_params)
        return self.flatten2layer(adv_params)

        """median = self.get_2val_geomedian(weight_range, good_params)
        self.unserialize_model(global_model)
        for i in tqdm(range(param.ADVERSARY_ITERATION), desc="client {} round {} adversarial training : ".format(self.id, rn)):
            for data, target in self.train_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)

                self.optimizer.zero_grad()
                output = self.model(data)

                # 计算 loss（取反方向优化）
                loss = -self.criterion(output, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                self.optimizer.step()
        #model_param = self.handle_weights(weight_range, param.EPS)
        model_param = self.serialize_model(type="raw")
        eps = param.EPS
        for idx in range(len(model_param)):
            coff = (math.exp(eps) + 1) / (math.exp(eps) - 1)
            c, r = weight_range[idx]["center"], weight_range[idx]["range"]
            model_param[idx] = torch.where(model_param[idx] > c,c + r * coff, c - r * coff)
        selected_layer = param.ADVERSARY_LAYER
        for idx in selected_layer:
            median[idx] = model_param[idx]
        return median"""
    
    def get_min_diff(self, weight_range, good_params, rn, k=1):
        median = self.get_2val_geomedian(weight_range, good_params) #几何平均数
        cnt = torch.zeros(len(good_params))
        for idx in range(len(good_params[0])):
            c, r = weight_range[idx]['center'], weight_range[idx]['range']
            median_offset = median[idx] > c
            good_offset = [client_param[idx] > c for client_param in good_params]
            diff_cnt = [torch.sum(median_offset ^ offset) for offset in good_offset]
            cnt += torch.tensor(diff_cnt)
        val, idx = torch.topk(cnt, k, dim=0, largest=False)
        return val.max().int().item()
            

    def generate_bad_param_selective(self, global_model, weight_range, good_params, rn, topk=10000):
        """
        基于反向训练后偏移度选择关键参数进行模型投毒。
        仅对 top-k 偏移最大的维度进行扰动，并排除掉与 median 一致的维度。
        """
        import torch.nn.functional as F

        # 初始化模型
        median = self.get_2val_geomedian(weight_range, good_params)
        median_flatten = torch.cat(median)
        good_num, bad_num = len(good_params), len(param.TAPPING_CLIENTS)
        all_num = good_num + bad_num
        good_flatten_params = [torch.cat(p) for p in good_params]
        good_flatten_sum = torch.sum(torch.stack(good_flatten_params, dim=0), dim=0)

        # 初始设为中位数 + good 平均
        avg_params = (median_flatten * bad_num + good_flatten_sum) / all_num
        self.unserialize_model(avg_params)

        # === 1. 执行反向训练 ===
        for i in tqdm(range(param.ADVERSARY_ITERATION), desc="client {} round {} adversarial training : ".format(self.id, rn)):
            for data, target in self.train_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = -self.criterion(output, target)  # 反方向优化
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                self.optimizer.step()

        # === 2. 获取训练后模型参数 ===
        final_param = self.serialize_model()
        diff_vector = final_param - median_flatten

        # === 3. 计算每层归一化后的偏离度 ===
        offset_scores = []
        offset_masks = []
        ptr = 0
        all_adv_param = self.clip_2val(self.flatten2layer(final_param), weight_range)
        all_adv_flatten = torch.cat(all_adv_param)

        for idx, w in enumerate(weight_range):
            length = len(median[idx].view(-1))
            center, radius = w['center'], w['range']
            coff = ((math.exp(param.EPS)+1)/(math.exp(param.EPS)-1)) 
            norm_factor = radius * coff + 1e-6
            layer_diff = diff_vector[ptr:ptr+length] / norm_factor  # 归一化

            # 生成不选 mask：该维度和 median 中一致的不要
            median_val = median[idx].view(-1)
            adv_val = all_adv_flatten[ptr:ptr+length]
            valid_mask = ((median_val - center) * (adv_val - center) <= 0.0)
            offset_scores.append(torch.abs(layer_diff))
            offset_masks.append(valid_mask)
            ptr += length

        # === 4. 合并并选 top-k ===
        flat_scores = torch.cat(offset_scores)
        flat_mask = torch.cat(offset_masks)
        max_diff = flat_mask.sum().item()
        flat_valid_scores = flat_scores * flat_mask.float()  # 不合法位置得分为0
        topk_val, topk_idx = torch.topk(flat_valid_scores, min(max_diff, topk))
        critical_mask = torch.zeros_like(flat_scores, dtype=torch.bool)
        critical_mask[topk_idx] = True

        # === 5. 生成最终扰动参数 ===
        adv_param = median_flatten.clone()
        adv_param[critical_mask] = all_adv_flatten[critical_mask]

        return self.flatten2layer(adv_param)
    
    def generate_bad_param_fisher(self, global_model, weight_range, good_params, rn, topk=10000, fisher_batch_num=3):
        """
        使用 Fisher 信息计算参数重要性，选择 top-k 参数维度进行有针对性的模型投毒。
        """
        median = self.get_2val_geomedian(weight_range, good_params)
        median_flatten = torch.cat(median)
        good_num, bad_num = len(good_params), len(param.TAPPING_CLIENTS)
        all_num = good_num + bad_num
        good_flatten_params = [torch.cat(p) for p in good_params]
        good_flatten_sum = torch.sum(torch.stack(good_flatten_params, dim=0), dim=0)

        # 设置模型初始状态
        avg_params = (median_flatten * bad_num + good_flatten_sum) / all_num
        self.unserialize_model(avg_params)
        self.model.train()

        # === Step 1: 计算 Fisher 信息分数（梯度平方平均）并考虑radius在内 ===
        fisher_scores = [torch.zeros_like(p.data) for p in self.model.parameters()]
        batch_count = 0

        for batch in self.train_loader:
            data, target = batch
            data, target = data.to(DEVICE), target.to(DEVICE)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()

            # 梯度平方累加
            for idx, params in enumerate(self.model.parameters()):
                if params.grad is not None:
                    fisher_scores[idx] += (params.grad.detach() ** 2)
            batch_count += 1
            if batch_count >= fisher_batch_num:
                break

        fisher_scores = [score / batch_count for score in fisher_scores]
        flat_scores = torch.cat([s.view(-1) for s in fisher_scores])
        if param.FISHER_NORMALIZE :
            normalizers = []
            ptr = 0
            for idx, w in enumerate(weight_range):
                length = len(median[idx].view(-1))
                radius = w['range']
                eps = param.EPS
                coff = (math.exp(eps) + 1) / (math.exp(eps) - 1)
                norm = radius * coff + 1e-6  # 避免除以0
                normalizers.append(torch.full((length,), norm, device=DEVICE))
                ptr += length
    
            normalizer_vector = torch.cat(normalizers)
            flat_scores = flat_scores / normalizer_vector  # scale down high-risk layers

        # === Step 2: 执行 adversarial training ===
        for i in tqdm(range(param.ADVERSARY_ITERATION), desc="client {} round {} adversarial training : ".format(self.id, rn)):
            for data, target in self.train_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = -self.criterion(output, target)  # 反向训练
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                self.optimizer.step()

        final_param = self.serialize_model()
        all_adv_param = self.clip_2val(self.flatten2layer(final_param), weight_range)
        all_adv_flatten = torch.cat(all_adv_param)

        # === Step 3: 构建扰动掩码，只考虑与 median 不一致的维度 ===
        offset_masks = []
        ptr = 0
        for idx, w in enumerate(weight_range):
            length = len(median[idx].view(-1))
            center = w['center']
            median_val = median[idx].view(-1)
            adv_val = all_adv_flatten[ptr:ptr+length]
            valid_mask = ((median_val - center) * (adv_val - center) <= 0.0)
            offset_masks.append(valid_mask)
            ptr += length

        flat_mask = torch.cat(offset_masks)
        valid_scores = flat_scores * flat_mask.float()

        # === Step 4: Top-k 筛选 + 应用扰动 ===
        topk_val, topk_idx = torch.topk(valid_scores, topk)
        critical_mask = torch.zeros_like(flat_scores, dtype=torch.bool)
        critical_mask[topk_idx] = True

        adv_param = median_flatten.clone()
        adv_param[critical_mask] = all_adv_flatten[critical_mask]

        return self.flatten2layer(adv_param)


    def handle_weights(self, weight_range, eps):
        """
            Add noise on weights.
        """
        global_model = self.serialize_model(type="raw")
        for idx in range(len(global_model)):
            logging.debug("Client {}: [{}] c={}, r={}".format(self.id, idx, weight_range[idx]["center"], weight_range[idx]["range"]))
            global_model[idx] = data_pertubation(global_model[idx], weight_range[idx]["center"], weight_range[idx]["range"], eps)
        return global_model
    
    def send_weights(self, global_model):
        latency = [(random.uniform(0, param.LATENCY_T), i) for i in range(len(global_model))]
        latency.sort()
        for idx, (lt, ln) in enumerate(latency):
            if idx == 0:
                time.sleep(lt)
            else:
                time.sleep(lt - latency[idx - 1][0])
            logging.debug("Send layer {} = {}".format(ln, global_model[ln][:10]))
            self.comm.send(0, {"ln": ln, "weight": global_model[ln]})
            if idx < len(latency) - 1:
                assert self.comm.recv(0) == "ACK"

        
    def evaluate(self):
        """
            Train the model on all rounds.
        """
        self.comm.initialize()
        for rn in range(self.round):
            if self.chose():
                if self.id in param.BAD_CLIENTS :
                    if self.id in param.TAPPING_CLIENTS:
                        global_model, weight_range = self.receive_global_model_range()
                        good_params = self.receive_other_parameters()
                        if param.TAPPING_SAME and self.id != param.TAPPING_CLIENTS[0]:
                            self.send_weights(self.flatten2layer(global_model))
                            continue
                        max_allow_diff_dim = self.get_min_diff(weight_range, good_params, rn, k=1)
                        adv_param = self.generate_bad_param_selective(global_model, weight_range, good_params, rn, 
                                                                      topk=int(param.ADVERSARY_SCALE[rn] * max_allow_diff_dim))
                        self.send_weights(adv_param)
                    else :
                        global_model, weight_range = self.receive_global_model_range()
                        adv_param = self.adv_train(global_model, weight_range, rn, mode=param.ATTACK_MODE)
                        self.send_weights(adv_param)
                else :
                    weight_range = self.train()
                    global_model = self.handle_weights(weight_range, param.EPS)
                    self.send_weights(global_model)
                    
                