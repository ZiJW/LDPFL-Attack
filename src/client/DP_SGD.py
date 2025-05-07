import torch
import logging
from geom_median.torch import compute_geometric_median 
from base_client import Base_client
from model import load_model, load_criterion, load_optimizer
from util import load_dataset, ExpM, pm_perturbation
from param import DEVICE
import param
from tqdm import tqdm
from opacus import PrivacyEngine
from opacus.accountants import RDPAccountant
from opacus.utils.batch_memory_manager import BatchMemoryManager

class DPSGD_client(Base_client):  
    """
        The client in FedSel
    """

    def __init__(self, id):
        super().__init__(id, param.N_NODES, param.MODEL, param.MODEL_PARAM,
                         param.OPTIMIZER, param.LEARNING_RATE, param.CRITERION, comm=param.COMM)
        self.train_loader = load_dataset(param.DATASET, param.FOLDER, [("train", True)], idx=self.id)[0]
        self.train_data_size = sum([len(x) for x, y in self.train_loader])
        logging.info("Client {} has {} data".format(self.id, self.train_data_size))
        self.valid_loader = load_dataset(param.DATASET, param.FOLDER, [("public", False)])[0] if id in param.BAD_CLIENTS else []
        self.round = param.N_ROUND
        self.privacy1_percent = 0.1
        self.ldp = param.LDP and self.id not in param.BAD_CLIENTS
        self.accountant = RDPAccountant()

        if self.id in param.BAD_CLIENTS:
            logging.info("Client {} is an adversary!".format(self.id))

    def fit(self, global_model, rn):
        self.unserialize_model(global_model)
        for epoch in range(param.N_EPOCH):
            theta = self.serialize_model()
            gradients = torch.zeros(global_model.shape).to(param.DEVICE)
            sample_count = 0
            for (x, y) in self.train_loader:
                if torch.rand(1).item() > param.P:
                    continue
                x, y = x.to(DEVICE), y.to(DEVICE)
                sample_count += len(x)
                self.model.zero_grad()
                outputs = self.model(x)
                loss = self.criterion(outputs, y)
                loss.backward()
                
                gradient = []
                for val in self.model.parameters():
                    gradient.append(val.grad.view(-1))
                gradient = torch.cat(gradient)
                norm = torch.sqrt(torch.sum(torch.pow(gradient, 2)))
                if torch.isnan(norm) or torch.isinf(norm):
                    logging.warning("Gradient norm is NaN or Inf, skipping batch.")
                    return global_model
                if norm > param.NORM_BOUND:
                    gradient.div_(norm.div(param.NORM_BOUND))
                gradients += gradient * len(x)
            if sample_count == 0:
                logging.debug("    Round {} epoch {} select {} / {} samples".format(rn, epoch, sample_count, self.train_data_size))
                continue  # avoid division by zero
            noise = self.GaussianNoise(param.SIGMA * param.NORM_BOUND, gradients.shape)
            gradients += noise
            gradients = gradients.div(sample_count)
            self.accountant.step(noise_multiplier=param.SIGMA, sample_rate=(sample_count / self.train_data_size))
            logging.debug("    Round {} epoch {} select {} / {} samples norm {}".format(rn, epoch, sample_count, self.train_data_size, norm))
            # Gradient descent step
            self.unserialize_model(theta - param.LEARNING_RATE * gradients)
        logging.debug("Round {} finished training with eps {}".format( rn, self.accountant.get_epsilon(delta=param.DELTA)))
        return self.serialize_model()

    def fit_opacus(self, global_model, rn):
        self.unserialize_model(global_model)
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=20.0)  #防止NAN
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        torch.cuda.empty_cache()
        logging.debug("Epoch {} train | Acc: {} | Loss: {} | Eps : {}".format(rn, 100.*correct/total, train_loss/len(self.train_loader), self.privacyEngine.get_epsilon(delta=param.DELTA)))
        return self.serialize_model()

    def fit_logicalbatch(self, global_model, rn):
        self.unserialize_model(global_model)
        train_loss = 0
        correct = 0
        total = 0
        with BatchMemoryManager(data_loader=self.train_loader, max_physical_batch_size=8, optimizer=self.optimizer) as memory_safe_data_loader:
            for batch_idx, (inputs, targets) in enumerate(memory_safe_data_loader):
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=20.0)  #防止NAN
                self.optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        torch.cuda.empty_cache()
        logging.debug("Epoch {} train | Acc: {} | Loss: {} | Eps : {}".format(rn, 100.*correct/total, train_loss/len(self.train_loader), self.privacyEngine.get_epsilon(delta=param.DELTA)))
        return self.serialize_model()

    def fit_one_epoch(self, rn, epoch, rev=False, add_noise=False):
        theta = self.serialize_model()
        gradients = torch.zeros(theta.shape).to(param.DEVICE)
        sample_count = 0
        for (x, y) in self.train_loader:
            if torch.rand(1).item() > param.P:
                continue
            x, y = x.to(DEVICE), y.to(DEVICE)
            sample_count += len(x)
            self.model.zero_grad()
            outputs = self.model(x)
            loss = - self.criterion(outputs, y) if rev else self.criterion(outputs, y)
            loss.backward()
            
            gradient = []
            for val in self.model.parameters():
                gradient.append(val.grad.view(-1))
            gradient = torch.cat(gradient)
            norm = torch.sqrt(torch.sum(torch.pow(gradient, 2)))
            if torch.isnan(norm) or torch.isinf(norm):
                logging.warning("Gradient norm is NaN or Inf, skipping batch.")
                return theta
            if norm > param.NORM_BOUND:
                if add_noise: gradient.div_(norm.div(param.NORM_BOUND))
            gradients += gradient * len(x)
        if sample_count == 0:
            logging.debug("    Round {} epoch {} select {} / {} samples".format(rn, epoch, sample_count, self.train_data_size))
            return theta
        noise = self.GaussianNoise(param.SIGMA * param.NORM_BOUND, gradients.shape)
        if add_noise : gradients += noise
        gradients = gradients.div(sample_count)
        self.accountant.step(noise_multiplier=param.SIGMA, sample_rate=(sample_count / self.train_data_size))
        logging.debug("    Round {} epoch {} select {} / {} samples norm {}".format(rn, epoch, sample_count, self.train_data_size, norm))
        # Gradient descent step
        self.unserialize_model(theta - param.LEARNING_RATE * gradients)
        return self.serialize_model()

    def generate_bad_gradients(self, global_model, rn, mode='front'):
        assert mode in {"front", "back", "front-norm-bound", "back-total-loss"}
        if mode.startswith("front"):
            logging.debug("Client {} generating round {} front adversarial parameter".format(self.id, rn))
            self.unserialize_model(global_model)
            for i in tqdm(range(param.ADVERSARY_ITERATION[rn]), desc="client {} round {} adversarial training : ".format(self.id, rn)):
                self.fit_one_epoch(rn, i, rev=True, add_noise=True)
            theta = self.serialize_model()
            return theta
        elif mode == "back":
            logging.debug("Client {} generating round {} back adversarial parameter".format(self.id, rn))
            self.unserialize_model(global_model)
            for i in tqdm(range(param.ADVERSARY_ITERATION[rn]), desc="client {} round {} adversarial training : ".format(self.id, rn)):
                self.fit_one_epoch(rn, i, rev=True, add_noise=False)
            theta = self.serialize_model()
            return theta
        elif mode == "back-total-loss":
            logging.debug("Client {} generating round {} back adversarial parameter with total loss".format(self.id, rn))
            self.unserialize_model(global_model)
            for i in tqdm(range(param.N_EPOCH), desc="client {} round {} estimate training : ".format(self.id, rn)):
                self.fit_one_epoch(rn, i, rev=False, add_noise=True)
            estimate_param = self.serialize_model()
            self.unserialize_model(global_model)
            param_after_agg = global_model
            for i in tqdm(range(param.ADVERSARY_ITERATION[rn]), desc="client {} round {} adversarial training : ".format(self.id, rn)):
                param_after_agg = self.fit_one_epoch(rn, i, rev=True, add_noise=False)
            param_adv = ( param_after_agg * (self.size - 1) - estimate_param * (self.size - 1 - len(param.BAD_CLIENTS)) )  / len(param.BAD_CLIENTS)
            return param_adv

    def test_on_public(self, model):
        """
            Test the accuracy and loss on validation dataset.
        """
        Acc, Loss = 0, 0.0
        LossList = []
        N = 0
        for data, target in self.valid_loader:
            with torch.no_grad():
                data, target = data.to(param.DEVICE), target.to(param.DEVICE)
                output = model(data)
                loss = self.criterion(output, target)
                _, pred = torch.max(output, dim=1)
                
                Loss += loss.item()
                LossList.append(loss.item())
                Acc += torch.sum(pred.eq(target)).item()
                N += len(target)

        Acc = Acc / N
        Loss = Loss / len(self.valid_loader)
        return Acc, Loss

    def send2server(self, msg):
        self.comm.send(0, msg)
        assert self.comm.recv(0) == "ACK"
        logging.debug("Client {}: send local grads to server".format(self.id))

    def receive_global_model(self):
        logging.debug("Client {}: wait for invitation ...".format(self.id))

        chose = self.comm.recv(0)
        if not chose:
            self.comm.send(0, "ACK")
            return
        
        logging.debug("Client {} : receive invitation {}".format(self.id, chose))
        self.comm.send(0, "ACK")

        # Download the global model parameters
        global_model = self.comm.recv(0)
        logging.debug("Client {}: get global weights from server: {}".format(self.id, global_model))
        return global_model

    def receive_other_parameter(self):
        record_param = self.comm.recv(0)
        logging.debug("Client {} got other clients' parameter from server".format(self.id))
        param_list = [value for idx, value in record_param]
        return param_list
    
    def generate_fit_gradients(self, global_model, rn):
        pass

    def generate_normed_gradients(self, global_model, param_list, rn):
        """
        训练 self.model 使其沿 loss 取反方向优化 并同时接近中心点  (median ) 
        最终输出相对于 global_model 的梯度
        """
        # 计算几何中位数（gradient_list 中的几何中心点）
        #param_list_cpu = []
        #for idx in range(len(param_list)):
        #    param_list_cpu.append(param_list[idx].clone().to("cpu"))
        #median = compute_geometric_median(param_list_cpu).median.to(DEVICE) #中心点
        median = compute_geometric_median(param_list).median
        reference_param = min(param_list, key=lambda g: torch.norm(g - median, p=2))
        # 允许的偏移范围是最近点到中心点之间的差值*lr，这表示两个对应模型之间的距离
        C = param.ADVERSARY_SCALE[rn] * torch.norm(reference_param - median, p=2) 

        self.unserialize_model(median)
        logging.debug("Client {} C {} parameter {}".format(self.id, C, median))

        # 训练 self.model，使其沿 loss 取反方向优化
        for i in tqdm(range(param.ADVERSARY_ITERATION), desc="client {} round {} adversarial training : ".format(self.id, rn)):
            for data, target in self.train_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)

                self.optimizer.zero_grad()
                output = self.model(data)

                # 计算 loss（取反方向优化）
                loss = -self.criterion(output, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=param.ADVERSARY_NORM)
                self.optimizer.step()
            #norm clip
            theta_adv = self.serialize_model()
            logging.debug("Client {} round {}.{} parameter {}".format(self.id, rn, i, theta_adv))
            diff = theta_adv - median
            rate = max(1.0, torch.norm(diff, p=2) / (C))
            clipped_theta = diff / rate + median
            self.unserialize_model(clipped_theta)
            logging.debug("Client {} round {}.{} rate {} parameter {}".format(self.id, rn, i, rate, clipped_theta))

        theta_fit_adv = self.serialize_model()
        return theta_fit_adv

    def generate_clip_gradients(self, global_model, param_list, rn):
        good_param_stack = torch.stack(param_list)
        mean = torch.mean(good_param_stack, dim = 0)
        self.unserialize_model(mean)
        sorted_vals, _ = torch.sort(good_param_stack, dim=0)  # shape: [num_clients, num_params]
        beta = param.TRIMMED_MEAN_BETA
        select = sorted_vals[int(beta * param.ADVERSARY_SCALE[rn]):-int(beta * param.ADVERSARY_SCALE[rn])]
        clip_mini, clip_maxi = select.min(dim=0).values, select.max(dim=0).values
        # 训练 self.model，使其沿 loss 取反方向优化
        for i in tqdm(range(param.ADVERSARY_ITERATION), desc="client {} round {} adversarial training : ".format(self.id, rn)):
            self.fit_one_epoch(rn, 1, rev=True, add_noise=False)
            theta_adv = self.serialize_model()
            theta_fit_adv = torch.clamp(theta_adv, min=clip_mini, max=clip_maxi)
            self.unserialize_model(theta_fit_adv)

        theta_fit_adv = self.serialize_model()
        return theta_fit_adv

    def evaluate(self):
        """
            Train the model on all rounds.
        """
        self.comm.initialize()
        for rn in range(self.round):
            if self.id in param.TAPPING_CLIENTS :
                global_model = self.receive_global_model()
                good_params = self.receive_other_parameter()
                if param.TAPPING_SAME and self.id != param.TAPPING_CLIENTS[0]:
                    self.send2server(global_model)
                    continue
                if param.MKRUM:
                    bad_param = self.generate_normed_gradients(global_model, good_params, rn)
                elif param.TRIMMED_MEAN:
                    bad_param = self.generate_clip_gradients(global_model, good_params, rn)
                else:
                    assert(0)
                logging.debug("Client {} bad model parameter shape : {}, other model parameter shape : {}".format(self.id, bad_param.shape, good_params[0].shape))
                self.send2server(bad_param)
            elif self.id in param.BAD_CLIENTS:
                global_model = self.receive_global_model()
                res = self.generate_bad_gradients(global_model, rn, mode=param.ATTACK_MODE)
                self.send2server(res)
            else :
                global_model = self.receive_global_model()
                res = self.fit(global_model, rn)
                self.send2server(res)