import itertools
import re
import subprocess
import time

# param.py 文件路径
param_file = "param.py"

# 参数及其可能取值
mkrum_values = [True, False]
adversary_iterations = [20]
models = ["MLP", "VGG_Mini"]
bad_clients_list = [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4, 5]]

# 生成所有参数组合
param_combinations = list(itertools.product(mkrum_values, adversary_iterations, models, bad_clients_list))

def modify_params(mkrum, adv_iter, model, bad_clients):
    """修改 param.py 中的指定参数"""
    with open(param_file, "r") as file:
        content = file.read()

    # 修改对应参数
    content = re.sub(r"MKRUM\s*=\s*[^\n]+", f"MKRUM = {mkrum}", content)
    content = re.sub(r"ADVERSARY_ITERATION\s*=\s*[^\n]+", f"ADVERSARY_ITERATION = {adv_iter}", content)
    content = re.sub(r'MODEL\s*=\s*["\'][^"\']+["\']', f'MODEL = "{model}"', content)
    content = re.sub(r"BAD_CLIENTS\s*=\s*\[[^\]]*\]", f"BAD_CLIENTS = {bad_clients}", content)

    # 写回 param.py
    with open(param_file, "w") as file:
        file.write(content)

# 遍历所有参数组合并执行 launcher.py
for i, (mkrum, adv_iter, model, bad_clients) in enumerate(param_combinations):
    print(f"Running experiment {i+1}/{len(param_combinations)}:")
    print(f"MKRUM = {mkrum}, ADVERSARY_ITERATION = {adv_iter}, MODEL = {model}, BAD_CLIENTS = {bad_clients}")

    # 修改参数
    modify_params(mkrum, adv_iter, model, bad_clients)

    # 运行 launcher.py
    result = subprocess.run(["python", "launcher.py"], capture_output=True, text=True)

    print("Waiting for 20 seconds before the next experiment...")
    time.sleep(20)

