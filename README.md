# MPA to LDPFL

A general framework of model poisoning attack to locally differentially private federated learning, which might be further used for experiment relatex to federated learning.

## 0 Environment

The recommanded versions of python and pytorch packages are in the file `src/environment.yaml` and `src/requirements.txt`

## 1 Run

1. Make a new log directory and go to the dataset forder

```bash
mkdir log && cd ./src/dataset
```

2. Split the data for each client. Modify `ALPHA` and `CLIENTS_NUM` to adjust non-IID degree and $N$ before run the python file

```bash
python split_noniid.py
```

3. Modify the setting in `param.py` and run!

```bash
cd ..
python launcher.py
```

Reference settings and explanation is in the `src/param.py`

