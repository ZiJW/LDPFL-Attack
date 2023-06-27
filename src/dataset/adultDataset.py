import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import pandas as pd

class Adult_dataset(Dataset):
    def __init__(
        self,
        root: str,
        train: bool = True
    ) -> None:
        super().__init__()
        self.root = root
        self.train = train
        self.map_dict = self.init_dict()
        self.x, self.y = self.load_data()

    def load_data(self):
        file_path = "adult/raw/adult.%s"%("data" if self.train else "test")
        df = pd.read_csv(file_path)
        df.drop("fnlwgt", axis=1, inplace=True)
        # for attr_name in df.columns:
        #     vals = df[attr_name].unique()
        #     print("df[%s].vals = %s" % (attr_name, vals))
        for attr_name in self.map_dict:
            df[attr_name] = df[attr_name].map(self.map_dict[attr_name])
        # for attr_name in df.columns:
        #     vals = df[attr_name].unique()
        #     print("df[%s].vals = %s" % (attr_name, vals))
        #     print("df[%s].max = %s" % (attr_name, df[attr_name].max()))
        #     print("df[%s].min = %s" % (attr_name, df[attr_name].min()))
        # print("train" if self.train else "test")
        # df.info()
        X, Y = df.drop("income", axis=1), df["income"]
        # return X, Y
        return torch.tensor(X.to_numpy(), dtype=torch.float), torch.tensor(Y.to_numpy())

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def init_dict(self):
        return {
            "education":{
                "HS-grad":0,
                "Some-college":1,
                "Bachelors":2,
                "Masters":3,
                "Assoc-voc":4,
                "11th":5,
                "Assoc-acdm":6,
                "10th":7,
                "7th-8th":8,
                "Prof-school":9,
                "9th":10,
                "12th":11,
                "Doctorate":12,
                "5th-6th":13,
                "1st-4th":14,
                "Preschool":15,
            },
            "workclass":{
                "Private":0,
                "Self-emp-not-inc":1,
                "Local-gov":2,
                "?":3,
                "State-gov":4,
                "Self-emp-inc":5,
                "Federal-gov":6,
                "Without-pay":7,
                "Never-worked":8,
            },
            "marital.status":{
                "Married-civ-spouse":0,
                "Never-married":1,
                "Divorced":2,
                "Separated":3,
                "Widowed":4,
                "Married-spouse-absent":5,
                'Married-AF-spouse':6,
            },
            "occupation":{
                "Prof-specialty":0,
                "Craft-repair":1,
                "Exec-managerial":2,
                "Adm-clerical":3,
                "Sales":4,
                "Other-service":5,
                "Machine-op-inspct":6,
                "?":7,
                "Transport-moving":8,
                "Handlers-cleaners":9,
                "Farming-fishing":10,
                "Tech-support":11,
                "Protective-serv":12,
                "Priv-house-serv":13,
                "Armed-Forces":14,
            },
            "relationship":{
                "Husband":0,
                "Not-in-family":1,
                "Own-child":2,
                "Unmarried":3,
                "Wife":4,
                "Other-relative":5,
            },
            "race":{
                "White":0,
                "Black":1,
                "Asian-Pac-Islander":2,
                "Amer-Indian-Eskimo":3,
                "Other":4,
            },
            "sex":{
                "Male":0,
                "Female":1,
            },
            "native.country":{
                "United-States":0,
                "Mexico":1,
                "?":2,
                "Philippines":3,
                "Germany":4,
                "Canada":5,
                "Puerto-Rico":6,
                "El-Salvador":7,
                "India":8,
                "Cuba":9,
                "England":10,
                "Jamaica":11,
                "South":12,
                "China":13,
                "Italy":14,
                "Dominican-Republic":15,
                "Vietnam":16,
                "Guatemala":17,
                "Japan":18,
                "Poland":19,
                "Columbia":20,
                "Taiwan":21,
                "Haiti":22,
                "Iran":23,
                "Portugal":24,
                "Nicaragua":25,
                "Peru":26,
                "France":27,
                "Greece":28,
                "Ecuador":29,
                "Ireland":30,
                "Hong":31,
                "Cambodia":32,
                "Trinadad&Tobago":33,
                "Laos":34,
                "Thailand":35,
                "Yugoslavia":36,
                "Outlying-US(Guam-USVI-etc)":37,
                "Honduras":38,
                "Hungary":39,
                "Scotland":40,
                "Holand-Netherlands":41,
            },
            "income":{
                "<=50K":0,
                ">50K":1,
            }
        }