import torch
from data import DataPreparation
import os
import numpy as np
import random
from sys import stderr
import sklearn.metrics as metrics
import torch.nn as nn
import torchvision
from abc import ABC, abstractmethod

class ModelContainerInterface(ABC):
    @abstractmethod
    def __init__(self,model_name,dataset:DataPreparation) -> None:
        self.model_name = model_name
        self.dataset = dataset


    def create_first_teacher(self):
        self.model = torchvision.models.get_model(self.model_name)

    def create_student(self):
        pass

class ModelContainer:
    def __init__(self, dataset:DataPreparation) -> None:
        self.dataset = dataset
        self.model = torchvision.models.detection.__dict__[self.dataset.args.model_name](pretrained=True)

    def create_first_teacher(self):
        self.model = torchvision.models.get_model(self.model_name)


class MetricsManagerInterface(ABC):

    @abstractmethod
    def __init__(self) -> None:
        pass


    @abstractmethod
    def eval(self, y_true, y_pred, labels=[]) -> dict:
        pass

    @abstractmethod
    def benchmark(self, y_true, y_pred, labels=[]) -> float:
        pass

    # def eval(self, y_true, y_pred, label=[]):
    #     result = {}
    #     for metric in self.metrics_list:
    #         print(metric.__name__)
    #         name = metric.__name__

    #         if metric.__name__ == "f1_score":
    #             result[metric.__name__] = metric(y_true,y_pred,labels=label, average='micro')
    #         else:
    #             result[metric.__name__] = metric(y_true,y_pred)
    #     best = self.best_model_benchmark(y_true, y_pred)

    #     return best, result

class myMetric(MetricsManagerInterface):
    def __init__(self) -> None:
        super().__init__()

    def benchmark(self, y_true, y_pred, labels=[]):
        pa=metrics.accuracy_score(y_true,y_pred)
        p2=metrics.accuracy_score(y_true,y_pred)
        return pa,p2

    
    def eval(self, y_true, y_pred, labels=[]):
        f1=metrics.f1_score(y_true,y_pred,labels=labels,average='micro')
        prc=metrics.recall_score(y_true,y_pred,labels=labels,average = 'micro')
        result = {'f1 score':f1, 'prc':prc}
        return result


y_true = [1,1,2,0,3]
y_pred = [3,1,2,2,3]
label = ['a','b','c','d']
manager = myMetric()
r = manager.eval(torch.Tensor(y_true), torch.Tensor(y_pred))
b = manager.benchmark(torch.Tensor(y_true), torch.Tensor(y_pred))
print(b,r)
