import torch
from data import DataPreparation
import torchvision
from abc import ABC, abstractmethod

# class e0Model:
#     def __init__(self,dataset:DataPreparation,phase) -> None:
#         #self.model = None
#         self.dataset = dataset
#         self.phase = phase
#         self.create_model()
        
#     def create_model(self):
#         if self.phase == 'first teacher':
#             self.model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
#         else:
#             self.model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT, stochastic_depth_prob=self.dataset.args.stoch_depth_prob)
#             self.model.classifier[0] = torch.nn.Dropout(p=self.dataset.args.drop_out_rate, inplace = True)

#         num_fts = self.model.classifier[1].in_features
#         self.model.classifier[1] = torch.nn.Linear(in_features=num_fts, out_features=len(self.dataset.CELL_TYPES), bias=True)


class ModelContainer:
    def __init__(self, dataset:DataPreparation) -> None:
        self.dataset = dataset

    def create_first_teacher(self):
        self.model = torchvision.models.get_model(self.dataset.args.model_name,weights="DEFAULT")
        #self.model = torchvision.models.__dict__[self.dataset.args.model_name](pretrained=True)
        #weights = torchvision.models.get_weight
        num_fts = self.model.classifier[1].in_features
        self.model.classifier[1] = torch.nn.Linear(in_features=num_fts, out_features=len(self.dataset.CELL_TYPES), bias=True)

    def create_student(self):
        self.model = torchvision.models.get_model(self.dataset.args.model_name,weights="DEFAULT",stochastic_depth_prob=self.dataset.args.stoch_depth_prob)
        #self.model = torchvision.models.__dict__[self.dataset.args.model_name](pretrained=True, stochastic_depth_prob=self.dataset.args.stoch_depth_prob)
        self.model.classifier[0] = torch.nn.Dropout(p=self.dataset.args.drop_out_rate, inplace = True)
        num_fts = self.model.classifier[1].in_features
        self.model.classifier[1] = torch.nn.Linear(in_features=num_fts, out_features=len(self.dataset.CELL_TYPES), bias=True)



class MetricsManagerInterface(ABC):

    @abstractmethod
    def __init__(self) -> None:
        pass


    @abstractmethod
    def evaluate(self, y_true, y_pred, labels=[]) -> dict:
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


