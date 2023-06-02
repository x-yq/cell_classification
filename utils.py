import torch
from data import DataPreparation
import torchvision
from abc import ABC, abstractmethod
from sklearn import metrics

class ModelContainerInterface(ABC):
    @abstractmethod
    def __init__(self, dataset:DataPreparation) -> None:
        pass

    @abstractmethod
    def create_first_teacher(self):
        pass

    @abstractmethod
    def create_student(self):
        pass

class ENModelContainer(ModelContainerInterface):
    def __init__(self, dataset:DataPreparation) -> None:
        self.dataset = dataset

    def create_first_teacher(self):
        self.model = torchvision.models.get_model(self.dataset.args.model_name,weights="DEFAULT")
        num_fts = self.model.classifier[1].in_features
        self.model.classifier[1] = torch.nn.Linear(in_features=num_fts, out_features=len(self.dataset.CELL_TYPES), bias=True)

    def create_student(self):
        self.model = torchvision.models.get_model(self.dataset.args.model_name,weights="DEFAULT",
                                                  stochastic_depth_prob=self.dataset.args.stoch_depth_prob)
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

class myMetrics(MetricsManagerInterface):
    def __init__(self) -> None:
        super().__init__()

    def benchmark(self, y_true, y_pred, labels=[]):
        pa=metrics.f1_score(y_true,y_pred,average='micro')
        return pa

    def evaluate(self, y_true, y_pred, labels=[]):
        f1=metrics.f1_score(y_true,y_pred,average='macro')
        class_report = metrics.classification_report(y_true,y_pred,target_names=labels)

        result = {'f1 score':f1, 'classification report':class_report}
        return result
