import os
import cell_classification as cc
import sklearn.metrics as metrics
from utils import MetricsManagerInterface

class myMetric(MetricsManagerInterface):
    def __init__(self) -> None:
        super().__init__()

    def benchmark(self, y_true, y_pred, labels=[]):
        pa=metrics.accuracy_score(y_true,y_pred)
        p2=metrics.accuracy_score(y_true,y_pred)
        return pa,p2

    
    def eval(self, y_true, y_pred, labels=[]):
        f1=metrics.f1_score(y_true,y_pred,labels=labels,average='micro')
        classification_report = metrics.classification_report(y_true,y_pred,target_names=labels)
        prc=metrics.recall_score(y_true,y_pred,labels=labels,average = 'micro')
        result = {'f1 score':f1, 'prc':prc}
        return result

def main(parser):
    
    prepare_data = cc.DataPreparation(parser)
    num_labeled_train = 35977
    
    #mlflow.set_tracking_uri("http://mlflow.172.26.62.216.nip.io")
    #mlflow.start_run()
    
    ## First teacher  
    prepare_data.update_args(num_psudo_labels=num_labeled_train)
    teacher = cc.e0Model(prepare_data,'first teacher')
    teacher_trainer = cc.Trainer(teacher,"iter_0",prepare_data,phase='first teacher')
    teacher_trainer.one_iteration()

    ## Iteration 1
    prepare_data.update_args(psudo_labels_csv=os.path.join(prepare_data.labels_path,"iter_0.csv"),num_psudo_labels=int(num_labeled_train*3.1))
    student_1 = cc.e0Model(prepare_data,'student 1')
    student_trainer_1 = cc.Trainer(student_1,"iter_1",prepare_data)
    student_trainer_1.one_iteration()
    
    
    ## Iteration 2
    prepare_data.update_args(psudo_labels_csv=os.path.join(prepare_data.labels_path,"iter_1.csv"))
    student_2 = cc.e0Model(prepare_data,'student 2')
    student_trainer_2 = cc.Trainer(student_2,"iter_2",prepare_data)
    student_trainer_2.one_iteration()

    #mlflow.end_run()

if __name__ == "__main__":
    main(cc.parser)