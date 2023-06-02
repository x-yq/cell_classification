import os
from train import Trainer
from data import DataPreparation
from hyper import parser
from utils import ENModelContainer
from utils import myMetrics

def main(parser):
    performance_metrics = myMetrics()
    prepare_data = DataPreparation(parser)
    num_labeled_train = 35977
    
    #mlflow.set_tracking_uri("http://mlflow.172.26.62.216.nip.io")
    #mlflow.start_run()
    
    ## First teacher  
    prepare_data.update_args(num_pseudo_labels=num_labeled_train)
    teacher = ENModelContainer(prepare_data)
    teacher.create_first_teacher()
    teacher_trainer = Trainer(teacher,"iter_0",prepare_data, performance_metrics, phase='first teacher')
    teacher_trainer.one_iteration()

    ## Iteration 1
    prepare_data.update_args(pseudo_labels_csv=os.path.join(prepare_data.labels_path,"iter_0.csv"),num_pseudo_labels=int(num_labeled_train*3.1))
    student_1 = ENModelContainer(prepare_data)
    student_1.create_student()
    student_trainer_1 = Trainer(student_1,"iter_1",prepare_data,performance_metrics)
    student_trainer_1.one_iteration()
    
    
    ## Iteration 2
    prepare_data.update_args(pseudo_labels_csv=os.path.join(prepare_data.labels_path,"iter_1.csv"))
    student_2 = ENModelContainer(prepare_data)
    student_2.create_student()
    student_trainer_2 = Trainer(student_2,"iter_2",prepare_data,performance_metrics)
    student_trainer_2.one_iteration()

    #mlflow.end_run()

if __name__ == "__main__":
    main(parser)