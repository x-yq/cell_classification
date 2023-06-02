import torch
from data import DataPreparation
import os
import numpy as np
import random
from sys import stderr
import datetime
import time
from utils import MetricsManagerInterface, ModelContainerInterface
import torch.nn
  

class Trainer:
    def __init__(self, model_container:ModelContainerInterface, iter, dataset:DataPreparation,metrics_container:MetricsManagerInterface, phase = 'student'):
        self.model_container = model_container
        self.iter = iter
        self.dataset = dataset
        self.metrics_container = metrics_container
        self.phase = phase

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, self.model_container.model.parameters()),
        lr=self.dataset.args.learning_rate,
        weight_decay=self.dataset.args.weight_decay,
        )
        self.model_container.model.to(self.device)

        ## make output directory
        date_=datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.output_path = os.path.join(self.dataset.args.output_folder,f"{date_}_{self.dataset.args.trial_name}_{self.iter}")
        os.makedirs(self.output_path)

        ## logging in text_file
        self.log_file = open(os.path.join(self.output_path,"log.txt"), "a")
        
        self.log('Output psudo labels directory: ' + self.dataset.labels_path)
        self.log('Output directory: ' + self.output_path)

        self.log("Used parameters...")
        for arg in sorted(vars(self.dataset.args)):
            self.log("\t" + str(arg) + " : " + str(getattr(self.dataset.args, arg)))
            #mlflow.log_param(str(arg), getattr(args, arg))

        self.args_dict = {}
        for arg in vars(self.dataset.args):
            self.args_dict[str(arg)] = getattr(self.dataset.args, arg)
    
    def log(self,msg):
        print(time.strftime("[%d.%m.%Y %H:%M:%S]: "), msg, file=stderr)
        self.log_file.write(time.strftime("[%d.%m.%Y %H:%M:%S]: ") + msg + os.linesep)
        self.log_file.flush()
        os.fsync(self.log_file)

    def forward_data(self,mode,data_loader):
        if mode == "train":
            self.model_container.model.train()
        elif mode in {"val","test"}:
            self.model_container.model.eval()

        cumulative_loss = 0.0

        y_pred = None
        y_true = None

        for index,(ids,inputs,labels,folders) in enumerate(data_loader):
            
            labels = labels.to(self.device).float()
            inputs = inputs.to(self.device)

            self.optimizer.zero_grad()

            with torch.set_grad_enabled(mode=="train"):
                outputs = self.model_container.model(inputs)
                loss = self.criterion(outputs,labels)
                
                if mode == "train":
                    loss.backward()
                    self.optimizer.step()
                    
            
            cumulative_loss += loss.item() * inputs.size(0)

            if index == 0:
                _,true = torch.max(labels,1)
                y_true = true.detach().cpu().numpy()
                _,pred = torch.max(outputs,1)
                y_pred = pred.detach().cpu().numpy()
                
            else:
                _,true = torch.max(labels,1)
                _,pred = torch.max(outputs,1)
                y_true = np.concatenate((y_true,true.detach().cpu().numpy()))
                y_pred = np.concatenate((y_pred,pred.detach().cpu().numpy()))

        cumulative_loss /= len(data_loader.dataset)
        report = self.metrics_container.evaluate(y_true, y_pred,labels=self.dataset.CELL_TYPES)
        benchmark = self.metrics_container.benchmark(y_true, y_pred,labels=self.dataset.CELL_TYPES)

        # f1 = metrics.f1_score(y_true, y_pred,average="micro")
        # acc = metrics.balanced_accuracy_score(y_true, y_pred)
        # class_report = metrics.classification_report(y_true,y_pred,target_names = self.dataset.CELL_TYPES)

        #return f1,cumulative_loss,acc,class_report
        return cumulative_loss,benchmark,report

    def logging(self,mode,epoch,f1,loss):
        self.log(f"Epoche {epoch} {mode}: (loss {loss:.4f}, F1 {f1:.4f})")
        # mlflow.log_param("iter",self.iter)
        # mlflow.log_param("epoch", epoch)
        # mlflow.log_param("mode", mode)
        # mlflow.log_metric("loss",loss)
        # mlflow.log_metric("f1",f1)

    def generate_pseudo_label(self,net,dataloader):
    
        net.to(self.device)
        net.eval()
        result = {}
        for index,(cell_id,img,label,folder) in enumerate(dataloader):
            with torch.no_grad():
                outputs = net(img.to(self.device))
                _,pred = torch.max(outputs,1)
                for i in range(len(cell_id)):
                ## new data set!!!
                    temp = []
                    temp.append(folder[i])
                    temp.append(pred[i].item())
                    result[cell_id[i]]=temp
                    #result[cell_id[i].item()]=pred[i].item()
        return result

    def one_iteration(self):

        train_loader = self.dataset.prepare_train_loader(phase=self.phase)

        SEED=42
        torch.manual_seed(SEED)
        random.seed(SEED)
        np.random.seed(SEED)

        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

        best_bench = 0
        model_path = ""

        for epoch in range(1,self.dataset.args.epochs+1):

            ## training
            loss_train,bench_train,report_train = self.forward_data("train",train_loader)
            self.logging("train",epoch,bench_train,loss_train)

            with torch.no_grad():
                #Validation
                loss_val,bench_val,report_val = self.forward_data("val",self.dataset.val_loader)
                self.logging("val",epoch,bench_val,loss_val)
            
            # save Checkpoint
            if epoch % 2 == 0 or epoch == self.dataset.args.epochs:
                current_state = {'epoch': epoch,
                                    'model_weights': self.model_container.model.state_dict(),
                                    'optimizer': self.optimizer.state_dict(),
                                    'args': self.args_dict,
                                    }

                model_path = os.path.join(self.output_path,f"checkpoint.pth.tar")
                torch.save(current_state,model_path)
                self.log(f"Saved checkpoint to: {model_path}")
            
            ### save best benchmark model on validation set
            if epoch  > 2 and bench_val > best_bench:
                
                best_bench = bench_val

                current_state = {'epoch': epoch,
                                    'model_weights': self.model_container.model.state_dict(),
                                    'optimizer': self.optimizer.state_dict(),
                                    'args': self.args_dict,
                                    'bench_val': bench_val,
                                    }
                model_path = os.path.join(self.output_path,f"model_best_bench.pth.tar")
                torch.save(current_state,model_path)
                self.log(f"Saved Model with best Validation benchmark to: {model_path}")
                
        ## test model
        self.log("--------------------------------------------------------------------------")
        self.log("Testing best Validation Model...")
        with torch.no_grad():
            path = os.path.join(self.output_path,f"model_best_bench.pth.tar")
            self.log(f"Loading best benchmark model through the path: {path}")
            self.model_container.model.load_state_dict(torch.load(path)['model_weights'])
            loss_test,bench_test,report_test = self.forward_data("test",self.dataset.test_loader)
            self.log(f"Test: (loss {loss_test:.4f}, Benchmark {bench_test:.4f})")
            for key, value in report_test.items():
                self.log(f"{key}: {value}")
            
            #self.log(f"Other Report:{report_test}")
            # mlflow.log_metric("loss",loss_test)
            # for class_name, metrics in class_report.items():
            #     for metric_name, metric_value in metrics.items():
            #         name= class_name+metric_name
            #         mlflow.log_metric(name, metric_value)

        ## generate pseudo labels
        self.log("--------------------------------------------------------------------------")
        self.log("Generating pseudo labels...")
        to_predict_loader = self.dataset.prepare_train_loader(pseudo_labeled=True)
        #path = os.path.join(f"/home/h1/yexu660b/project/results_new/bestAfterOld/20230412-153040_new_dataset_iter_0/",f"model_best_f1.pth.tar")
        path = os.path.join(self.output_path,f"model_best_bench.pth.tar")
        self.model_container.model.load_state_dict(torch.load(path)['model_weights'])
        result = self.generate_pseudo_label(self.model_container.model, to_predict_loader)
        self.dataset.append_pseudo_label(result, self.iter)
        self.log(f"Generation done. See the file: {self.dataset.labels_path}_{self.iter}. Totally {self.dataset.args.num_pseudo_labels} labels generated")
        del train_loader,to_predict_loader
        self.log_file.close()
