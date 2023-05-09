import torch
from data import DataPreparation
import os
import numpy as np
import random
from sys import stderr
import datetime
import time
from utils import MetricsManagerInterface
import torch.nn
from torchvision.models import efficientnet_b0,EfficientNet_B0_Weights



class e0Model:
    def __init__(self,dataset:DataPreparation,phase) -> None:
        #self.model = None
        self.dataset = dataset
        self.phase = phase
        self.create_model()
        
    def create_model(self):
        if self.phase == 'first teacher':
            self.model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        else:
            self.model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT, stochastic_depth_prob=self.dataset.args.stoch_depth_prob)
            self.model.classifier[0] = torch.nn.Dropout(p=self.dataset.args.drop_out_rate, inplace = True)

        num_fts = self.model.classifier[1].in_features
        self.model.classifier[1] = torch.nn.Linear(in_features=num_fts, out_features=len(self.dataset.CELL_TYPES), bias=True)


class Trainer:
    def __init__(self, model_container:e0Model, iter, dataset:DataPreparation,metrics_container:MetricsManagerInterface, phase = 'student'):
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
        report = self.metrics_container.eval(y_true, y_pred,labels=self.dataset.CELL_TYPES)
        benchmark = self.metrics_container.benchmark(y_true, y_pred,labels=self.dataset.CELL_TYPES)

        # f1 = metrics.f1_score(y_true, y_pred,average="micro")
        # acc = metrics.balanced_accuracy_score(y_true, y_pred)
        # class_report = metrics.classification_report(y_true,y_pred,target_names = self.dataset.CELL_TYPES)

        #return f1,cumulative_loss,acc,class_report
        return cumulative_loss,benchmark,report

    def logging(self,mode,epoch,f1,loss,acc):
        self.log(f"Epoche {epoch} {mode}: (loss {loss:.4f}, F1 {f1:.4f}, acc {acc:.4f})")
        # mlflow.log_param("iter",self.iter)
        # mlflow.log_param("epoch", epoch)
        # mlflow.log_param("mode", mode)
        # mlflow.log_metric("loss",loss)
        # mlflow.log_metric("f1",f1)

    def generate_psudo_label(self,net,dataloader):
    
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
            self.log(f"Other Report:{report_test}")
            # mlflow.log_metric("loss",loss_test)
            # for class_name, metrics in class_report.items():
            #     for metric_name, metric_value in metrics.items():
            #         name= class_name+metric_name
            #         mlflow.log_metric(name, metric_value)

        ## generate psudo labels
        self.log("--------------------------------------------------------------------------")
        self.log("Generating psudo labels...")
        to_predict_loader = self.dataset.prepare_train_loader(psudo_labeled=True)
        #path = os.path.join(f"/home/h1/yexu660b/project/results_new/bestAfterOld/20230412-153040_new_dataset_iter_0/",f"model_best_f1.pth.tar")
        path = os.path.join(self.output_path,f"model_best_bench.pth.tar")
        self.model_container.model.load_state_dict(torch.load(path)['model_weights'])
        result = self.generate_psudo_label(self.model_container.model, to_predict_loader)
        self.dataset.append_pseudo_label(result, self.iter, self.dataset.labels_path)
        self.log(f"Generation done. See the file: {self.dataset.labels_path}_{self.iter}. Totally {self.dataset.args.num_psudo_labels} labels generated")
        del train_loader,to_predict_loader
        self.log_file.close()

    


# def forward_data(data_loader,mode,model,optimizer,criterion,device):
    
#     if mode == "train":
#         model.train()
#     elif mode in {"val","test"}:
#         model.eval()

#     cumulative_loss = 0.0

#     y_pred = None
#     y_true = None

#     for index,(ids,inputs,labels,folders) in enumerate(data_loader):
        
#         labels = labels.to(device).float()
#         inputs = inputs.to(device)

#         optimizer.zero_grad()

#         with torch.set_grad_enabled(mode=="train"):
#             outputs = model(inputs)
#             loss = criterion(outputs,labels)
            
#             if mode == "train":
#                 loss.backward()
#                 optimizer.step()
                
        
#         cumulative_loss += loss.item() * inputs.size(0)

#         if index == 0:
#             _,true = torch.max(labels,1)
#             y_true = true.detach().cpu().numpy()
#             _,pred = torch.max(outputs,1)
#             y_pred = pred.detach().cpu().numpy()
            
#         else:
#             _,true = torch.max(labels,1)
#             _,pred = torch.max(outputs,1)
#             y_true = np.concatenate((y_true,true.detach().cpu().numpy()))
#             y_pred = np.concatenate((y_pred,pred.detach().cpu().numpy()))
        
#     f1 = metrics.f1_score(y_true, y_pred,average="micro")
    
#     acc = metrics.balanced_accuracy_score(y_true, y_pred)

#     cumulative_loss /= len(data_loader.dataset)

#     class_report = metrics.classification_report(y_true,y_pred,target_names = CELL_TYPES)

#     return f1,cumulative_loss,acc,model,optimizer,class_report
    


# def logging(mode,iter,log,epoch,f1,loss,acc):
#     log(f"Epoche {epoch} {mode}: (loss {loss:.4f}, F1 {f1:.4f}, acc {acc:.4f})")
#     # mlflow.log_param("iter",iter)
#     # mlflow.log_param("epoch", epoch)
#     # mlflow.log_param("mode", mode)
#     # mlflow.log_metric("loss",loss)
#     # mlflow.log_metric("f1",f1)


# def generate_psudo_label(net, dataloader,device):
    
#     net.to(device)
#     net.eval()
#     result = {}
#     for index,(cell_id,img,label,folder) in enumerate(dataloader):
#         with torch.no_grad():
#             outputs = net(img.to(device))
#             _,pred = torch.max(outputs,1)
#             for i in range(len(cell_id)):
#             ## new data set!!!
#                 temp = []
#                 temp.append(folder[i])
#                 temp.append(pred[i].item())
#                 result[cell_id[i]]=temp
#                 #result[cell_id[i].item()]=pred[i].item()
#     return result

# def one_iteration(args, model, ITER, IMG_SIZE, test_loader, val_loader, labels_path, prepare_data, phase='student'):
    
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
#     ## make output directory
#     date_=datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#     output_path = os.path.join(args.output_folder,f"{date_}_{args.trial_name}_{ITER}")
#     os.makedirs(output_path)
    
#     ## logging in text_file
#     log_file = open(os.path.join(output_path,"log.txt"), "a")
#     def log(msg):
#         print(time.strftime("[%d.%m.%Y %H:%M:%S]: "), msg, file=stderr)
#         log_file.write(time.strftime("[%d.%m.%Y %H:%M:%S]: ") + msg + os.linesep)
#         log_file.flush()
#         os.fsync(log_file)
    
#     log('Output psudo labels directory: ' + labels_path)
#     log('Output directory: ' + output_path)

#     log("Used parameters...")
#     for arg in sorted(vars(args)):
#         log("\t" + str(arg) + " : " + str(getattr(args, arg)))
#         #mlflow.log_param(str(arg), getattr(args, arg))

#     args_dict = {}
#     for arg in vars(args):
#         args_dict[str(arg)] = getattr(args, arg)
        

#     train_loader = prepare_data.prepare_train_loader(args, IMG_SIZE, phase=phase)
#     model.to(device)
#     criterion = torch.nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(
#         filter(lambda p: p.requires_grad, model.parameters()),
#         lr=args.learning_rate,
#         weight_decay=args.weight_decay,
#     )

#     SEED=42
#     torch.manual_seed(SEED)
#     random.seed(SEED)
#     np.random.seed(SEED)

#     torch.backends.cudnn.benchmark = True
#     torch.backends.cudnn.enabled = True

#     best_f1 = 0
#     model_path = ""

#     for epoch in range(1,args.epochs+1):

#         ## training
#         f1_train,loss_train,acc_train,model,optimizer, class_report = forward_data(train_loader,"train",model,optimizer,criterion,device)
#         logging("train",ITER,log,epoch,f1_train,loss_train,acc_train)

#         with torch.no_grad():
#             #Validation
#             f1_val,loss_val,acc_val,model,optimizer, class_report = forward_data(val_loader,"val",model,optimizer,criterion,device)
#             logging("val",ITER,log,epoch,f1_val,loss_val,acc_val)
        
#         # save Checkpoint
#         if epoch % 10 == 0 or epoch == args.epochs:
#             current_state = {'epoch': epoch,
#                                  'model_weights': model.state_dict(),
#                                  'optimizer': optimizer.state_dict(),
#                                  'args': args_dict,
#                                  }

#             model_path = os.path.join(output_path,f"checkpoint.pth.tar")
#             torch.save(current_state,model_path)
#             log(f"Saved checkpoint to: {model_path}")
        
#         ### save best f1 model on validation set
#         if epoch  > 10 and f1_val > best_f1:
            
#             best_f1 = f1_val

#             current_state = {'epoch': epoch,
#                                  'model_weights': model.state_dict(),
#                                  'optimizer': optimizer.state_dict(),
#                                  'args': args_dict,
#                                  'f1_val': f1_val,
#                                  }
#             model_path = os.path.join(output_path,f"model_best_f1.pth.tar")
#             torch.save(current_state,model_path)
#             log(f"Saved Model with best Validation F1 to: {model_path}")
            
#     ## test model
#     log("--------------------------------------------------------------------------")
#     log("Testing best Validation Model...")
#     with torch.no_grad():
#         path = os.path.join(output_path,f"model_best_f1.pth.tar")
#         log(f"Loading best f1 model through the path: {path}")
#         model.load_state_dict(torch.load(path)['model_weights'])
#         f1_test,loss_test,acc_test,model,optimizer,class_report = forward_data(test_loader,"test",model,optimizer,criterion,device)
#         log(f"Test: (loss {loss_test:.4f}, F1 {f1_test:.4f}, acc {acc_test:.4f})")
#         log(f"Classification Report:{class_report}")
#         # mlflow.log_metric("loss",loss_test)
#         # for class_name, metrics in class_report.items():
#         #     for metric_name, metric_value in metrics.items():
#         #         name= class_name+metric_name
#         #         mlflow.log_metric(name, metric_value)

#     ## generate psudo labels
#     log("--------------------------------------------------------------------------")
#     log("Generating psudo labels...")
#     to_predict_loader = prepare_data.prepare_train_loader(args, IMG_SIZE, psudo_labeled=True)
#     #path = os.path.join(f"/home/h1/yexu660b/project/results_new/bestAfterOld/20230412-153040_new_dataset_iter_0/",f"model_best_f1.pth.tar")
#     path = os.path.join(output_path,f"model_best_f1.pth.tar")
#     model.load_state_dict(torch.load(path)['model_weights'])
#     result = generate_psudo_label(model, to_predict_loader,device)
#     prepare_data.append_pseudo_label(result, ITER, labels_path)
#     log(f"Generation done. See the file: {labels_path}_{ITER}. Totally {args.num_psudo_labels} labels generated")

#     del train_loader,to_predict_loader, model

#     log_file.close()