import torch
import torchvision
import data
from data import CELL_TYPES
import os
import numpy as np
import random
from sys import stderr
from hyper import parser
import datetime
import time
import sklearn.metrics as metrics
import torch.nn
# import mlflow
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


def forward_data(data_loader,mode,model,optimizer,criterion,device):

    assert(mode in {"train","val","test"})
    
    if mode == "train":
        model.train()
    elif mode in {"val","test"}:
        model.eval()

    cumulative_loss = 0.0

    y_pred = None
    y_true = None

    for index,(ids,inputs,labels) in enumerate(data_loader):
        
        # Forward Pass
        labels = labels.to(device).float()
        inputs = inputs.to(device)

        optimizer.zero_grad()

        with torch.set_grad_enabled(mode=="train"):
            outputs = model(inputs)
            loss = criterion(outputs,labels)
            
            if mode == "train":
                loss.backward()
                optimizer.step()
                
        
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
        
    f1 = metrics.f1_score(y_true, y_pred,average="micro")

    cumulative_loss /= len(data_loader.dataset)

    class_report = metrics.classification_report(y_true,y_pred,target_names = CELL_TYPES)

    return f1,cumulative_loss,model,optimizer,class_report


def logging(mode,iter,log,epoch,f1,loss):
    log(f"Epoche {epoch} {mode}: (loss {loss:.4f}, F1 {f1:.4f})")
    # mlflow.log_param("iter",iter)
    # mlflow.log_param("epoch", epoch)
    # mlflow.log_param("mode", mode)
    # mlflow.log_metric("loss",loss)
    # mlflow.log_metric("f1",f1)


def generate_psudo_label(net, dataloader,device):
    
    net.to(device)
    net.eval()
    result = {}
    for index,(cell_id,img,label) in enumerate(dataloader):
        with torch.no_grad():
            outputs = net(img.to(device))
            #TODO: debug
            _,pred = torch.max(outputs,1)
            for i in range(len(cell_id)):
                result[cell_id[i].item()]=pred[i].item()
    return result

def one_iteration(args, model, ITER, IMG_SIZE, test_loader, val_loader,phase='student'):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    ## make output directory
    date_=datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = os.path.join(args.output_folder,f"{date_}_{args.trial_name}_{ITER}")
    os.makedirs(output_path)
    
    ## logging in text_file
    log_file = open(os.path.join(output_path,"log.txt"), "a")
    def log(msg):
        print(time.strftime("[%d.%m.%Y %H:%M:%S]: "), msg, file=stderr)
        log_file.write(time.strftime("[%d.%m.%Y %H:%M:%S]: ") + msg + os.linesep)
        log_file.flush()
        os.fsync(log_file)
    
    log('Output directory: ' + output_path)

    log("Used parameters...")
    for arg in sorted(vars(args)):
        log("\t" + str(arg) + " : " + str(getattr(args, arg)))
        #mlflow.log_param(str(arg), getattr(args, arg))

    args_dict = {}
    for arg in vars(args):
        args_dict[str(arg)] = getattr(args, arg)
        

    ## DataLoader
    train_loader = data.prepare_train_loader(args, IMG_SIZE, phase=phase)

    ## Model
    model.to(device)
    

    ## Loss function
    criterion = torch.nn.CrossEntropyLoss()
    
    ## optimizer
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    ## seeding
    SEED=42
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    best_f1 = 0
    model_path = ""

    for epoch in range(1,args.epochs+1):

        ## training
        f1_train,loss_train,model,optimizer, class_report = forward_data(train_loader,"train",model,optimizer,criterion,device)
        logging("train",ITER,log,epoch,f1_train,loss_train)

        with torch.no_grad():
            ##Validation
            f1_val,loss_val,model,optimizer, class_report = forward_data(val_loader,"val",model,optimizer,criterion,device)
            logging("val",ITER,log,epoch,f1_val,loss_val)
        
        ## save Checkpoint
        if epoch % 10 == 0 or epoch == args.epochs:
            current_state = {'epoch': epoch,
                                 'model_weights': model.state_dict(),
                                 'optimizer': optimizer.state_dict(),
                                 'args': args_dict,
                                 }

            model_path = os.path.join(output_path,f"checkpoint.pth.tar")
            torch.save(current_state,model_path)
            log(f"Saved checkpoint to: {model_path}")
        

        ### save best model on validation set
        if epoch  > 10 and f1_val > best_f1:
            
            best_f1 = f1_val

            current_state = {'epoch': epoch,
                                 'model_weights': model.state_dict(),
                                 'optimizer': optimizer.state_dict(),
                                 'args': args_dict,
                                 'f1_val': f1_val,
                                 }
            model_path = os.path.join(output_path,f"model_best_f1.pth.tar")
            torch.save(current_state,model_path)
            log(f"Saved Model with best Validation F1 to: {model_path}")
            
    ## test model
    log("Testing best Validation Model")
    with torch.no_grad():
        model.load_state_dict(torch.load(model_path)['model_weights'])
        f1_test,loss_test,model,optimizer,class_report = forward_data(test_loader,"test",model,optimizer,criterion,device)
        log(f"Test: (loss {loss_test:.4f}, F1 {f1_test:.4f})")
        log(f"Classification Report:{class_report}")
        # mlflow.log_metric("loss",loss_test)
        # for class_name, metrics in class_report.items():
        #     for metric_name, metric_value in metrics.items():
        #         name= class_name+metric_name
        #         mlflow.log_metric(name, metric_value)

    #generate psudo labels
    log("Generating psudo labels")
    to_predict_loader = data.prepare_train_loader(args, IMG_SIZE, psudo_labeled=True)
    model.load_state_dict(torch.load(model_path)['model_weights'])
    result = generate_psudo_label(model, to_predict_loader,device)
    data.append_psudo_label(result, ITER)
    log(f"Generation done. See the file: {ITER}")

    del train_loader,to_predict_loader, model

    log_file.close()

def main(args):
    num_classes=18
    weight = EfficientNet_B0_Weights.DEFAULT
    
    #mlflow.set_tracking_uri("http://mlflow.172.26.62.216.nip.io")
    #mlflow.start_run()
    
    test_loader,val_loader = data.prepare_test_val_loader(args,args.img_size)
    
    teacher_model = efficientnet_b0(weights=weight)
    num_fts = teacher_model.classifier[1].in_features
    teacher_model.classifier[1] = torch.nn.Linear(in_features=num_fts, out_features=num_classes,bias=True)
    one_iteration(args, teacher_model,"iter_1",args.img_size, test_loader,val_loader, phase="first teacher")

    parser.set_defaults(psudo_labels_csv="iter_1.csv")
    parser.set_defaults(num_psudo_labels=30000)

    student_model_1 = efficientnet_b0(weights=weight, stochastic_depth_prob=args.stoch_depth_prob)
    num_fts = student_model_1.classifier[1].in_features
    student_model_1.classifier[0] = torch.nn.Dropout(p=args.drop_out_prob, inplace = True)
    student_model_1.classifier[1] = torch.nn.Linear(in_features=num_fts, out_features=num_classes, bias=True)
    one_iteration(args, student_model_1,"iter_2",args.img_size, test_loader,val_loader)
    
    parser.set_defaults(psudo_labels_csv="iter_2.csv")
    
    student_model_2 = efficientnet_b0(weights=weight, stochastic_depth_prob=args.stoch_depth_prob)
    num_fts = student_model_2.classifier[1].in_features
    student_model_2.classifier[0] = torch.nn.Dropout(p=args.drop_out_prob, inplace = True)
    student_model_2.classifier[1] = torch.nn.Linear(in_features=num_fts, out_features=num_classes, bias=True)
    one_iteration(args, student_model_2,"iter_3",args.img_size, test_loader,val_loader)

    #mlflow.end_run()

        

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)