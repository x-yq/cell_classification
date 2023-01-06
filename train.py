import torch
import data
from data import CELL_TYPES
import os
import numpy as np
import random
from sys import stderr
from hyper import parser
import datetime
import copy
import time
import sklearn.metrics as metrics
import torch.nn
from torchvision.models import efficientnet_b0,efficientnet_b7
from torch.utils.tensorboard import SummaryWriter


def forward_data(data_loader,mode,model,optimizer,criterion,device):

    assert(mode in {"train","val","test"})
    
    if mode == "train":
        model.train()
    elif mode in {"val","test"}:
        model.eval()

    cumulative_loss = 0

    y_pred = None
    y_true = None


    for index,(ids,images,labels_cpu) in enumerate(data_loader):
        
        # Forward Pass
        outputs = model(images.to(device))

        loss = criterion(outputs, labels_cpu.to(device).float())
        cumulative_loss += loss.item() * labels_cpu.size(0)


        if mode == "train":
            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        #TODO: y_ored debug
        if index == 0:
            y_true = labels_cpu.numpy()
            y_pred = torch.nn.functional.softmax(outputs, dim=1).detach().cpu().numpy()
        else:
            y_true = np.concatenate((y_true,labels_cpu.numpy()))
            y_pred = np.concatenate((y_pred,torch.nn.functional.softmax(outputs, dim=1).detach().cpu().numpy()))

    #debug
    f1 = metrics.f1_score(y_true, y_pred)

    cumulative_loss /= len(data_loader.dataset)

    class_report = metrics.classification_report(y_true,y_pred, target_names = CELL_TYPES)

    return f1,cumulative_loss,model,optimizer,class_report


def logging(mode,writer,log,epoch,f1,loss):
    log(f"Epoche {epoch} {mode}: (loss {loss:.4f}, F1 {f1:.4f})")

    if writer:
        writer.add_scalar(f"{mode}/Loss", loss, epoch)
        writer.add_scalar(f"{mode}/F1", f1, epoch)
    

def generate_psudo_label(net, dataloader,device):
    
    net.eval()
    result = {}
    for cell_id,img,label in dataloader:
        with torch.no_grad():
            #TODO: debug
            output = torch.nn.functional.softmax(net([img.to(device)])).tolist()
            # cell_anno = torch.round(torch.nn.ReLU()(output)).detach().cpu().numpy()
            print(output)
            result[cell_id] = output
    return result

def one_iteration(args, model, ITER, IMG_SIZE, phase='student'):

    NUM_TYPES = 13
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

    args_dict = {}
    for arg in vars(args):
        args_dict[str(arg)] = getattr(args, arg)
        
    ## tensorboard writer 
    writer = SummaryWriter(log_dir=output_path)
    writer.add_text("Args", str(args_dict), global_step=0)

    ## DataLoader
    test_loader,val_loader = data.prepare_test_val_loader(args,IMG_SIZE)
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
        f1_train,loss_train,model,optimizer = forward_data(train_loader,"train",model,optimizer,criterion,device)
        logging("train",ITER,writer,log,epoch,f1_train,loss_train)

        with torch.no_grad():
            ##Validation
            f1_val,loss_val,model,optimizer = forward_data(val_loader,"val",model,optimizer,criterion,device)
            logging("val",ITER,writer,log,epoch,f1_val,loss_val)
        
        ## save Checkpoint
        if epoch % 10 == 0 or epoch == args.epochs:
            current_state = {'epoch': epoch,
                                 'model_weights': model.state_dict(),
                                 'optimizer': optimizer.state_dict(),
                                 'args': args_dict,
                                 }

            model_path = os.path.join(args.output_path,f"checkpoint.pth.tar")
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
            model_path = os.path.join(args.output_path,f"model_best_f1.pth.tar")
            torch.save(current_state,model_path)
            log(f"Saved Model with best Validation F1 to: {model_path}")

    ## test model
    log("Testing best Validation Model")
    with torch.no_grad():
        model.load_state_dict(torch.load(model_path)['model_weights'])
        f1_test,loss_test,model,optimizer,class_report = forward_data(test_loader,"test",model,optimizer,torch.nn.MSELoss(),args,device)
        log(f"Test: (loss {loss_test:.4f}, F1 {f1_test:.4f})")
        log(f"Classification Report:{class_report}")

    #generate psudo labels
    log("Generating psudo labels")
    to_predict_loader = data.prepare_train_loader(args, IMG_SIZE, psudo_labeled=True)
    result = generate_psudo_label(teacher_model, to_predict_loader,device)
    data.append_psudo_label(result, ITER)
    log(f"Generation done. See the file: {ITER}")

    del test_loader, val_loader,train_loader,to_predict_loader,teacher_model

    log_file.close()
    writer.flush()
    writer.close()

def main(args):


    #train and test first teacher (baseline)

    teacher_model = efficientnet_b0(pretrained=True, num_classes=13)
    one_iteration(args, teacher_model,"iter_1",224, phase="first teacher")


    #train student

    student_model = efficientnet_b7(pretrained=True, dropout=0.5, stochastic_depth_prob=0.3,num_classes=13)
    one_iteration(args, student_model,"iter_2",600)

    #train student

    student_model = efficientnet_b7(pretrained=True, dropout=0.5, stochastic_depth_prob=0.3,num_classes=13)
    one_iteration(args, student_model,"iter_3",600)
        

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)