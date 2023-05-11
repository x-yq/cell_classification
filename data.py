import torch
import torchvision.transforms as T
import torch.utils.data as data
from PIL import Image
import csv
import math
import os
import datetime
from argparse import ArgumentParser

class DataPreparation:
    def __init__(self,parser:ArgumentParser):
        #s = se.SequenceOn()
        self.parser = parser
        self.args = parser.parse_args()
        self.LABELED_ANNO_FILE = ""
        self.UNLABELED_ANNO_FILE = ""
        self.IMG_FOLDER = ""
        self.CELL_TYPES = []

        ## make pseudo labels folder
        date_=datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.labels_path = os.path.join(self.parser.parse_args().output_folder,f"{date_}_{self.args.trial_name}_pseudo_labels_csv_files")
        os.makedirs(self.labels_path)

        self.test_set = []
        self.val_set = []
        self.labeled_train_set = []
        self.LABELED_ANNO_FILE = self.args.labeled_anno_file
        self.UNLABELED_ANNO_FILE = self.args.unlabeled_anno_file
        self.IMG_FOLDER = self.args.image_folder
        self.CELL_TYPES = self.get_all_class(self.args.labeled_anno_file)

        self.normalize = T.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
        self.transforms = T.Compose(
            [
                T.Resize(self.args.img_size),
                T.ToTensor(),
                self.normalize
            ]
        )
        self.test_loader, self.val_loader = self.prepare_test_val_loader()

    def update_args(self,**kws):
        #s = se.SequenceOn()
        
        self.parser._defaults.update(kws)
        
        for action in self.parser._actions:
            if action.dest in kws:
                action.default = kws[action.dest]

        self.args = self.parser.parse_args()
    
    def get_all_class(self,path):
        #s = se.SequenceOn()
        with open(path,'r') as p:
            reader = list(csv.reader(p))
            result = []
            for ind,line in enumerate(reader):
                if line[0]=='id':
                    result = line[2:]
                    break
        return result

    def prepare_test_val_loader(self):
        #s = se.SequenceOn()
        with open(self.LABELED_ANNO_FILE,'r') as f:
            reader = csv.reader(f)
            reader_list = list(reader)
            for i in range(len(self.CELL_TYPES)):
                temp = []
                for line in reader_list:
                    if len(line) == 0:
                        continue
                    if line[0] == "id":
                        continue
    #                if int(line[1:][i]) == 1:
    #                    to_append = list(map(int,line))
    #                    temp.append(to_append)
                    ## new data set!!!!
                    if int(line[2:][i]) == 1:
                        to_append = list(line)
                        temp.append(to_append)
                end1 = int(math.floor(0.6*len(temp)))
                end2 = int(math.ceil(0.2*len(temp)))
                for item in temp[:end1]:
                    self.labeled_train_set.append(item)
                for item in temp[end1:end1+end2]:
                    self.val_set.append(item)
                for item in temp[end1+end2:]:
                    self.test_set.append(item)
            print("Size of validation set", len(self.val_set))
            print("Size of labeled train set", len(self.labeled_train_set))
            test_dataloader = data.DataLoader(CellDataset(self.test_set, self.IMG_FOLDER,self.args.img_size,transform=self.transforms), 
                                        batch_size=self.args.batch_size,
                                    shuffle=False,
                                    num_workers=self.args.workers,
                                    pin_memory=self.args.pin_memory)
            val_dataloader = data.DataLoader(CellDataset(self.val_set, self.IMG_FOLDER, self.args.img_size,transform=self.transforms), 
                                        batch_size=self.args.batch_size,
                                    shuffle=False,
                                    num_workers=self.args.workers,
                                    pin_memory=self.args.pin_memory)
        return test_dataloader, val_dataloader
    
    def prepare_train_loader(self, phase="teacher", pseudo_labeled=False):
        #s = se.SequenceOn()

        if pseudo_labeled:
            to_predict = []
            with open(self.UNLABELED_ANNO_FILE,'r') as f:
                reader = csv.reader(f)
                for index,line in enumerate(list(reader)):
                    if len(line) == 0:
                        print("---------------------",index,line)
                        continue
                    if line[0] == "id":
                        continue
                    if index+1 > self.args.num_pseudo_labels:
                        break
                    ## new data set!!!
                    #to_predict.append(list(map(int,line)))
                    to_predict.append(list(line))
                data_set = CellDataset(to_predict, self.IMG_FOLDER, self.args.img_size,transform=self.transforms)
                unlabeled_dataloader = data.DataLoader(data_set, batch_size=self.args.batch_size,
                                    shuffle=False,
                                    num_workers=self.args.workers,
                                    pin_memory=self.args.pin_memory)
                return unlabeled_dataloader

        if phase == "first teacher":
            data_set = CellDataset(self.labeled_train_set, self.IMG_FOLDER, self.args.img_size,transform=self.transforms)
            sampler = self.weighted_sampling(list(self.labeled_train_set), threshold = self.args.threshold)
            train_dataloader = data.DataLoader(data_set,batch_size=self.args.batch_size,
                                        sampler = sampler,
                                    num_workers=self.args.workers,
                                    pin_memory=self.args.pin_memory)

        else:
            with open(self.args.pseudo_labels_csv,'r') as f:
                train_set = []
                for item in list(csv.reader(f)):
                    #train_set.append(list(map(int,item))) 
                    ## new data set!!!
                    train_set.append(list(item))
                for item in self.labeled_train_set:
                    train_set.append(item)
                trans = T.Compose(
            [
                T.RandAugment(),
                self.transforms
            ]
            )
                sampler = self.weighted_sampling(train_set, threshold = self.args.threshold)
                data_set = CellDataset(train_set, self.IMG_FOLDER, self.args.img_size, transform=trans)
                train_dataloader = data.DataLoader(data_set,batch_size=self.args.batch_size,
                                        sampler = sampler,
                                    num_workers=self.args.workers,
                                    pin_memory=self.args.pin_memory)

        return train_dataloader
    
    def append_pseudo_label(self,pseudo_labels, iter_id):
        #s = se.SequenceOn()

        target_path = os.path.join(self.labels_path,str(iter_id)+".csv")
        
        with open(target_path,"w") as target_file:
            writer = csv.writer(target_file, delimiter=',')

            for key,value in pseudo_labels.items():
                temp = []
                temp.append(key)
                temp.append(value[0])
                temp.extend(self.index2hard(value[1]))
                writer.writerow(temp)
            
    def codes2index(self,labels):
        max_label = max(labels)
        ind = labels.index(max_label)
        return ind

    def index2hard(self,ind):
        result = []
        for i in range(len(self.CELL_TYPES)):
            if i == ind:
                result.append(1)
            else:
                result.append(0)
        return result
            

    def weighted_sampling(self,dataset, threshold = 200000):
        #s = se.SequenceOn()

        labels = []
        class_counts = []
        for anno in dataset:
            pseudo_type = self.codes2index(list(map(int,anno[2:])))
            labels.append(pseudo_type)
        for i in range(len(self.CELL_TYPES)):
            class_counts.append(labels.count(i))

        num_samples = sum(class_counts)
        class_weights = [min((num_samples/class_counts[i]),threshold) for i in range(len(class_counts))]
        weights = [class_weights[labels[i]] for i in range(int(num_samples))]
        sampler = data.WeightedRandomSampler(torch.DoubleTensor(weights), int(num_samples),replacement=True)
        return sampler
    
    

# #LABELED_ANNO_FILE = "labeled_cells_anno_1.csv"
# LABELED_ANNO_FILE = "labeled_new.csv"
# #UNLABELED_ANNO_FILE = "unlabeled_cells.csv"
# UNLABELED_ANNO_FILE = "unlabeled_new.csv"
# #IMG_FOLDER = "/lustre/scratch2/ws/1/s7740678-data_07/processed/ai4hematology_data/aml_healthy_50x_pseudonyms_scale_removed_png_cell_slices"
# IMG_FOLDER = "/lustre/scratch2/ws/1/s7740678-data_07/raw/BM_cytomorphology_data/"
# #CELL_TYPES = ['Promyelocyte', 'Blast', 'Lymphocyte', 'Erythroblast', 'Myelocyte', 
# #        'Plasma cell', 'Metamyelocyte', 'Immature Monocyte', 'Band eosinophil granulocyte', 
# #        'Segmented eosinophil granulocyte', 'Band neutrophil granulocyte', 'Segmented neutrophil granulocyte', 
# #        'Segmented basophil granulocyte', 'Monocyte', 'Megakaryozyt', 'Band basophil granulocyte', 
# #        'Immature Lymphocyte', 'Platelet']
# CELL_TYPES = ['BAS', 'NGB', 'EBO', 'FGC', 'MON', 'LYT', 'NIF', 'MYB', 'PMO', 'ART', 'HAC', 'PEB', 'KSC', 'OTH', 'LYI', 'MMZ', 'ABE', 'BLA', 'PLM', 'NGS', 'EOS']

# test_set = []
# val_set = []
# global labeled_train_set
# labeled_train_set=[]
              
# def prepare_test_val_loader(args, img_size):
#     normalize = T.Normalize(
#             mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
#         )
    
#     transforms = T.Compose(
#         [
#             T.Resize(img_size),
#             T.ToTensor(),
#             normalize
#         ]
#     )
    
#     with open(LABELED_ANNO_FILE,'r') as f:
#         reader = csv.reader(f)
#         reader_list = list(reader)
#         for i in range(len(CELL_TYPES)):
#             temp = []
#             for line in reader_list:
#                 if len(line) == 0:
#                     continue
#                 if line[0] == "id":
#                     continue
# #                if int(line[1:][i]) == 1:
# #                    to_append = list(map(int,line))
# #                    temp.append(to_append)
#                 ## new data set!!!!
#                 if int(line[3:][i]) == 1:
#                     to_append = list(line)
#                     temp.append(to_append)
#             end1 = int(math.floor(0.6*len(temp)))
#             end2 = int(math.ceil(0.2*len(temp)))
#             for item in temp[:end1]:
#                 labeled_train_set.append(item)
#             for item in temp[end1:end1+end2]:
#                 val_set.append(item)
#             for item in temp[end1+end2:]:
#                 test_set.append(item)
#         num_labeled_train_data = len(labeled_train_set)  
#         test_dataloader = data.DataLoader(CellDataset(test_set, IMG_FOLDER,img_size,transform=transforms), 
#                                     batch_size=args.batch_size,
#                                    shuffle=False,
#                                    num_workers=args.workers,
#                                    pin_memory=args.pin_memory)
#         val_dataloader = data.DataLoader(CellDataset(val_set, IMG_FOLDER, img_size,transform=transforms), 
#                                     batch_size=args.batch_size,
#                                    shuffle=False,
#                                    num_workers=args.workers,
#                                    pin_memory=args.pin_memory)
#     return test_dataloader, val_dataloader



# def prepare_train_loader(args, img_size, phase="teacher", pseudo_labeled=False):

#     normalize = T.Normalize(
#             mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
#         )
    
#     transforms = T.Compose(
#         [
#             T.Resize(img_size),
#             T.ToTensor(),
#             normalize
#         ]
#     )

#     if pseudo_labeled:
#         to_predict = []
#         with open(UNLABELED_ANNO_FILE,'r') as f:
#             reader = csv.reader(f)
#             for index,line in enumerate(list(reader)):
#                 if len(line) == 0:
#                     print("---------------------",index,line)
#                     continue
#                 if line[0] == "id":
#                     continue
#                 if index+1 > args.num_pseudo_labels:
#                     break
#                 ## new data set!!!
#                 #to_predict.append(list(map(int,line)))
#                 to_predict.append(list(line))
#             data_set = CellDataset(to_predict, IMG_FOLDER, img_size,transform=transforms)
#             unlabeled_dataloader = data.DataLoader(data_set, batch_size=args.batch_size,
#                                    shuffle=False,
#                                    num_workers=args.workers,
#                                    pin_memory=args.pin_memory)
#             return unlabeled_dataloader

#     if phase == "first teacher":
#         data_set = CellDataset(labeled_train_set, IMG_FOLDER, img_size,transform=transforms)
#         sampler = weighted_sampling(list(labeled_train_set), threshold = args.threshold)
#         train_dataloader = data.DataLoader(data_set,batch_size=args.batch_size,
#                                     sampler = sampler,
#                                    num_workers=args.workers,
#                                    pin_memory=args.pin_memory)

#     else:
#         with open(args.pseudo_labels_csv,'r') as f:
#             train_set = []
#             for item in list(csv.reader(f)):
#                 #train_set.append(list(map(int,item))) 
#                 ## new data set!!!
#                 train_set.append(list(item))
#             for item in labeled_train_set:
#                 train_set.append(item)
#             trans = T.Compose(
#           [
#             T.RandAugment(),
#             T.Resize(img_size),
#             T.ToTensor(),
#             normalize,
#           ]
#         )
#             sampler = weighted_sampling(train_set, threshold = args.threshold)
#             data_set = CellDataset(train_set, IMG_FOLDER, img_size, transform=trans)
#             train_dataloader = data.DataLoader(data_set,batch_size=args.batch_size,
#                                     sampler = sampler,
#                                    num_workers=args.workers,
#                                    pin_memory=args.pin_memory)

#     return train_dataloader


# #根据所给标签集合，给UNLABELED_CSV_FILE里的cellType加上soft label（pseudo label），return csv file path
# #@param: pseudo_labels 为dict, keys: cell id，value: list of float(soft labels)
# #@param: iter_id 表示了第几次iteration，用来特定化csv file name
# #def append_pseudo_label(pseudo_labels, iter_id, labels_path):
# #
# #    target_path = os.path.join(labels_path,str(iter_id)+".csv")
# #    
# #    with open(target_path,"w",newline='') as target_file:
# #        writer = csv.writer(target_file, delimiter=',')
# #
# #        for key,value in pseudo_labels.items():
# #            temp = []
# #            temp.append(key)
# #            temp.extend(index2hard(value))
# #            writer.writerow(temp)

# def append_pseudo_label(pseudo_labels, iter_id, labels_path):

#     target_path = os.path.join(labels_path,str(iter_id)+".csv")
    
#     with open(target_path,"w") as target_file:
#         writer = csv.writer(target_file, delimiter=',')

#         for key,value in pseudo_labels.items():
#             temp = []
#             temp.append(key)
#             temp.append(value[1])
#             temp.append('')
#             temp.extend(index2hard(value[0]))
#             writer.writerow(temp)
            
# def codes2index(labels):
#     max_label = max(labels)
#     ind = labels.index(max_label)
#     return ind

# def index2hard(ind):
#     result = []
#     for i in range(len(CELL_TYPES)):
#         if i == ind:
#             result.append(1)
#         else:
#             result.append(0)
#     return result
        

# def weighted_sampling(dataset, threshold = 200000):

#     labels = []
#     class_counts = []
#     for anno in dataset:
#     ## new data set !!! 1-->3
#         pseudo_type = codes2index(list(map(int,anno[3:])))
#         labels.append(pseudo_type)
#     for i in range(len(CELL_TYPES)):
#         class_counts.append(labels.count(i))

#     num_samples = sum(class_counts)
#     #class_weights = [(num_samples/class_counts[i]) for i in range(len(class_counts))]
#     class_weights = [min((num_samples/class_counts[i]),threshold) for i in range(len(class_counts))]
#     weights = [class_weights[labels[i]] for i in range(int(num_samples))]
#     sampler = data.WeightedRandomSampler(torch.DoubleTensor(weights), int(num_samples),replacement=True)

#     return sampler


class CellDataset(data.Dataset):

    def __init__(self, cell_annos, cell_img_folder, img_size, transform = None):
        #s = se.SequenceOn()
        super(CellDataset).__init__()
        self.cell_annos = cell_annos
        self.cell_img_folder = cell_img_folder
        self.img_size = img_size
        self.transform = transform

    def __getitem__(self, idx):

        anno = self.cell_annos[idx]
        cell_id = anno[0]
        cell_folder = anno[1]
        cell_anno = list(map(int,anno[2:]))

        path1 = os.path.join(self.cell_img_folder,str(anno[1]),str(cell_id)+".jpg")
        path2 = os.path.join(self.cell_img_folder,str(anno[1]),str(cell_id)+".png")
        label = torch.tensor(cell_anno)

        try:
            im = Image.open(path1).convert('RGB')
        except:
            print("Wrong file type: expected png but jpg.")
            print(path1)
            im = Image.open(path2).convert('RGB')
        
        img=im.resize((self.img_size, self.img_size))

        if self.transform is not None:
            img = self.transform(img)

        return cell_id, img, label, cell_folder

    def __len__(self):
        return len(self.cell_annos)