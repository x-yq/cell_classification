import torch
import torchvision.transforms as T
import torch.utils.data as data
from PIL import Image
import csv
import math
from hyper import parser

PROJECT_FOLDER = "/home/h1/yexu660b/project/"
LABELED_ANNO_FILE = "labeled_cells_anno_1.csv"
UNLABELED_ANNO_FILE = "unlabeled_cells.csv"
IMG_FOLDER = "/lustre/scratch2/ws/1/s7740678-data_06/processed/ai4hematology_data/aml_healthy_50x_pseudonyms_scale_removed_png_cell_slices"
#IMG_FOLDER = "/beegfs/.global1/ws/s7740678-data_05/processed/ai4hematology_data/aml_healthy_50x_pseudonyms_scale_removed_png_cell_slices"
CELL_TYPES = ['Promyelocyte', 'Blast', 'Lymphocyte', 'Erythroblast', 'Myelocyte', 
        'Plasma cell', 'Metamyelocyte', 'Immature Monocyte', 'Band eosinophil granulocyte', 
        'Segmented eosinophil granulocyte', 'Band neutrophil granulocyte', 'Segmented neutrophil granulocyte', 
        'Segmented basophil granulocyte', 'Monocyte', 'Megakaryozyt', 'Band basophil granulocyte', 
        'Immature Lymphocyte', 'Platelet']
  
test_set = []
val_set = []
global labeled_train_set
labeled_train_set=[]
              
def prepare_test_val_loader(args, img_size):
    normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    
    transforms = T.Compose(
        [
            T.Resize(img_size),
            T.ToTensor(),
            normalize
        ]
    )
    
    with open(LABELED_ANNO_FILE,newline='') as f:
        reader = csv.reader(f)
        reader_list = list(reader)
        for i in range(len(CELL_TYPES)):
            temp = []
            for line in reader_list:
                if line[0] == "id":
                    continue
                if int(line[1:][i]) == 1:
                    to_append = list(map(int,line))
                    temp.append(to_append)
            end1 = int(math.floor(0.6*len(temp)))
            end2 = int(math.ceil(0.2*len(temp)))
            for item in temp[:end1]:
                labeled_train_set.append(item)
            for item in temp[end1:end1+end2]:
                val_set.append(item)
            for item in temp[end1+end2:]:
                test_set.append(item)
            
        test_dataloader = data.DataLoader(CellDataset(test_set, IMG_FOLDER,img_size,transform=transforms), 
                                    batch_size=args.batch_size,
                                   shuffle=False,
                                   num_workers=args.workers,
                                   pin_memory=args.pin_memory)
        val_dataloader = data.DataLoader(CellDataset(val_set, IMG_FOLDER, img_size,transform=transforms), 
                                    batch_size=args.batch_size,
                                   shuffle=False,
                                   num_workers=args.workers,
                                   pin_memory=args.pin_memory)
    return test_dataloader, val_dataloader



def prepare_train_loader(args, img_size, phase="teacher", psudo_labeled=False):

    normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    
    transforms = T.Compose(
        [
            T.Resize(img_size),
            T.ToTensor(),
            normalize
        ]
    )

    if psudo_labeled:
        to_predict = []
        with open(UNLABELED_ANNO_FILE,newline='') as f:
            reader = csv.reader(f)
            for index,line in enumerate(list(reader)):
                if index > args.num_psudo_labels:
                    break
                to_predict.append(list(map(int,line)))
            data_set = CellDataset(to_predict, IMG_FOLDER, img_size,transform=transforms)
            unlabeled_dataloader = data.DataLoader(data_set, batch_size=args.batch_size,
                                   shuffle=False,
                                   num_workers=args.workers,
                                   pin_memory=args.pin_memory)
            return unlabeled_dataloader

    if phase == "first teacher":
        #TESTING!!!
        data_set = CellDataset(labeled_train_set, IMG_FOLDER, img_size,transform=transforms)
        sampler = weighted_sampling(list(labeled_train_set))
        train_dataloader = data.DataLoader(data_set,batch_size=args.batch_size,
                                    sampler = sampler,
                                   num_workers=args.workers,
                                   pin_memory=args.pin_memory)

    else:
        with open(args.psudo_labels_csv,newline='') as f:
            train_set = []
            for item in list(csv.reader(f)):
                train_set.append(list(map(int,item))) 
            for item in labeled_train_set:
                train_set.append(item)
            trans = T.Compose(
          [
            T.RandAugment(),
            T.Resize(img_size),
            T.ToTensor(),
            normalize,
          ]
        )
            sampler = weighted_sampling(train_set)
            data_set = CellDataset(train_set, IMG_FOLDER, img_size, transform=trans)
            train_dataloader = data.DataLoader(data_set,batch_size=args.batch_size,
                                    sampler = sampler,
                                   num_workers=args.workers,
                                   pin_memory=args.pin_memory)

    return train_dataloader


#根据所给标签集合，给UNLABELED_CSV_FILE里的cellType加上soft label（psudo label），return csv file path
#@param: psudo_labels 为dict, keys: cell id，value: list of float(soft labels)
#@param: iter_id 表示了第几次iteration，用来特定化csv file name
def append_psudo_label(psudo_labels, iter_id):

    target_path = str(iter_id)+".csv"
    target_file = open(target_path,"w",newline='')
    writer = csv.writer(target_file, delimiter=',')

    for key,value in psudo_labels.items():
        temp = []
        temp.append(key)
        temp.extend(index2hard(value))
        writer.writerow(temp)
    target_file.close()
    parser.set_defaults(psudo_labels_csv=target_path)

def codes2index(labels):
    max_label = max(labels)
    ind = labels.index(max_label)
    return ind

def index2hard(ind):
    result = []
    for i in range(len(CELL_TYPES)):
        if i == ind:
            result.append(1)
        else:
            result.append(0)
    return result
        

def weighted_sampling(dataset, threshold = 250):

    labels = []
    class_counts = []
    for anno in dataset:
        psudo_type = codes2index(anno[1:])
        labels.append(psudo_type)
    for i in range(len(CELL_TYPES)):
        class_counts.append(labels.count(i))

    num_samples = sum(class_counts)
    class_weights = [min((num_samples/class_counts[i]),threshold) for i in range(len(class_counts))]
    weights = [class_weights[labels[i]] for i in range(int(num_samples))]
    sampler = data.WeightedRandomSampler(torch.DoubleTensor(weights), int(num_samples),replacement=True)

    return sampler


class CellDataset(data.Dataset):

    def __init__(self, cell_annos, cell_img_folder, img_size, transform = None):
        #@param data, list of cell_id
        super(CellDataset).__init__()
        self.cell_annos = cell_annos
        self.cell_img_folder = cell_img_folder
        self.img_size = img_size
        self.transform = transform

    def __getitem__(self, idx):

        # print("cell_anno length",len(self.cell_annos))
        # print("index",idx)
        anno = self.cell_annos[idx]
        cell_id = anno[0]
        cell_anno = anno[1:]

        #TODO:get single image's id to locate it
        path = self.cell_img_folder + "/" + "cell" + str(cell_id) + ".png"
        label = torch.tensor(cell_anno)
        try:
            im = Image.open(path).convert('RGB')
            img=im.resize((self.img_size, self.img_size))
        except:
            print("open error!!!!!!!!!!!!!!!!!")

        if self.transform is not None:
            img = self.transform(img)
        return cell_id, img, label

    def __len__(self):
        return len(self.cell_annos)