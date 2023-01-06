import torch
import torchvision.transforms as T
import torch.utils.data as data
from PIL import Image
import csv
import math

#TODO: 确定unlabled是否为labeled anno的后1000个

LABELED_ANNO_FILE = "labeled_cells_anno.csv"
PSUDO_ANNO_FILE = ""
IMG_FOLDER = ""
CELL_TYPES = ["Promyelocyte", "Myelocyte", "Monoblast", "Lymphocyte", 
            "Metamyelocyte", "Myeloblast", "Erythroblast", "Lymphoblast", 
            "Granulocyte", "Plasma cell", "erythrocyte", "Monocyte", 
            "Megakaryozyt"]
            
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

    test_set = []
    val_set = []
    global labeled_train_set
    labeled_train_set = []
    with open(LABELED_ANNO_FILE,newline='') as f:
        for type in CELL_TYPES:
            temp = []
            index = CELL_TYPES.index(type)
            for line in list(csv.reader(f)):
                if line[1:][index] == 1:
                    temp.append(line)
            end1 = int(math.ceil(0.2*len(temp)))
            end2 = int(math.ceil(0.1*len(temp)))
            test_set.append(temp[:end1+1])
            val_set.append(temp[end1+1:end1+1+end2])
            labeled_train_set.append(temp[end1+1+end2:])
        test_dataloader = data.DataLoader(CellDataset(test_set, IMG_FOLDER, args.width, args.height,transform=transforms), 
                                    batch_size=args.batch_size,
                                   shuffle=False,
                                   num_workers=args.workers,
                                   pin_memory=args.pin_memory)
        val_dataloader = data.DataLoader(CellDataset(val_set, IMG_FOLDER, args.width, args.height,transform=transforms), 
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
        with open(LABELED_ANNO_FILE,newline='') as f:
            to_predict = list(csv.reader(f))[6335:]
            data_set = CellDataset(to_predict, IMG_FOLDER, args.width, args.height)
            unlabeled_dataloader = data.DataLoader(data_set, batch_size=args.batch_size,
                                   shuffle=False,
                                   num_workers=args.workers,
                                   pin_memory=args.pin_memory)
            return unlabeled_dataloader

    if phase == "first teacher":
        data_set = CellDataset(labeled_train_set, IMG_FOLDER, args.width, args.height,transform=transforms)
        sampler = weighted_sampling(labeled_train_set)
        train_dataloader = data.DataLoader(data_set,batch_size=args.batch_size,
                                   shuffle=True,sampler = sampler,
                                   num_workers=args.workers,
                                   pin_memory=args.pin_memory)

    else:
        with open(PSUDO_ANNO_FILE,newline='') as f:
            train_set = list(csv.reader(f))
            train_set.append(labeled_train_set)
            sampler = weighted_sampling(train_set)
            data_set = CellDataset(train_set, IMG_FOLDER, args.width, args.height, T.RandAugment(magnitude=27))
            train_dataloader = data.DataLoader(data_set,batch_size=args.batch_size,
                                    sampler = sampler,
                                   shuffle=True,
                                   num_workers=args.workers,
                                   pin_memory=args.pin_memory)

    return train_dataloader


#根据所给标签集合，给UNLABELED_CSV_FILE里的cellType加上soft label（psudo label），return csv file path
#@param: psudo_labels 为dict, keys: cell id，value: list of float(soft labels)
#@param: iter_id 表示了第几次iteration，用来特定化csv file name
def append_psudo_label(psudo_labels, iter_id):
    target_path = str(iter_id)
    with open(target_path, newline='') as target:
        writer = csv.writer(target, delimiter=',')
        temp = ['id']
        temp.append(CELL_TYPES)
        writer.writerow(temp)

        for i in psudo_labels.items():
            temp = []
            temp.append(i[0])
            temp.append(i[1])
            writer.writerow(temp)
        
    PSUDO_ANNO_FILE = target_path

def soft2hard(label):
    max = math.max(label)
    ind = label.index(max)
    return CELL_TYPES[ind]


# def count_classes(test_set, anno_file):
#     result = []
#     columns = []
#     with open(anno_file,'r',newline='') as F:
#         reader = csv.reader(F)
#         for row in reader:
#             if row[0] in test_set: 
#                 continue
#             psudo_type = soft2hard(row)
#             columns.append(psudo_type)
#         for type in CELL_TYPES:
#             result.append(columns.count(type))
#     return result,columns
    
    

def weighted_sampling(dataset, threshold = 190):

    labels = []
    class_counts = []
    for anno in dataset:
        psudo_type = soft2hard(anno[1:])
        labels.append(psudo_type)
    for i in range(14):
        class_counts.append(labels.count(i))

    num_samples = sum(class_counts)
    class_weights = [math.min((num_samples/class_counts[i]),threshold) for i in range(len(class_counts))]
    weights = [class_weights[labels[i]] for i in range(int(num_samples))]
    sampler = data.WeightedRandomSampler(torch.DoubleTensor(weights), int(num_samples),replacement=True)

    return sampler

    '''
    {'Promyelocyte'}": 1038[830], "{'Myelocyte'}": 389[311], "{'Monoblast'}": 312[249], 
"{'Lymphocyte'}": 500[400], "{'Metamyelocyte'}": 275, "{'Myeloblast'}": 2255, 
"{'Erythroblast'}": 1127, "{'Lymphoblast'}": 35, "{'Granulocyte'}": 338, 
"{'Plasma cell'}": 52, "{'erythrocyte'}": 2 (10), 
"{'Monocyte'}": 6 (20), "{'Megakaryozyt'}": 3 (20)}
'''
    # Erythrocyte = [], Megakaryozyt = [], Monocyte = [], Lymphoblast = [], Plasma = [],
    # Metamyelocyte = [], Monoblast = [], Granulocyte = [], Myelocyte = [], Lymphocyte = [],
    # Promyelocyte = [], Erythroblast = [], Myeloblast = []


class CellDataset(data.Dataset):

    def __init__(self, cell_annos, cell_img_folder, width, height, transform = None):
        #@param data, list of cell_id
        super(CellDataset).__init__()
        self.cell_annos = cell_annos
        self.cell_img_folder = cell_img_folder
        self.height = height
        self.width = width
        self.transform = transform

    def __getitem__(self, idx):

        cell_id = self.cell_annos[idx][0]
        cell_anno = self.cell_annos[idx][1:]

        #TODO:get single image's id to locate it
        path = self.cell_img_folder + "/" + cell_id + ".png"
        label = torch.tensor(cell_anno)
        img = Image.open(path).convert('RGB') #WSI应该使用什么Image？？

        if self.transform is not None:
            img = self.transform(img)

        return cell_id, img, label

    def __len__(self):
        return len(self.cell_annos)
 
