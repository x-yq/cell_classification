import csv

LABELED_CSV_FILE = "aml_healthy_50x_pseudonyms_scale_removed_png_cell_slices_labeled.csv"
UNLABELED_CSV_FILE = "aml_healthy_50x_pseudonyms_scale_removed_png_cell_slices_all.csv"


types = {}
types_keys = []
unlabeled = []

def count_types():
    with open(LABELED_CSV_FILE,'r',newline='') as F:
        reader = csv.reader(F)
        for item in list(reader):
            if item[1] == "label":
                continue
            # if item[1] == "":
            #     unlabeled.append(item[0])
            #     continue
            if item[1] not in types:
                types[str(item[1])] = 1
            else:
                types[str(item[1])] += 1
        
        for i in types.keys():
            types_keys.append(i)

    print(types)    
    print(types_keys)
    
    with open(UNLABELED_CSV_FILE,'r',newline='') as X:
        reader = csv.reader(X)
        for item in list(reader):
            if item[1] == "label":
                continue
            if item[1] == "":
                unlabeled.append(item[0])
            #     continue
            # if item[1] not in types:
            #     types[str(item[1])] = 1
            # else:
            #     types[str(item[1])] += 1
        


def idx(list, key):
    if key not in list:
        return -1
    else:
        return list.index(key)

def generate_unlabeled_cells():
    writer = csv.writer(unlabel_csv)
    for id in unlabeled:
        writer.writerow([int(id),0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
                

def generate_annotation():
    with open(LABELED_CSV_FILE) as F:
        reader = csv.reader(F)
        for id,item in enumerate(list(reader)):
            writer = csv.writer(LABELED_ANNO, delimiter=',')
            if id == 0:
                temp = ['id']
                temp += types_keys
                print(temp)
                writer.writerow(temp)
                continue

            temp = [item[0]]
            type_idx = idx(types_keys, item[1])
            for i in range(len(types_keys)):
                if i == type_idx:
                    temp.append(1)
                else:
                    temp.append(0)
            writer.writerow(temp)

LABELED_ANNO = open("labeled_cells_anno_1.csv","w",newline='')
unlabel_csv = open("unlabeled_cells.csv","w",newline="")
count_types()
generate_annotation()
generate_unlabeled_cells()
LABELED_ANNO.close()
unlabel_csv.close()