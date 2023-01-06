import csv

CSV_FILE = "D:/mds_diagnosis/csv/labeled_cells.csv"

types = {}
types_keys = []

#TODEBUG: 在labeled cells anno最后一列加上 labeled 

def count_types():
    with open(CSV_FILE,'r',newline='') as F:
        reader = csv.reader(F)
        for item in list(reader):
            if item[3] == "" or item[3] == "cellType":
                continue
            if item[3] not in types:
                types[str(item[3])] = 1
            else:
                types[str(item[3])] += 1
        
        for i in types.keys():
            types_keys.append(i)
        types_keys.remove("{'Promyelocyte', 'Erythroblast'}")

    print(types_keys)    

def idx(list, key):
    if key not in list:
        return -1
    else:
        return list.index(key)

def generate_annotation():
    with open(CSV_FILE) as F:
        reader = csv.reader(F)
        for id,item in enumerate(list(reader)):
            writer = csv.writer(LABELED_CSV_FILE, delimiter=',')
            if id == 0:
                temp = ['id']
                temp += types_keys
                print(temp)
                writer.writerow(temp)
                continue

            temp = [item[0]]
            type_idx = idx(types_keys, item[3])
            for i in range(len(types_keys)):
                if i == type_idx:
                    temp.append(1)
                else:
                    temp.append(0)
            writer.writerow(temp)

LABELED_CSV_FILE = open("labeled_cells_anno.csv","w",newline='')
count_types()
generate_annotation()
LABELED_CSV_FILE.close()