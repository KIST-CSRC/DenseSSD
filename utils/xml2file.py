import json
import natsort
import os


def get_attributes(file):
    f = open(file)
    info_df = json.load(f)
    info = []

    for i in info_df['images']:
        file_name = i['file_name']

    info.append(file_name)
    for i in info_df['annotations']:
        bbox = i['bbox']
        category_id = i['category_id']
        info.append(bbox[0])
        info.append(bbox[1])
        info.append(bbox[2])
        info.append(bbox[3])
        info.append(int(category_id))

    return info


if __name__ == '__main__':
    path = 'Vial Positioning DB/Empty/train/annotation'
    text_file = 'dataset/vialPositioningDataset.txt'

    folder = os.listdir(path)
    folder = natsort.natsorted(folder)

    f = open(text_file, "w")

    for file in folder:
        p = os.path.join(path + '/', file)
        json_info = get_attributes(p)

        for i in range(len(json_info)):
            if i == 0:
                f.write(str(json_info[i]) + " ")
            elif i == len(json_info)-1:
                f.write(str(json_info[i]))
            else:
                f.write(str(json_info[i]) + " ")
        f.write("\n")
    f.close()
