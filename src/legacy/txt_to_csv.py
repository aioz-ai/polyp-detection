import os
import pandas as pd

def txt_to_csv(path):
    txt_reader = open(path)
    values = []
    image_w = 574
    image_h = 500
    # image_w = 1225
    # image_h = 966
    class_name = 'polyp'
    for line in txt_reader:
        if line[-1] == "\n":
            line = line[:-1]
        filename,x,y,w,h = line.split(" ")
        value = [filename, image_w, image_h, class_name, x, y, str(int(x)+int(w)), str(int(y)+int(h))]
        value = [str(elem) for elem in value]
        values.append(value)

    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(values, columns=column_name)
    return xml_df


def main():
    txt_path = "bboxes/cvc_colon_train_bboxes.txt"
    xml_df = txt_to_csv(txt_path)
    xml_df.to_csv("csv/polyp_cvccolon_train.csv", index=None)
    print('Successfully converted xml to csv.')

if __name__ == "__main__":
    main()
