import os

train_txt_path=os.path.join("data_tosplit","train.txt")
train_dir=os.path.join("data_tosplit","train")
valid_txt_path=os.path.join("data_tosplit","valid.txt")
valid_dir=os.path.join("data_tosplit","valid")

def gen_txt(txt_path,img_dir):
    f=open(txt_path,'w')

    for root,s_dirs,files in os.walk(img_dir):
        for sub_dir in s_dirs:
            i_dir=os.path.join(root,sub_dir)
            img_list=os.listdir(i_dir)
            # print(img_list)
            for i in range(len(img_list)):
                if not img_list[i].endswith('jpg'):
                    continue
                label=img_list[i].split('.')[0]
                if label=="dog":
                    label="0"
                elif label=="cat":
                    label="1"
                img_path=os.path.join(i_dir,img_list[i])
                # print(img_path)
                line=img_path+' '+label+'\n'
                f.write(line)

    f.close()

if __name__ == "__main__":
    gen_txt(train_txt_path,train_dir)
    gen_txt(valid_txt_path,valid_dir)

