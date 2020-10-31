import os
from os.path import join as ospj

dataset_name = 'ffhq'

prob_ratio = 0.5
prob_ratio_s = str(prob_ratio).replace('.', '')
data_path = f"/D_data/Face_Editing/face_editing/data/{dataset_name}"
attr_prob_dir = f"{dataset_name}_attr_prob"

all_attr_threshold_by_ratio = f'/D_data/Face_Editing/face_editing/data/{dataset_name}/all_attr_threhold_by_ratio_{prob_ratio_s}.txt'

attr_pseudo_label = f"{dataset_name}_pseudo_label_{prob_ratio_s}"
label_dir = ospj(data_path, attr_pseudo_label)
os.makedirs(label_dir, exist_ok=True)
all_attr_threshold_by_ratio = ospj(data_path, f'all_attr_threhold_by_ratio_{prob_ratio_s}.txt')


all_attrs = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes',
             'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
             'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
             'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
             'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
             'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
             'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair',
             'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace',
             'Wearing_Necktie', 'Young', 'Skin_0', 'Skin_1', 'Skin_2', 'Skin_3']


num_imgs = 70000 if dataset_name == 'ffhq' else 30000
num_all_attrs = 44 * num_imgs

all_attr_thres = {}
with open(all_attr_threshold_by_ratio, 'r') as f:
    name_threses = f.readlines()
    for nt in name_threses:
        name_thres = nt.strip().split()
        name, thres0, thres1 = name_thres[0], float(name_thres[1]), float(name_thres[2])
        all_attr_thres[name] = (thres0, thres1)


all_attrs_cnts = {}
for attr in all_attrs:
    all_attrs_cnts[attr] = [0] * 2

for i in range(num_imgs):
    sub_dir = f'{(i // 1000):02d}000'
    file_name = f'{i:05d}.txt'
    attr_file = ospj(data_path, attr_prob_dir, sub_dir, file_name)
    label_file = ospj(label_dir, file_name)
    with open(attr_file) as f:
        ls = f.readlines()
    assert len(ls) == 44
    with open(label_file, 'w') as fl:
        for i, l in enumerate(ls):
            s = l.strip().split()
            v0 = float(s[1])
            v1 = float(s[2])
            thres0, thres1 = all_attr_thres[s[0]]
            if v0 > v1:
                all_attrs_cnts[s[0]][0] += v0 >= thres0
                fl.write(s[0] + " " + str(float(v0 >= thres0)) + "\n")  # 1.0 means, we train model with it
            else:
                all_attrs_cnts[s[0]][1] += v1 >= thres1
                fl.write(s[0] + " " + str(float(v1 >= thres1)) + "\n")  # 1.0 means, we train model with it

all_attr_count_by_ratio = ospj(data_path, f'all_attr_count_by_ratio_{prob_ratio_s}.txt')
cnt_f = open(all_attr_count_by_ratio, 'w')
for attr in all_attrs:
    cnt_f.write(f"{attr} {all_attrs_cnts[attr][0]} {all_attrs_cnts[attr][1]}\n")
