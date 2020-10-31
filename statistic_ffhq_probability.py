from os.path import join as ospj

dataset_name = 'ffhq'

ffhq_data_path = f"/D_data/Face_Editing/face_editing/data/{dataset_name}"
attr_prob_dir = f"{dataset_name}_attr_prob"
all_attrs = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes',
             'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
             'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
             'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
             'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
             'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
             'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair',
             'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace',
             'Wearing_Necktie', 'Young', 'Skin_0', 'Skin_1', 'Skin_2', 'Skin_3']

all_attr_probs_0 = {}
all_attr_probs_1 = {}

for attr in all_attrs:
    all_attr_probs_0[attr] = []
    all_attr_probs_1[attr] = []


num_imgs = 70000 if dataset_name == 'ffhq' else 30000
num_all_attrs = 44 * num_imgs

for i in range(num_imgs):
    sub_dir = f'{(i // 1000):02d}000'
    file_name = f'{i:05d}.txt'
    attr_file = ospj(ffhq_data_path, attr_prob_dir, sub_dir, file_name)
    with open(attr_file) as f:
        ls = f.readlines()
    assert len(ls) == 44
    for l in ls:
        s = l.strip().split()
        v0 = float(s[1])
        v1 = float(s[2])
        # we always consider the predicted label with high confident
        if v0 > v1:
            # predict this image does not have this attribute
            all_attr_probs_0[s[0]].append(v0)
        else:
            all_attr_probs_1[s[0]].append(v1)


prob_ratio = 0.5
prob_ratio_s = str(prob_ratio).replace('.', '')
all_attr_threshold_by_ratio = f'/D_data/Face_Editing/face_editing/data/{dataset_name}/all_attr_threhold_by_ratio_{prob_ratio_s}.txt'

with open(all_attr_threshold_by_ratio, 'w') as f:
    for attr in all_attrs:
        cur_attr_probs_0 = all_attr_probs_0[attr]
        cur_attr_probs_1 = all_attr_probs_1[attr]
        cur_attr_probs_0 = sorted(cur_attr_probs_0, reverse=True)
        cur_attr_probs_1 = sorted(cur_attr_probs_1, reverse=True)

        index0 = int(len(cur_attr_probs_0) * prob_ratio)
        index1 = int(len(cur_attr_probs_1) * prob_ratio)
        thres_0 = cur_attr_probs_0[index0]
        thres_1 = cur_attr_probs_1[index1]
        name_thres = attr + " " + f"{thres_0:.6f} {thres_1:.6f}\n"
        f.write(name_thres)
        print(name_thres)
