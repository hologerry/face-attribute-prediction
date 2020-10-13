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
selected_attrs = ['Arched_Eyebrows', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Eyeglasses',
                  'Gray_Hair', 'Heavy_Makeup', 'Male', 'Mouth_Slightly_Open', 'Mustache',
                  'No_Beard', 'Smiling', 'Young', 'Skin_0', 'Skin_1', 'Skin_2', 'Skin_3']
all_cnt = 0
all_cnts = {}
selected_cnt = 0
selected_cnts = {}

for attr in all_attrs:
    all_cnts[attr] = 0
    if attr in selected_attrs:
        selected_cnts[attr] = 0

threshold = 0.95
num_imgs = 70000 if dataset_name == 'ffhq' else 30000
num_all_attrs = 44 * num_imgs
num_selected_attrs = len(selected_attrs) * num_imgs

for i in range(num_imgs):
    sub_dir = f'{(i // 1000):02d}000'
    file_name = f'{i:05d}.txt'
    attr_file = ospj(ffhq_data_path, attr_prob_dir, sub_dir, file_name)
    with open(attr_file) as f:
        ls = f.readlines()
    for l in ls:
        s = l.strip().split()
        v0 = float(s[1])
        v1 = float(s[2])
        if max(v0, v1) > threshold:
            all_cnt += 1
            all_cnts[s[0]] += 1
            if s[0] in selected_attrs:
                selected_cnts[s[0]] += 1
                selected_cnt += 1


print("For threshold", threshold, "on dataset", dataset_name)
print(f"number of confident attributes [{all_cnt}/{num_all_attrs}] = [{all_cnt/num_all_attrs:.6f}]")
print(f"selected attributes [{selected_cnt}/{num_selected_attrs}] = [{selected_cnt/num_selected_attrs:.6f}]")

all_attr_prob_file = f'{dataset_name}_all_confident_attr_num.txt'
with open(all_attr_prob_file, 'w') as f:
    for k, v in all_cnts.items():
        f.write(k + " " + str(v) + "\n")

selected_attr_prob_file = f'{dataset_name}_selected_confident_attr_num.txt'
with open(selected_attr_prob_file, 'w') as f:
    for k, v in selected_cnts.items():
        f.write(k + " " + str(v) + "\n")
