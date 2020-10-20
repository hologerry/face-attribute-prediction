import os

celeba_hq_file_list = '/D_data/Face_Editing/face_editing/data/celeba/CelebA-HQ-to-CelebA-mapping.txt'

new_celeba_img_path = '/D_data/Face_Editing/face_editing/data/celeba/img_celeba'

# img_names = sorted(os.listdir(new_celeba_img_path))
# celeba_hq_img_names = []
# lines = open(celeba_hq_file_list).readlines()[1:]

# for l in lines:
#     celeba_hq_img_names.append(l.strip().split()[-1])

extra_img_file_names = '/D_data/Face_Editing/face_editing/data/celeba/extra_imgs.txt'
# extra_img_file_names_selected = '/D_data/Face_Editing/face_editing/data/celeba/extra_imgs_selected.txt'
# f = open(extra_img_file_names, 'w')
# fs = open(extra_img_file_names_selected, 'w')

# cnt = 0
# for img in img_names:
#     if img not in celeba_hq_img_names:
#         f.write(img + '\n')
#         cnt += 1
#         if cnt <= 2000:
#             fs.write(img + '\n')
# f.close()
# fs.close()

all_celeba_annoations = 'list_attr_celeba.txt'
img_name_to_anno = {}
lines = open(all_celeba_annoations).readlines()
annos = lines[2:]
for l in annos:
    s = l.split('.jpg')
    img_name_to_anno[s[0]] = s[1]

anno_extra_img_file_names = '/D_data/Face_Editing/face_editing/data/celeba/list_attr_extra_imgs.txt'
# anno_extra_img_file_names = '/D_data/Face_Editing/face_editing/data/celeba/list_attr_extra_imgs_selected.txt'
afs = open(anno_extra_img_file_names, 'w')
extra_file_names = open(extra_img_file_names).readlines()

afs.write(str(len(extra_file_names))+'\n')
afs.write(lines[1])
for efn in extra_file_names:
    efn = efn.strip()
    n = efn.split('.')[0]
    afs.write(efn + img_name_to_anno[n])
