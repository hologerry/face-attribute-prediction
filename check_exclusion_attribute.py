
attr_file = '/D_data/Face_Editing/face_editing/data/celebahq/CelebAMask-HQ-attribute-anno-skin.txt'

# attributes = ['Goatee', 'Mustache', 'No_Beard', 'Sideburns']  # 两鬓胡须
'''
cnt 0: 2037
cnt 1: 25917
cnt 2: 1272
cnt 3: 774
cnt 4: 0
'''
# attributes = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']
'''
cnt 0: 10472
cnt 1: 19171
cnt 2: 357
cnt 3: 0
cnt 4: 0
'''

attributes = ['Bald', 'Straight_Hair', 'Wavy_Hair']
'''
cnt 0: 12331
cnt 1: 17459
cnt 2: 210
'''

check_beard_file = 'hair_type_exclusion.txt'
f = open(check_beard_file, 'w')
lines = open(attr_file, 'r').readlines()

all_attrs = lines[1].strip().split()

cnts = [0] * (len(attributes) + 1)
merge = [0, 0]
for l in lines[2:]:
    s = l.strip().split()
    vs = s[1:]
    cnt = 0
    f.write(s[0])
    for a in attributes:
        if vs[all_attrs.index(a)] == '1':
            f.write(' ' + a)
            cnt += 1
    f.write('\n')
    cnts[cnt] += 1

f.close()

for i, cnt in enumerate(cnts):
    print(f"cnt {i}: {cnt}")
