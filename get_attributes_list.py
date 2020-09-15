attribute_file = 'ffhq_attributes_list.txt'

with open(attribute_file) as f:
    lines = f.readlines()

select_attr = 'Eyeglasses'
all_attrs = lines[0].strip().split()

select_attr_idx = all_attrs.index(select_attr)

select_attr_img_lists = f'select_{select_attr}.txt'
cnt = 0
with open(select_attr_img_lists, 'w') as f:
    for l in lines[1:]:
        splits = l.strip().split()
        name = splits[0]
        v = splits[select_attr_idx]
        if v == '1':
            f.write(name + '\n')
            cnt += 1

print(f"number of select attr {select_attr}:", cnt)
