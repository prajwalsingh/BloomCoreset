import os
from shutil import copyfile


ROOT = '../../dataset/food11'

classes = list(os.listdir(os.path.join(ROOT, 'training')))

if not os.path.isdir(os.path.join(ROOT, 'train')):
    os.makedirs(os.path.join(ROOT, 'train'))

mode = ['training', 'validation']

for m in mode:
    source_root = os.path.join(ROOT,  m)
    target_root = os.path.join(ROOT, 'train')

    for cls in classes:
        file_list = os.listdir(os.path.join(source_root, cls))

        for file in file_list:
            source = os.path.join(source_root, cls, file)
            target = os.path.join(target_root, cls, '{}_{}'.format(m, file))
            if not os.path.exists(os.path.join(target_root, cls)):
                os.makedirs(os.path.join(target_root, cls))
            copyfile(source, target)
