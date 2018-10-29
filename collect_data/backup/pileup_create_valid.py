import json
import os
import os.path as osp
from operator import itemgetter

dataset_dir = osp.abspath(raw_input('>>> Directory of entire dataset: '))
env_list = os.listdir(dataset_dir)
for env in env_list:
    env_full = osp.join(dataset_dir,env)
    if not osp.isdir(env_full):
        env_list.remove(env)

valid_env_list = []
valid_room_list = []
while(True):
    valid_room = raw_input('>>> Validation room (e.g. house/dining_room, Q to end): ')
    valid_room = valid_room.split('/')
    if valid_room[0]=='Q':
        break
    else:
        valid_env_list.append(valid_room[0])
        valid_room_list.append(valid_room[1])

txt_fname_sfx = raw_input('>>> Text file name suffix: ')
txt_fname = 'CG_train_valid_{}.txt'.format(txt_fname_sfx)
txt_fname = osp.join(dataset_dir,txt_fname)


img_id = 0 # aggregate globally
node_id = 0 # aggregate globally
room_id = 0 # aggregate globally
prev_node_id = 0
with open(txt_fname, 'w') as txt:
    txt.write('image_id node_id room_id image_path\n')

    for env in env_list:
        room_file_list = []
        room_file_tuple_list = []
        if env in valid_env_list:
            env_dir = osp.join(dataset_dir,env)

            for idx, valid_room in enumerate(valid_room_list):
                if env==valid_env_list[idx]:
                    room_dir = osp.join(env_dir, valid_room)
                    room_file_list = os.listdir(room_dir)
                    for room_f in room_file_list:
                        room_fformat = room_f.split('.')[-1]
                        if room_fformat=='jpg':
                            room_fnode_id = int(room_f.split('_')[-3])
                            room_file_tuple_list.append((valid_room,room_id,room_fnode_id,room_f))
                    room_id += 1

            room_file_tuple_list = sorted(room_file_tuple_list, key=itemgetter(2))
            
            prev_room_fnode_id = 99999
            for (valid_room,room_id_tmp,room_fnode_id,room_f) in room_file_tuple_list:
                if room_fnode_id!=prev_room_fnode_id:
                    node_id += 1
                img_path = osp.join(env,valid_room,room_f)
                txt.write('{} {} {} {}\n'.format(img_id,
                                                 node_id,
                                                 room_id_tmp,
                                                 img_path))
                img_id += 1
                prev_room_fnode_id = room_fnode_id

            print('number of nodes at {}: {}'.format(env, node_id-prev_node_id))
            prev_node_id = node_id

print('')
print('Total number of nodes: {}'.format(node_id))
print('Total number of images: {}'.format(img_id))


