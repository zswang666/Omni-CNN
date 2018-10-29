import json
import os
import os.path as osp
import pdb
from operator import itemgetter

txt_fname = 'CG_train.txt'

dataset_dir = osp.abspath(raw_input('>>> Directory of entire dataset: '))
env_list = os.listdir(dataset_dir)
for env in env_list:
    env_full = osp.join(dataset_dir,env)
    if not osp.isdir(env_full):
        env_list.remove(env)
txt_fname = osp.join(dataset_dir,txt_fname)

img_id = 0 # aggregate globally
node_id = 0 # aggregate globally
room_id = 0 # aggregate globally
env_id = 0
with open(txt_fname, 'w') as txt:
    txt.write('image_id node_id room_id image_path\n')

    for env in env_list:
        env_dir = osp.join(dataset_dir,env)
        if osp.isdir(env_dir):
            room_list = os.listdir(env_dir)

            room_file_list = []
            room_file_tuple_list = []
            for room in room_list:
                room_dir = osp.join(env_dir, room)
                if osp.isdir(room_dir):
                    cur_room_file_list = os.listdir(room_dir)
                    room_file_list.append(cur_room_file_list)
                    # sort room-file-list
                    for room_f in cur_room_file_list:
                        room_fformat = room_f.split('.')[-1]
                        if room_fformat=='jpg':
                            room_fnode_id = int(room_f.split('_')[-3])
                            room_file_tuple_list.append((room,room_id,room_fnode_id,room_f))
                    room_id += 1
        
            room_file_tuple_list = sorted(room_file_tuple_list, key=itemgetter(2))

            for (room,room_id_tmp,room_fnode_id,room_f) in room_file_tuple_list:
                node_id_tmp = node_id + room_fnode_id
                img_path = osp.join(env,room,room_f)
                txt.write('{} {} {} {}\n'.format(img_id,
                                                 node_id_tmp,
                                                 room_id_tmp,
                                                 img_path))
                img_id += 1

            cur_env_node_num = room_file_tuple_list[-1][2] + 1
            node_id += (cur_env_node_num)
            print('number of nodes at {}: {}'.format(env, cur_env_node_num))

            env_id += 1

print('')
print('Total number of environment: {}'.format(env_id))
print('Total number of rooms: {}'.format(room_id))
print('Total number of nodes: {}'.format(node_id))
print('Total number of images: {}'.format(img_id))

