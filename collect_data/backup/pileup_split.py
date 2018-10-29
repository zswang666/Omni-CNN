import json
import os, sys
import os.path as osp
import pdb
from operator import itemgetter

dataset_dir = osp.abspath(raw_input('>>> Directory of entire dataset: '))
env_list = os.listdir(dataset_dir)
for env in env_list:
    env_full = osp.join(dataset_dir,env)
    if not osp.isdir(env_full):
        env_list.remove(env)

leave_env_list = []
leave_room_list = []
while(True):
    leave_room = raw_input('>>> Choose one room to be split out (e.g. house/dining_room, Q to end): ')
    leave_room = leave_room.split('/')
    if leave_room[0]=='Q':
        break
    else:
        leave_env_list.append(leave_room[0])
        leave_room_list.append(leave_room[1])

txt_fname_sfx = raw_input('>>> Text file name suffix: ')
txt_fname = 'CG_train_{}.txt'.format(txt_fname_sfx)
txt_fname = osp.join(dataset_dir,txt_fname)

test_fname = 'CG_test_{}.txt'.format(txt_fname_sfx)
test_fname = osp.join(dataset_dir,test_fname)

img_id = 0 # aggregate globally
node_id = 0 # aggregate globally
room_id = 0 # aggregate globally
env_id = 0
total_leave_node_num = 0

test_lines = []
with open(txt_fname, 'w') as txt:
    txt.write('image_id room_id posx posy posz angx angy angz image_path\n')

    for env in env_list:
        env_dir = osp.join(dataset_dir,env)
        if osp.isdir(env_dir):
            room_list = os.listdir(env_dir)

            room_file_list = []
            room_file_tuple_list = []
            if env in leave_env_list:
                try:
                    for idx, leave_room in enumerate(leave_room_list):
                        if env==leave_env_list[idx]:
                            room_list.remove(leave_room)
                            leave_room_dir = osp.join(dataset_dir,env,leave_room)
                            leave_room_f_list = os.listdir(leave_room_dir)
                            leave_room_node_min = 9999
                            leave_room_node_max = 0
                            for room_f in leave_room_f_list:
                                room_fformat = room_f.split('.')[-1]
                                if room_fformat=='jpg':
                                    room_fnode_id = int(room_f.split('_')[-3])
                                    if room_fnode_id>leave_room_node_max:
                                        leave_room_node_max = room_fnode_id
                                    if room_fnode_id<leave_room_node_min:
                                        leave_room_node_min = room_fnode_id
                            leave_room_node_num = leave_room_node_max - leave_room_node_min + 1
                            print('*Leaving {}/{} out, omitting {} nodes'.format(env,leave_room,leave_room_node_num))

                            total_leave_node_num += leave_room_node_num
                except:
                    print('Warning: cannot find {}/{} or fail to parsing it'.format(env,leave_room))

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
print('Total number of nodes: {}'.format(node_id-total_leave_node_num))
print('Total number of images: {}'.format(img_id))

