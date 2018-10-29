import json
import os
import os.path as osp
import numpy as np

jsonDataKeys = ['position',
                'angle',
                'settingID',
                'image_path',
                'nodeID'
                ]

pos_range = [-0.1, 0.1]

def checkDir(dir_path):
    if not osp.exists(dir_path):
        print('{} does not exist, create a new directory'.format(dir_path))
        os.makedirs(dir_path)
        return
    else:
        tmp = raw_input('>>> Warning: {} already exists, are you sure to cover it?(Y/N, program will quit if N) '\
                        .format(dir_path))
        if tmp=='Y':
            return
        else:
            print('Program quit!')
            sys.exit()

def sample_pos(pos, n):
    noise = np.random.uniform(pos_range[0], pos_range[1], (n,2))
    s_pos = np.tile(np.array(pos),(n,1))
    s_pos[:,0] += noise[:,0]
    s_pos[:,2] += noise[:,1]
    return s_pos.tolist()

def sample_ang(ang, n):
    noise = np.random.uniform(0, 360, n)
    s_ang = np.tile(np.array(ang),(n,1))
    s_ang[:,1] = noise
    return s_ang.tolist()

if __name__=='__main__':
    model_dir = osp.abspath(raw_input('>>> Model directory path: '))
    if not osp.exists(model_dir):
        raise NameError('{} does not exist'.format(model_dir))

    tmp = raw_input('>>> Use the same setting ID and example number?(Y/N) ')
    if tmp=='Y':
        use_same_setID_exNum = True
    else:
        use_same_setID_exNum = False

    room_list = os.listdir(model_dir)
    for idx, room in enumerate(room_list):
        print('Collecting images in {}'.format(room))
        if idx==0:
            setID = int(raw_input('>>> Setting ID: '))
            exNum = int(raw_input('>>> Number of examples: '))
        else:
            if not use_same_setID_exNum:
                setID = int(raw_input('>>> Setting ID: '))
                exNum = int(raw_input('>>> Number of examples: '))

        room_dir = osp.join(model_dir, room)
        json_fname = osp.join(room_dir, '{}_raw.json'.format(room))
        # load .json file
        with open(json_fname, 'r') as f:
            json_data = json.load(f)

        json_data_sampled = {
            'image_path': [], 
            'position': [],
            'angle': [],
            'settingID': [],
            'nodeID': []
        }
        
        data_len = len(json_data['image_path'])
        for i in range(data_len):
            pos = json_data['position'][i]
            ang = json_data['angle'][i]
            node_id = json_data['nodeID'][i]
            
            s_pos = sample_pos(pos, exNum)
            s_ang = sample_ang(ang, exNum)
            
            for ex_id in range(exNum):
                short_imgp = '{}_{}_{}_{}.jpg'.format(room, node_id, setID, ex_id)

                json_data_sampled['image_path'].append(short_imgp)
                json_data_sampled['settingID'].append(setID)
                json_data_sampled['nodeID'].append(node_id)
                json_data_sampled['position'].append(s_pos[ex_id])
                json_data_sampled['angle'].append(s_ang[ex_id])

        json_fname_sampled = '{}_{}_{}.json'.format(room, setID, exNum)
        json_fname_sampled = osp.join(room_dir, json_fname_sampled)
        if osp.exists(json_fname_sampled):
            tmp = raw_input('>>> Warning: {} already exists, are you sure to cover it?(Y/N) '.format(json_fname_sampled))
            if tmp=='Y':
                with open(json_fname_sampled, 'w') as f:
                    json.dump(json_data_sampled, f)
        else:
            with open(json_fname_sampled, 'w') as f:
                json.dump(json_data_sampled, f)
        

