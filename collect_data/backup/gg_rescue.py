import json
import os
import os.path as osp

exNum = 3
setNum = 2

env_dir = osp.abspath(raw_input('>>> Environment directory: '))
room_list = os.listdir(env_dir)

for room in room_list:
    '''
    room_dir = osp.join(env_dir,room)
    json_fname = osp.join(room_dir,'{}.json'.format(room))
    new_json_fname = osp.join(room_dir,'{}_raw.json'.format(room))
    os.system('mv {} {}'.format(json_fname, new_json_fname))
    '''

    new_json_data = {
        'image_path': [],
        'position': [],
        'angle': [],
        'settingID': [],
        'nodeID': []
    }

    room_dir = osp.join(env_dir,room)
    json_fname = osp.join(room_dir,'{}.json'.format(room))
    with open(json_fname, 'r') as f:
        json_data = json.load(f)
    
    data_len = len(json_data['image_path'])
    for i in range(data_len):
        if (i%3==0) and (json_data['settingID'][i]==0):
            new_json_data['image_path'].append(json_data['image_path'][i])
            new_json_data['position'].append(json_data['position'][i])
            new_json_data['angle'].append(json_data['angle'][i])
            new_json_data['settingID'].append(json_data['settingID'][i])
            new_json_data['nodeID'].append(json_data['nodeID'][i])

    new_json_fname = osp.join(room_dir,'{}_raw.json'.format(room))
    with open(new_json_fname, 'w') as f:
        json.dump(new_json_data,f)
    os.system('rm {}'.format(json_fname))

    f_list = os.listdir(room_dir)
    for f in f_list:
        fformat = f.split('.')[-1]
        if fformat=='jpg':
            print('remove {}'.format(f))
            os.system('rm {}'.format(osp.join(room_dir,f)))

