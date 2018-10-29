import os.path as osp
import os, sys
import json
import cv2
import pdb

import GameController as gc
import imgdecode as imd

jsonDataKeys = ['position',
                'angle',
                'settingID',
                'image_path',
                'nodeID'
                ]

if __name__=='__main__':
    model_dir = osp.abspath(raw_input('>>> Model directory path: '))
    if not osp.exists(model_dir):
        raise NameError('{} does not exist'.format(model_dir))

    setID = raw_input('>>> Setting ID: ')

    try:
        con = gc.Controller()
        con.connect()
    except:
        print('Error: Socket connection failed')

    room_list = os.listdir(model_dir)

    try:
        for room in room_list:
            print('Collecting images in {}'.format(room))
            room_dir = osp.join(model_dir, room)
            room_f = os.listdir(room_dir)
            json_fname = []
            for rf in room_f:
                #print rf
                sp_fname = rf.split('_')
                rf_fformat = sp_fname[-1].split('.')[-1]
                if rf_fformat=='json':
                    try:
                        rf_set_id = sp_fname[-2]
                        if rf_set_id==str(setID):
                            json_fname.append(rf)
                    except:
                        print('Exception at {}'.format(rf))
                        pass
            if len(json_fname)!=1: # serveral .json file in a room with the same setting ID
                print('There are multiple .json file with the same setting ID at {}'.format(room))
                for idx, jf in enumerate(json_fname):
                    print('  {}: {}'.format(idx, jf))
                tmp = raw_input('>>> Which .json to be used:(0-{}) '.format(len(json_fname)-1))
                json_fname = json_fname[int(tmp)]
            elif len(json_fname)==0:
                print('Error: missing {}_{}_x.json'.format(room,setID))
                
            else:
                json_fname = json_fname[0]
            json_fname = osp.join(room_dir, json_fname)
            # load .json file
            with open(json_fname, 'r') as f:
                json_data = json.load(f)
            
            # check data format
            data_len = len(json_data['image_path'])
            for k in jsonDataKeys:
                if k not in json_data.keys():
                    print('No {} key'.format(k))
                    print('Error: .json file is not in the correct format')
                    sys.exit()
                if len(json_data[k])!=data_len:
                    print('Data length of key {} does not equal to {}'.format(k, data_len))
                    print('Error: .json file is not in the correct format')
                    sys.exit()

            for i in range(data_len):
                pos = json_data['position'][i]
                ang = json_data['angle'][i]
                # go to the pose
                con.setPos(pos[0], pos[1], pos[2])
                con.setRot(ang[0], ang[1], ang[2])
                # get pose
                json_data['position'][i] = con.getPos()
                json_data['angle'][i] = con.getRot()
                # take snapshot
                img = con.getSpherical()
                img = imd.decode(img)
                # save image
                img_path = json_data['image_path'][i]
                img_path = osp.join(room_dir, img_path)
                cv2.imwrite(img_path, img)

            # update some robot poses in json file
            with open(json_fname, 'w') as f:
                json.dump(json_data, f)
    finally:
       con.close()
       print('Connection end!')
       print('Program end!')
