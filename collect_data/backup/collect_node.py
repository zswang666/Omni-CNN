import sys, select, termios, tty
import numpy as np
import os.path as osp
import os
import json

import GameController as gc

msg = '''
welcome
'''

keyMoves = ['w', 's', 'a', 'd']
keyUtils = ['q', 'h', ' ', 'n']

setNum = 1
exNum = 1
pos_range = [-0.0, 0.0]

def getKey():
    tty.setraw(sys.stdin.fileno())
    select.select([sys.stdin], [], [], 0)
    key = sys.stdin.read(1)
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

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
    return [ang]*n
    #noise = np.random.uniform(0, 360, n)
    #s_ang = np.tile(np.array(ang),(n,1))
    #s_ang[:,1] = noise
    #return s_ang.tolist()

if __name__=="__main__":
    model_dir = osp.abspath(raw_input('>>> Model directory path: '))
    checkDir(model_dir)

    try:
        con = gc.Controller()
        con.connect()
    except:
        print('Error: Socket connection failed')
    
    settings = termios.tcgetattr(sys.stdin) 
    try:
        print(msg)

        rn_pos = []
        rn_ang = []

        room_id = 0
        node_id = 0

        room_name = raw_input('>>> Room name: ')
        room_dir = osp.join(model_dir,'{}'.format(room_name))
        checkDir(room_dir)
        
        json_data = {
            'image_path': [], 
            'position': [],
            'angle': [],
            'settingID': [],
            'nodeID': []
        }

        keystate = [0,0,0,0]
        while(1):
            key = getKey()

            if key in keyMoves:
                idx = keyMoves.index(key)
                keystate[idx] = 1 - keystate[idx]
                if keystate[idx]:
                    print('Keyboard Response: key down {}'.format(key))
                    con.KeyDown(key)
                else:
                    print('Keyboard Response: key up {}'.format(key))
                    con.KeyUp(key)
            elif key in keyUtils:
                if key==' ':
                    print('Keyboard Response: take a snapshot')
                    # query data
                    posx, posy, posz = con.getPos()
                    angx, angy, angz = con.getRot()
                    img = con.getFirstView()

                    # location and view angle prototype
                    rn_pos.append([posx, posy, posz])
                    rn_ang.append([angx, angy, angz])
                elif key=='n': # define new room
                    print('Keyboard Response: go to next room')
                    # sample points and construct .json
                    for set_id in range(setNum):
                        for rn_id in range(len(rn_pos)):
                            # sample position and view angle
                            s_pos = sample_pos(rn_pos[rn_id], exNum)
                            s_ang = sample_ang(rn_ang[rn_id], exNum)

                            # append json data
                            for ex_id in range(exNum):
                                short_imgp = 'room{}_{}_{}_{}.jpg'.format(room_id,
                                                                          rn_id,
                                                                          set_id,
                                                                          ex_id)
                                json_data['image_path'].append(short_imgp)
                                json_data['settingID'].append(set_id)
                                json_data['nodeID'].append(node_id+rn_id)
                                json_data['position'].append(s_pos[ex_id])
                                json_data['angle'].append(s_ang[ex_id])
                    # save .json file for current room
                    json_fname = '{}_raw.json'.format(room_name) #'room{}.json'.format(room_id)
                    json_fname = osp.join(room_dir, json_fname)
                    if osp.exists(json_fname):
                        tmp = raw_input('>>> Warning: {} already exists, are you sure to cover it?(Y/N) '.format(json_fname))
                        if tmp=='Y':
                            with open(json_fname, 'w') as f:
                                json.dump(json_data, f)
                    else:
                        with open(json_fname, 'w') as f:
                            json.dump(json_data, f)
                    # refresh memory
                    node_id += len(rn_pos)
                    json_data = {
                        'image_path': [], 
                        'position': [],
                        'angle': [],
                        'settingID': [],
                        'nodeID': []
                    }
                    rn_pos = []
                    rn_ang = []
                    # go to next room
                    room_id += 1
                    room_name = raw_input('>>> Room name: ')
                    room_dir = osp.join(model_dir,'{}'.format(room_name))
                    checkDir(room_dir)
                elif key=='h':
                    print('Keyboard Response: help')
                elif key=='q':
                    print('Keyboard Response: quit')
                    # sample points and construct .json
                    for set_id in range(setNum):
                        for rn_id in range(len(rn_pos)):
                            # sample position and view angle
                            s_pos = sample_pos(rn_pos[rn_id], exNum)
                            s_ang = sample_ang(rn_ang[rn_id], exNum)

                            # append json data
                            for ex_id in range(exNum):
                                short_imgp = 'room{}_{}_{}_{}.jpg'.format(room_id,
                                                                          rn_id,
                                                                          set_id,
                                                                          ex_id)
                                json_data['image_path'].append(short_imgp)
                                json_data['settingID'].append(set_id)
                                json_data['nodeID'].append(node_id+rn_id)
                                json_data['position'].append(s_pos[ex_id])
                                json_data['angle'].append(s_ang[ex_id])
                    # save .json file for current room
                    json_fname = '{}_raw.json'.format(room_name)
                    json_fname = osp.join(room_dir, json_fname)
                    if osp.exists(json_fname):
                        tmp = raw_input('>>> Warning: {} already exists, are you sure to cover it?(Y/N) '.format(json_fname))
                        if tmp=='Y':
                            with open(json_fname, 'w') as f:
                                json.dump(json_data, f)
                    else:
                        with open(json_fname, 'w') as f:
                            json.dump(json_data, f)

                    break
            else:
                pass

    #except ValueError:
    #    print('gg')

    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)        
        con.close()
        print('Connection end!')
        print('Program end!')
