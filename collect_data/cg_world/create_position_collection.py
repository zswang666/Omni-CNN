import os
import sys
sys.path.append('./socketapi')
import GameController as gc
import imgdecode as imd
import json
import select, termios, tty
import numpy as np
import time
import threading
import cv2
import random
import math
import pdb

"""
Human will drive robot in unity and take position collection in this file
each Scene will be ran once and create one root collection json file, ex: Scene_0-root.json
each room in Scene will have 8~40 points chosen, depends on the size of the room
json file will store the following:
    scene_name: string; Scene_0
    node_ids: list of string; string will have prefix Scene_name, ex: Scene_0-1
    room_ids: list of string; string will have prefix Scene_name, ex: Scene_0-tv_room
    positions: list of list of float (x,y,z)
    angles: list of list of float (ang_x, ang_y, ang_z)
"""

keyMoves = ['w', 's', 'a', 'd']
keyUtils = ['q', ' ', 'h', 'n', 'c']

class Recorder:
    """
    The txt file writer, will collect picken point information and save it to json file
    """
    def __init__(self, collection_path, scene_name):
        """
        Input:
            collection_path: string; the path of the saved collection json file
            scene_name: string; the name of the scene
        """
        # image filename to a dictionary of position, angle, semantic_tag
        print "myname:"+scene_name
        self.collection_path = collection_path
        self.scene_name = scene_name
        self.node_ids = []
        self.room_ids = []
        self.positions = []
        self.angles = []
        self.room_tag = 'None'
        self.node_id = 0
        self.room_points_count = 0
        self.reset_controller()


    def record_position(self):
        """
        record the position, angle, node_id and room_id for future sampling usage
        Input:
            None
        Return:
            sample_point_nums: int; number of sample points
        """
        self.node_id += 1
        self.room_points_count += 1
        position = list(self.con.getPos())
        angle = list(self.con.getRot())
        room_id = self.scene_name + "_" + self.room_tag
        node_id = self.scene_name + "_" + str(self.node_id)
        self.node_ids.append(node_id)
        self.room_ids.append(room_id)
        self.positions.append(position)
        self.angles.append(angle)
        print("{} points collected, room {} {} nodes collected".format(self.node_id, self.room_tag, self.room_points_count))
        return

    def change_tag(self):
        """
        change the semantic tag
        """
        self.room_tag = raw_input("please type the room tag you want to change to: ")
        self.room_points_count = 0
        print("room tag set to: {}".format(self.room_tag))

    def save(self):
        """
        save the points taken by human
        """
        print("\nstart saving collection json file")
        json_data = {}
        json_data['scene_name'] = self.scene_name
        json_data['node_ids'] = self.node_ids
        json_data['room_ids'] = self.room_ids
        json_data['positions'] = self.positions
        json_data['angles'] = self.angles
        with open(self.collection_path, 'w') as f:
            json.dump(json_data, f)
            print("collection json file successfully saved in: {}\n".format(self.collection_path))

    def reset_controller(self):
        """
        Reset the controller
        """
        try:
            self.con = gc.Controller()
            self.con.connect()
            self.con.setSpeed(2.0)
            self.con.setRotateSpeed(40.)
            print("successfully connect to game controller")
        except:
            print('Error: Socket Connection failed')
            print('did you open your unity exe file?')
            sys.exit()

    def get_controller(self):
        """
        get controller
        """
        return self.con


def getKey():
    tty.setraw(sys.stdin.fileno())
    select.select([sys.stdin], [], [], 0)
    key = sys.stdin.read(1)
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key


if __name__ == '__main__':
    # user input root directory of the whole final test
    root_dir = raw_input('>>> Final test root directory (enter "d" to use default: /home/joehuang/Desktop/room_data_all): ')
    if root_dir == 'd':
        root_dir = '/home/joehuang/Desktop/room_data_all'
    # open location bag json file
    scene_name = raw_input('>>> The scene name you will going to collect points in, for example: Scene_7 (enter "d" to use default: Scene_7): ')
    if scene_name == 'd':
        scene_name = 'Scene_7'
    collection_filename = scene_name + '-root.json'
    collection_path = os.path.join(root_dir, collection_filename)

    raw_input('>>> please pick about 8~40 points in each room in average, at least 4 points in a room (press any key to continue)')

    # remind user to open the unity exe file
    unity_exe_filename = scene_name + '.x86_64'
    raw_input('>>> remember to open unity exe file in {} (press any key to continue)'
            .format(unity_exe_filename))
    recorder = Recorder(collection_path, scene_name)

    settings = termios.tcgetattr(sys.stdin)
    keystate = [0,0,0,0]

    print("use 'wsad' to control the robot, ' ' to save image locaiton, 'n' to change tag, 'q' to quit, 'h' to get information, 'c' to change speed")
    print("start taking points!!\n")
    recorder.change_tag()
    sample_point_count = 0
    while True:
        key = getKey()
        if key in keyMoves:
            idx = keyMoves.index(key)
            keystate[idx] = 1 - keystate[idx]
            if keystate[idx]:
                print('Keyboard Response: key down {}'.format(key))
                recorder.con.KeyDown(key)
            else:
                print('Keyboard Response: key up {}'.format(key))
                recorder.con.KeyUp(key)
        elif key in keyUtils:
            if key==' ': # start recording
                print('Keyboard Response: save location as sampling point')
                recorder.record_position()
            elif key=='n': # change tag
                print('Keyboard Response: change tag')
                recorder.change_tag()
            elif key=='q': # end recording
                print('Keyboard Response: quit')
                break
            elif key=='c': # set speed
                print('Keyboard Response: change speed')
                speed = float(raw_input('set speed to (default is 2.0): '))
                rotate_speed = float(raw_input('set rotation speed to (default is 40.0): '))
                recorder.con.setSpeed(speed)
                recorder.con.setRotateSpeed(rotate_speed)
            elif key=='h': # get help
                print('Keyboard Response: help')
                print("use 'wsad' to control the robot, ' ' to save image locaiton, 'n' to change tag, 'q' to quit, 'h' to get information, 'c' to change speed")
                continue
            else:
                pass
        else:
            pass
    recorder.con.close()
    recorder.save()
    print "program successfully quit"
