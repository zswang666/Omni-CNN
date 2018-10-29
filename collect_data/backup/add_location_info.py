import os
import json

if __name__ == '__main__':
    root_dir = raw_input('>>> the root directory of the data (press "d" to use default "/home/joehuang/Desktop/room_data_continuous"): ')
    if root_dir == 'd':
        root_dir = "/home/joehuang/Desktop/room_data_continuous"
    if not os.path.exists(root_dir):
        raise ValueError('root directory not exists')
    transfer_txt_filename = raw_input('>>> the txt file you want to add location information: ')
    transfer_txt_path = os.path.join(root_dir, transfer_txt_filename)
    if not os.path.exists(transfer_txt_path):
        raise ValueError('the transfer file not exists in the root directory')

    image_id_list = []
    node_id_list = []
    room_id_list = []
    image_path_list = []
    position_list = []
    angle_list = []
    room_id_to_json_data = {}

    with open(transfer_txt_path, 'r') as f:
        first_line = True
        for line in f:
            if first_line:
                first_line = False
                continue
            else:
                image_info = line[:-1].split(' ')
                image_id = int(image_info[0])
                node_id = int(image_info[1])
                room_id = int(image_info[2])
                image_path = str(image_info[3])
                image_filename = image_path.split('/')[-1]

                image_id_list.append(image_id)
                node_id_list.append(node_id)
                room_id_list.append(room_id)
                image_path_list.append(image_path)
                if not room_id in room_id_to_json_data.keys():
                    json_subdir = image_path.split('/')
                    json_subdir = os.path.join(json_subdir[0], json_subdir[1])
                    json_dir = os.path.join(os.path.abspath(root_dir), os.path.relpath(json_subdir))
                    files = os.listdir(json_dir)
                    for candidate in files:
                        if candidate[-5:] != '.json':
                            continue
                        elif 'raw' in candidate:
                            continue
                        else:
                            json_path = os.path.join(json_dir, candidate)
                            print("get json file {}".format(json_path))
                            with open(json_path, 'r') as json_file:
                                json_data = json.load(json_file)
                                room_id_to_json_data[room_id] = json_data
                            break
                json_data = room_id_to_json_data[room_id]
                index = json_data['image_path'].index(image_filename)
                image_position = json_data['position'][index]
                image_angle = json_data['angle'][index]
                position_list.append(image_position)
                angle_list.append(image_angle)
    location_txt_filename = transfer_txt_filename[:-4] + '_location.txt'
    location_txt_path = os.path.join(root_dir, location_txt_filename)
    with open(location_txt_path, 'w') as f:
        f.write('image_id room_id pos_x pos_y pos_z ang_x ang_y ang_z image_path\n')
        for index in range(len(image_id_list)):
            image_id = image_id_list[index]
            node_id = node_id_list[index]
            room_id = room_id_list[index]
            position = position_list[index]
            pos_x = position[0]
            pos_y = position[1]
            pos_z = position[2]
            angle = angle_list[index]
            ang_x = angle[0]
            ang_y = angle[1]
            ang_z = angle[2]
            image_path = image_path_list[index]
            f.write('{} {} {} {} {} {} {} {} {}\n'.format(image_id, room_id, pos_x, pos_y, pos_z, ang_x, ang_y, ang_z, image_path))

