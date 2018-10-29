Steps to collect data:
# we have about 15 Scenes
# for each Scene, we have about 5 settings
# for each Scene, we have some numbers of rooms, room id named as: SceneX_bathroom, so that different setting will have same id
# for each room, we have some numbers of nodes, node id named as: SceneX_1
# for each setting in same scene, they share same room id and node id

# after created, each Scene will have a Scene_0-root.json
# each settings will have a Scene_0_1-collection.txt

1. run "create_position_collection.py" on each Scene to collect Scene_0-root.json (root collection for Scene_0)
##IMPORTANT: after running 1, you should copy one root.json to a dataset specific for node, ex: room_data_node

## for room_data_continuous (_omni, _mono, _quad)
2. run "create_position_sampling_omni.py" on each setting to collect Scene_0_1-collection.txt and create Scene_0_1 with subdirectory and images inside
3. so do other sensors

3-1. run "collect_position_sampling_with_location.py" on the whole "room_data_continuous_omni" dataset, it will read a leftout_CG1_location.txt to split it the whole dataset into two, one for train and one for test. It will collect also the location information

## for room_data_node (_omni, _mono, _quad)
4. run "create_node_sampling_omni.py" on each setting to collect Scene_0_1-collection.txt and create Scene_0_1 with subdirectory and images inside
5. so do other sensors


