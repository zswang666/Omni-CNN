import os

def movephoto(DATA_DIR, number):
	count = 1
	sort = sorted(os.listdir(DATA_DIR))
	for filename in sort:
		if(count<=int(number)):
			print "Loading: %s" % filename
			os.system('mv ' + DATA_DIR + '/' + filename + ' ' + room_DIR)
			os.system('mv ' + room_DIR + '/' + filename + ' ' + room_DIR + '/' + str(count) + '.jpg')
			count = count+1
		else:
			break

DATA_DIR = '/media/james/0403-0201/capture'
end = 0
scene =raw_input('scene = ')
scene_DIR = '/home/joehuang/Desktop/' + scene
os.mkdir(scene_DIR)
while(end<1):
	room = raw_input('room = ')
	room_DIR = scene_DIR + '/' + room
	number = raw_input('number = ')
	os.mkdir(room_DIR)
	os.mkdir(room_DIR + '/equi')
	os.mkdir(room_DIR + '/cubic')
	movephoto(DATA_DIR, number)
	end = int(raw_input('continue? y = 0 or n = 1 '))


