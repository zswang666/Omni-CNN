import SocketClient as SC
import LogWriter as logw
import struct
import select
import socket

class Controller(object):
	def __init__(self):
		self.controller = SC.clientSock()
		self.log = logw.LogWriter()

	def setLog(self, path):
		self.log.setLog(path)

	def cleanLog(self):
		self.log.clean()

	def connect(self, ip='127.0.0.1', port=4567, timeout=10):
		try:
			self.controller.settimeout(timeout)
			self.controller.connect(ip, port)
			self.log.Log('Connecting to server: ip = {0} / port = {1}'.format(ip, port))
		except socket.error as e:
			self.controller.close()
			self.log.Error(logw.SERVER_CONNECTION_ERROR.format(ip, port))
			self.log.Error(str(e))
			#self.log.Error('Some error occurred when try to connect to server: ip = {0} / port = {1}'.format(ip, port))
			

######### Key Event #########
	def KeyDown(self, keycode):
		try:
			self.send(request='Key', param=keycode, evt='Down')
			#self.log.Log('Event = {0} / Param = {1}'.format('KeyDown', keycode))
		except socket.error as e:
			self.controller.close()
			self.log.Error(logw.KEYEVENT_ERROR.format('KeyDown', keycode))
			#self.log.Error('Some error occurred when try to send KeyDown event with Key {0}'.format(keycode))
			self.log.Error(str(e))

	def KeyUp(self, keycode):
		try:
			self.send(request='Key', param=keycode, evt='Up')
			#self.log.Log('Event = {0} / Param = {1}'.format('KeyUp', keycode))
		except socket.error as e:
			self.controller.close()
			self.log.Error(logw.KEYEVENT_ERROR.format('KeyUp', keycode))
			#self.log.Error('Some error occurred when try to send KeyUp event with Key {0}'.format(keycode))
			self.log.Error(str(e))

	def KeyPress(self, keycode):
		try:
			self.send(request='Key', param=keycode, evt='Press')
			#self.log.Log('Event = {0} / Param = {1}'.format('KeyPress', keycode))

		except socket.error as e:
			self.controller.close()
			self.log.Error(logw.KEYEVENT_ERROR.format('KeyPress', keycode))
			#self.log.Error('Some error occurred when try to send KeyPress event with Key {0}'.format(keycode))
			self.log.Error(str(e))


######### Speed #########
	def setSpeed(self, speed):
		try:
			self.send(request='Speed' , param=speed)
			#self.log.Log('Event = {0} / Param = {1}'.format('SetSpeed', speed))
		except socket.error as e:
			self.controller.close()
			self.log.Error(logw.REQUEST_PARAM_ERROR.format('setSpeed', speed))
			#self.log.Error('Some error occurred when try to send setSpeed event with Speed {0}'.format(speed))
			self.log.Error(str(e))

	def Speed(self, speed):
		self.setSpeed(speed)

	def setRotateSpeed(self, speed):
		try:
			self.send(request='RSpeed' , param=speed)
			#self.log.Log('Event = {0} / Param = {1}'.format('SetRotateSpeed', speed))
		except socket.error as e:
			self.controller.close()
			self.log.Error(logw.REQUEST_PARAM_ERROR.format('setRotateSpeed', speed))
			#self.log.Error('Some error occurred when try to send setRotateSpeed request with Speed {0}'.format(speed))
			self.log.Error(str(e))

	def RSpeed(self, speed):
		self.setRotateSpeed(speed)


######### Camera #########
	def getFirstView(self):
		try:
			self.send(request='FPS')
			bt = self.controller.recv(4)
			(bt ,) = struct.unpack('i', bt)
			self.log.Log('Request for First Person View {0} bytes'.format(bt)) #log
			image = self.controller.recv(bt)
			self.log.Log('Receive First Person View {0} bytes'.format(bt)) #log
			return image
		except socket.error as e:
			self.controller.close()
			self.log.Error(logw.REQUEST_ERROR.format('getFirstCamera'))
			#self.log.Error('Some error occurred when try to send \'getFirstCamera\' request')
			self.log.Error(str(e))

		return -1

	def getDepth(self):
		try:
			self.send(request='Depth')
			bt = self.controller.recv(4)
			(bt, ) = struct.unpack('i', bt)
			self.log.Log('Request for Depth Map {0} bytes'.format(bt))
			image = self.controller.recv(bt)
			self.log.Log('Receive Depth Map {0} bytes'.format(bt))
			return image
		except socket.error as e:
			self.controller.close()
			self.log.Error(logw.REQUEST_ERROR.format('getDepth'))
			self.log.Error(str(e))

		return -1

	def getSpherical(self):
		try:
			self.send(request='Spherical')
			bt = self.controller.recv(4)
			(bt, ) = struct.unpack('i', bt)
			self.log.Log('Request for Spherical Camera {0} bytes'.format(bt))
			image = self.controller.recv(bt)
			self.log.Log('Receive Spherical Camera {0} bytes'.format(bt))
			return image
		except socket.error as e:
			self.controller.close()
			self.log.Error(logw.REQUEST_ERROR.format('getSpherical'))
			self.log.Error(str(e))

		return -1

	#def getThirdView(self):
	#	self.log.Log('Request for Third Person View') #log
	#	try:
	#		self.send('TPS')
	#		bt = self.controller.recv(4)
	#		(bt, ) = struct.unpack('i', bt)
	#		self.log.Log('Request for Third Person View {0} bytes'.format(bt)) #log
	#		image = self.controller.recv(bt)
	#		self.log.Log('Receive Third Person View {0} bytes'.format(bt))
	#		return image
	#	except socket.error as e:
	#		self.controller.close()
	#		self.log.Error('Some error occurred when try to send getThirdCamera event')
	#		self.log.Error(str(e))
	#	return 0


######### Position/Rotation #########
	def getPos(self):
		try:
			self.send(request='getPos')
			bt = self.controller.recv(12)
			(x,y,z) = struct.unpack('fff', bt)
			self.log.Log('Receive robot Position ({0}, {1}, {2})'.format(x,y,z))
			return x,y,z
		except socket.error as e:
			self.controller.close()
			self.log.Error(logw.REQUEST_ERROR.format('getPos'))
			self.log.Error(str(e))
		return -1

	def setPos(self, vec):
		try:
			self.setPos(vec[0], vec[1], vec[2])
		except IndexError:
			print('Index Error in function setPos(vec)\n')
			self.log.Error('Index Error in function setPos(vec)')
			print('\tvec must contain at least 3 numbers\n')
			self.log.Error('vec must contain at least 3 numbers')

	def setPos(self, x, y, z):
		try:
			self.send(request='setPos', param=[x, y, z])
		except socket.error as e:
			self.controller.close()
			self.log.Error(logw.REQUEST_ERROR.format('setPos'))
			self.log.Error(str(e))

	def getRot(self):
		self.log.Log('Request for robot Rotation')
		try:
			self.send(request='getRot')
			bt = self.controller.recv(12)
			(x,y,z) = struct.unpack('fff', bt)
			self.log.Log('Receive robot Rotation ({0}, {1}, {2})'.format(x,y,z))
			return x,y,z
		except socket.error as e:
			self.controller.close()
			self.log.Error(logw.REQUEST_ERROR.format('getRot'))
			self.log.Error(str(e))

	def setRot(self, vec):
		try:
			self.setRot(vec[0], vec[1], vec[2])
		except IndexError:
			print('Index Error in function setRot(vec)\n')
			self.log.Error('Index Error in function setRot(vec)')
			print('\tvec must contain at least 3 numbers\n')
			self.log.Error('vec must contain at least 3 numbers')

	def setRot(self, x, y, z):
		try:
			self.send(request='setRot', param=[x, y, z])
		except socket.error as e:
			self.controller.close()
			self.log.Error(logw.REQUEST_ERROR.format('setRot'))
			self.log.Error(str(e))

	def setRandPos(self):
		try:
			self.send(request='setRandPos')
		except socket.error as e:
			self.controller.close()
			self.log.Error(logw.REQUEST_ERROR.format('setRandPos'))
			self.log.Error(str(e))


######### Other Option #########

	def setTimeScale(self, scale):
		try:
			self.send(request='setTimeScale', param=scale)
		except socket.error as e:
			self.controller.close()
			self.log.Error(logw.REQUEST_ERROR.format('setTimeScale'))
			self.log.Error(str(e))

	# Error return -1
	def getTimeScale(self):
		try:
			self.send(request='getTimeScale')
			bt = self.controller.recv(4)
			(bt, ) = struct.unpack('f', bt)
			self.log.Log('Receive Time Scale {0}'.format(bt))
			return bt;
		except socket.error as e:
			self.controller.close()
			self.log.Error(logw.REQUEST_ERROR.format('getTimeScale'))
			self.logError(str(e))
		return -1

######### Sending Event #########
	def send(self, request, param='', evt=''):
		if request == 'Close': #00
			self.sendascii(chr(0)+chr(0))
		elif request == 'Key': #01 Down/02 Up/03 Press
			if evt == 'Down':
				self.sendascii(chr(0x01) + param)
			elif evt == 'Up':
				self.sendascii(chr(0x02) + param)
			else:  
				self.sendascii(chr(0x03) + param)
			
		elif request == 'Speed': #04
			self.sendascii(chr(0x04) + chr(0))
			self.controller.send(struct.pack('f', param))
		elif request == 'RSpeed': #05
			self.sendascii(chr(0x05) + chr(0))
			self.controller.send(struct.pack('f', param))
		elif request == 'getPos': #06
			self.sendascii(chr(0x06) + chr(0))
		elif request == 'FPS': #07
			self.sendascii(chr(0x07) + chr(0))
		elif request == 'getRot': #08
			self.sendascii(chr(0x08) + chr(0))
		elif request == 'setPos': #09
			self.sendascii(chr(0x09) + chr(0))
			self.controller.send(struct.pack('fff', param[0], param[1], param[2]))
		elif request == 'setRot': #0a
			self.sendascii(chr(0x0a) + chr(0))
			self.controller.send(struct.pack('fff', param[0], param[1], param[2]))
		elif request == 'Depth': #0b
			self.sendascii(chr(0x0b) + chr(0))
		elif request == 'setTimeScale': #0c
			self.sendascii(chr(0x0c) + chr(0))
			self.controller.send(struct.pack('f', param))
		elif request == 'getTimeScale': #0d
			self.sendascii(chr(0x0d) + chr(0))
		elif request == 'setRandPos': #0e
			self.sendascii(chr(0x0e) + chr(0))
		elif request == 'Spherical': #0f
			self.sendascii(chr(0x0f) + chr(0))
		elif request == 'S': #ff
			self.sendascii(chr(0xff) + chr(len(param)))
			self.controller.send(param.encode("UTF-8"))
			return;
		else: #ff
			self.log.Error('Sending Unknown request: {0} / param={1} / evt={2}'.format(request, param, evt))
			return;
		self.log.Log('Sending request: {0} / param={1} / evt={2}'.format(request, param, evt))

	def sendascii(self, message):
		self.controller.send(message.encode('ascii', 'replace'))

	def sendstr(self, message):
		self.send(request='S', param=message)
		self.log.Log('Sending message : ' + message) #log

	def close(self):
		self.controller.close()

	def __del__(self):
		self.close()
		self.log.close()
