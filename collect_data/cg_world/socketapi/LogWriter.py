import time
from datetime import datetime

import sys

SERVER_CONNECTION_ERROR = 'Some error occurred when try to connect to server: ip = {0} / port = {1}'
KEYEVENT_ERROR = 'Some error occurred when try to send {0} event with Key {1}'
REQUEST_ERROR = 'Some error occurred when try to send \'{0}\' request'
REQUEST_PARAM_ERROR = 'Some error occurred when try to send \'{0}\' request with param={1}'

class LogWriter:
	def __init__(self, path='pyLog.log'):
		self.filepath = path
		self.isOpened = False

	def open(self):
		try:
			if sys.version_info.major == 3:
				self.file = open(self.filepath, "a", encoding='UTF-8')
			else:
				self.file = open(self.filepath, "a")
		except (OSError, IOError) as e:
			print(str(e))

	def writeLog(self, lgtype, msg):
		if self.isOpened == False:
			self.open()
			self.isOpened = True
		try:
			msg = '{0} ::{1}:: {2}\n'.format(datetime.now(), lgtype, msg)
			self.file.write(msg)
			self.file.flush()
			if lgtype == 'ERROR':
				print(msg)
		except (OSError, IOError) as e:
			if lgtype == 'ERROR':
				print(msg)
			print(str(e))

	def clean(self):
		self.close()
		try:
			self.file = open(self.filepath, "w", encoding='UTF-8')
		except (OSError, IOError) as e:
			print(str(e))
	
	def Log(self, msg):
		self.writeLog('INFO', msg)


	def Error(self, msg):
		self.writeLog('ERROR', msg)


	def setLog(self, path):
		self.file.close()
		self.filepath = path

	def close(self):
		if self.isOpened:
			self.file.close()
		self.isOpened = False

	def __del__(self):
		self.close()
