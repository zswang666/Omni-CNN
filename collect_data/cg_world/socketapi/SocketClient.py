import socket


class clientSock:
	def __init__(self, family=socket.AF_INET, protocol=socket.SOCK_STREAM):
		self.sock = socket.socket(family, protocol)
		self.isConnected = False

	def connect(self, ip='127.0.0.1', port=4567):
		self.server_addr = {'ip': ip, 'port': port}
		self.sock.connect((ip, port))
		self.isConnected=True

	def connect_self(self, port):
		self.sock.connect(('localhost', port))

	def send(self, message):
		self.sock.send(message)

	def recv(self, btn):
		self.recvData = b''
		while len(self.recvData) < btn:
			data = self.sock.recv(btn - len(self.recvData))
			if not data:
				return data
			self.recvData += data
		return self.recvData;

	def settimeout(self, timeout):
		self.sock.settimeout(timeout)

	def close(self):
		if self.isConnected:
			self.send((chr(0)+chr(0)).encode('ascii', 'replace'))
			self.sock.close()
		self.isConnected = False

	def __del__(self):
		self.close()