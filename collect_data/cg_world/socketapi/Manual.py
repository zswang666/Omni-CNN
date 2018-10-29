# import controller
import GameController as gc

# Create game controller
# this will create socket and logwriter too by default setting
con = gc.Controller()

# if you want to modify the path of the logwriter
# you can use 'setLog' to change your log path
con.setLog('pyLog.log')

# use 'connect' to connect to unity
con.connect()  # defalut is set to ip=127.0.0.1 port=4567 timeout=10

# use 'KeyDown' 'KeyUp' 'KeyPress' to pass key event to unity
con.KeyPress('W')
con.KeyUp('A')
con.KeyDown('S')
con.KeyUp('S')

# use 'setSpeed' and 'setRotateSpeed' to pass setting to unity
con.setSpeed(10)  # or con.Speed(10)
con.setRotateSpeed(20)  # or con.RSpeed(20)

# use 'getFirstView' 'getThirdView' to get image from unity
# ::NOTICE:: the return image is encoded as PNG format
# you need to use 'decode' function in 'imgdecode' module to decode it to cvmat form
# or you can just write your own decode function like 'decode' which use 'np.fromstring' and 'cv2.imdecode'
import imgdecode as imd

data = con.getFirstView()

img = imd.decode(data)
#  or
arr = np.fromstring(data, up.uint8)
img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
# then you can use other cv2 function to do process to your image
cv2.imshow('test', img)



# if any error occerred then you can check log file

