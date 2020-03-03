from fastai.vision import *
import cv2
from PIL import Image as Img
import os
import matplotlib.pyplot as plt
# os.system('ffmpeg -i test/an.mp4  -vsync 0 test/output%03d.png')

learn = load_learner('model')
learn.load('fin')
# for a in os.listdir('test/'):
# 	if '.png' in a:
# 		img = open_image(f'test/{a}')
# 		# print(learn.predict(img)[0])

# 		cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
vid = cv2.VideoCapture('test/an.mp4')
frame_width = int(vid.get(3))
frame_height = int(vid.get(4))
out = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc(
    'M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

while(vid.isOpened()):
    ret, frame = vid.read()
    # try:
    t = torch.tensor(np.ascontiguousarray(
        np.flip(frame, 2)).transpose(2, 0, 1)).float() / 255
    img = Image(t)
    p = learn.predict(img)[0]
    print(p)

    cv2.putText(frame, str(p), (20, 20), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 2, cv2.LINE_AA)
    out.write(frame)

    # except Exception as e:
    # 	print(e)
    #     break

out.release()
vid.release()