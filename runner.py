from fastai.vision import *
import cv2 
from PIL import Image as Img
import os
import matplotlib.pyplot as plt
# os.system('ffmpeg -i test/an.mp4  -vsync 0 test/output%03d.png')

learn = load_learner('model')
learn.load('fin')
for a in os.listdir('test/'):
	if '.png' in a:
		img = open_image(f'test/{a}')
		print(learn.predict(img)[0])
		
		cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)