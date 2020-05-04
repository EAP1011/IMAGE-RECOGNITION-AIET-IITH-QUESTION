import cv2
import plotly.graph_objs as go
from plotly.offline import iplot
import numpy as np
import matplotlib.pyplot as plt
indteambgr=cv2.imread("indteam.jpg")
indteam_gray=cv2.cvtColor(indteambgr,cv2.COLOR_BGR2GRAY)
haar_cascade_face=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
faces_rects=haar_cascade_face.detectMultiScale(indteam_gray,scaleFactor=1.2,minNeighbors=5);
i=0
p=list()
x,s1=plt.subplots(3,len(faces_rects)//3)
for (x, y, w, h) in faces:
  r = max(w, h) / 2
  centerx = x + w / 2
  centery = y + h / 2
  nx = int(centerx - r)
  ny = int(centery - r)
  nr = int(r * 2)
  faceimg = indteambgr[ny:ny+nr, nx:nx+nr]
  lastimg = cv2.resize(faceimg, (32, 32))
  i += 1
  p.append(np.array(lastimg,dtype=np.int16))
a=0
for x in range(0,3):
  for y in range(0,(len(faces_rects)//3)):
    s1[x][y].imshow(p[a])
    a=a+1

