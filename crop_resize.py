import cv2
from cv2 import destroyAllWindows
import numpy as np
file="C:/Users/sk503/Downloads/lambo.png"
img =cv2.imread(file)
cv2.imshow("Image",img)
print(img.shape)

if (cv2.waitKey(0) & 0xFF)==27: destroyAllWindows()