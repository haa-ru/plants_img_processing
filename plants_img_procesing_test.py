#this file is for test code

import numpy as np
import matplotlib.pyplot as plt
import plant_img_procesing

test = plant_img_procesing.PlantImgProcessor(imgPath='test.png')
img = test.maskingImg()

# 참고: 실행시 그린색이 몇십 그룹 이상으로 분산되어 있는 사진이면, 그룹핑해서 계산하기 때문에 몇분정도 시간이 소요될 수 있음
highPixelGroup= test.descendingGrop(resultObjectCount=2)
meanColor, hex = test.imgColorAverage()

print('----------test-result----------')
print('highPixelGroup: ',highPixelGroup) # [(14, 27681), (13, 17877)]
print('meanColor: ',meanColor) # [50 59 25]
print('hex: ',hex) # #323b19
print('-------------------------------')

plt.imshow(img)
plt.show()