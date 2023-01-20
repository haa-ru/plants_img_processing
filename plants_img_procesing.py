
import cv2
import numpy as np


# plant img processor class 
class PlantImgProcessor:
    def __init__(self, imgPath):
        bgrImg = cv2.imread(imgPath, cv2.IMREAD_COLOR)
        self.img = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)



# use the hsv color spectrum, 3channel eg. [25,40,40]
    def makingGreenColorMask(self):
        hsvImg = cv2.cvtColor(self.img, cv2.COLOR_RGB2HSV)

        lowerGreen = np.array([25,40,40])
        upperGreen = np.array([85,255,255])

        mask = cv2.inRange(hsvImg, lowerGreen, upperGreen)

        return mask



    def maskingImg(self):
        mask = self.makingGreenColorMask()
        maskedImg = cv2.bitwise_and(self.img, self.img, mask=mask)

        return maskedImg



    def imgColorAverage(self):
    
        maskedImg = self.maskingImg()
        allPixelCountInImg = maskedImg.shape[0] * maskedImg.shape[1]
        img1D = maskedImg.reshape(allPixelCountInImg,3)
        totalColor = [0,0,0]
        pixelCount = 0

        for onePixel in img1D:
            if str(onePixel) != '[0 0 0]':
                pixelCount += 1
                totalColor += onePixel

        meanColor = totalColor / pixelCount
        meanColorInt = meanColor.astype(int)

        r,g,b = meanColorInt
   
        hex = '#{:02x}{:02x}{:02x}'.format(r,g,b)

        return meanColorInt, hex



    def descendingGrop(self, resultObjectCount=1):
        resultObjectCount += 1 # to not include the 0th group
        binaryImg = self.makingGreenColorMask()

        groupCount, labeledImage, pixelInfo, notused  = cv2.connectedComponentsWithStats(binaryImg)
        desendListOfTupleGrops = []

        for index in range(groupCount):
            countPixels = pixelInfo[index][4]
            indexNum = index


            if len(desendListOfTupleGrops) != 0:
                IndexTuple = (indexNum, countPixels)

                # sort
                for comparedLocation in range(len(desendListOfTupleGrops)):
                    if desendListOfTupleGrops[comparedLocation][1] < IndexTuple[1]:
                        desendListOfTupleGrops.insert(comparedLocation, IndexTuple)
                        break
                    elif comparedLocation == len(desendListOfTupleGrops)-1:
                        desendListOfTupleGrops.append(IndexTuple)
                    else :
                        continue
            else :
                desendListOfTupleGrops.append((indexNum, countPixels))

        desendListOfTupleGrops = desendListOfTupleGrops[1:resultObjectCount] # 0th is backgrond pixel grop
        return desendListOfTupleGrops



    def boxing(self): ## todo: boxing, 이전데이터와 비교해서 더 확장되거나 확장이 되다가 합쳐지는 경우를 구분
        return None

