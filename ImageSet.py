import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import re


class YaleImage:
    def __init__(self, image_array):
        self.caption = ''
        self.width = image_array.shape[0]
        self.height = image_array.shape[1]
        self.data = image_array

    def Label(self):
        # Convert a text string to an image label
        # Very dataset specific. This code is for the Yale Faces set

        # Search the filename for the label
        regex = re.compile(r'\d+')
        targetNum = regex.search(self.caption)
        #print("target num: ", targetNum)
        # Convert label to an integer
        num = int(targetNum.group())
        #print("check num: ", num)
        return(num)

    def LabelVector(self, labelCount):
        temp_encoder = np.zeros((1, labelCount))
        temp_encoder[0, self.Label() - 1] = 1

        return temp_encoder


class ImageSet:
    # All images must be same height and width
    # Currently assumes that the first image loaded sets these
    # Should add check to make sure this is true
    def __init__(self):
        self.images = []

    def GetImageCount(self):
        return len(self.images)

    def LoadImage(self, filename):
        # Load a single image from a file.
        frame = Image.open(filename).convert('L')
        # Convert to greyscale
        imgData = np.true_divide(np.array(frame), 255)
        imgObj = YaleImage(imgData)
        imgObj.caption = filename
        self.images.append(imgObj)

    def LoadFromDir(self, dirname):
        # Load a set of images from a directory.
        # Assumes that every file is a valid image.
        for root, dirs, files in os.walk(dirname):
            for filename in files:
                self.LoadImage(os.path.join(root, filename))

    def PlotImage(self, image_index):
        image = self.images[image_index]
        fig = plt.figure()
        plt.imshow(image.data, cmap="gray")
        plt.axis('off')
        caption = f'{image.caption}\n{str(image.Label())}'
        fig.text(.5, .05, caption, ha='center')
        plt.show()

    def GetUniqueLabelCount(self):
        # Collect data on unique labels
        uniqueLabels = {}
        for img in self.images:
            imgLabel = img.Label()
            if imgLabel in uniqueLabels:
                uniqueLabels[imgLabel] += 1
            else:
                uniqueLabels[imgLabel] = 1
        return len(uniqueLabels)

    def GetImageRange(self, indexes):
        # Get a range of images as a single tensorflow compatible array
        # Returns a dict
        # 'data': image data in a numImages x pixelsInImage array
        # 'labels': a label for each image in a numImages x numLabels array
        numLabels = self.GetUniqueLabelCount()
        print("num of unique:", numLabels)
        numImages = len(indexes)
        imgWidth = self.images[0].width
        imgHeight = self.images[0].height

        # Declare outputs
        imgArray = np.empty((numImages, imgWidth * imgHeight), np.uint8)
        lblArray = np.empty((numImages, numLabels))

        curImg = 0
        for imgIndex in indexes:
            # Reshape image data to match what tf expects
            img = self.images[imgIndex]
            imgData = img.data.flatten()
            imgArray[curImg] = imgData
            # Add label vector
            lblArray[curImg] = img.LabelVector(numLabels)
            curImg += 1

        return {'data': imgArray, 'labels': lblArray}

    def LoadFromList(self, filelist, dirname):
        # Load a set of images specified in a list
        # Assumes that every file is a valid image.
        for root, dirs, file in os.walk(dirname):
            for filename in filelist:
                self.LoadImage(os.path.join(root, filename))


    def GetRandomImages(self, indexes, numofclass):
        # Get a range of images as a single tensorflow compatible array
        # Returns a dict
        # 'data': image data in a numImages x pixelsInImage array
        # 'labels': a label for each image in a numImages x numLabels array
        numImages = len(indexes)
        imgWidth = self.images[0].width
        imgHeight = self.images[0].height

        # Declare outputs
        imgArray = np.empty((numImages, imgWidth * imgHeight), np.uint8)
        lblArray = np.empty((numImages, numofclass))

        curImg = 0
        for imgIndex in indexes:
            # Reshape image data to match what tf expects
            img = self.images[imgIndex]
            imgData = img.data.flatten()
            imgArray[curImg] = imgData
            # Add label vector
            lblArray[curImg] = img.LabelVector(numofclass)
            curImg += 1

        return {'data': imgArray, 'labels': lblArray}








