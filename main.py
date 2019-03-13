import numpy as np
import cv2
from PIL import  Image
import ImageHeader
import Utility
import os
from ImageSet import ImageSet
from Monolithic import NeuralNet
from Preprocessor import ProcessImage

def main():

    imgHdr = ImageHeader.ImageHeader()
    imgHdr.maxImages = 165
    imgHdr.imgWidth = 320
    imgHdr.imgHeight = 243

    ut = Utility.Utility()
    dirname = "./yale/yalefaces"
    filename = ut.readFileLabel(dirname)
    pcaFaces = np.empty((165,160*160))

    for ii in range(len(filename)):
        img = ut.readGIFFile("./Yale/yalefaces/" + filename[ii], imgHdr)
        imgprocess = ProcessImage(imgHdr.imgWidth, imgHdr.imgHeight, img)
        imgprocess.DetectFace()
        croppedImg = imgprocess.CropImage()
        croppedPath = "./Yale/croppedfaces/" + filename[ii] + ".jpg"
        ut.SaveImageFile(croppedPath, croppedImg)
        scaledImg = imgprocess.ScaleImage(croppedImg, 160, 160)
        pcaFaces[ii] = scaledImg.flatten()

    preprocess = ProcessImage(imgHdr.imgWidth, imgHdr.imgHeight, pcaFaces)
    eigenMean, eigenFaces, eigenValues = preprocess.ApplyEigenFilter(pcaFaces)

    flattenedValue = 160 * 160

    eigenNew = np.empty((165, flattenedValue))

    for ii in range(165):
        for jj in range(flattenedValue):
            eigenNew[ii][jj] = eigenValues[0] * (pcaFaces[ii][jj] - eigenMean[0][jj])
            #print("eigenface value: ", eigenMean[0][jj] + eigenFaces[ii][jj])

    for jj in range(len(filename)):
        filePath = "./Yale/eigenfaces/" + filename[jj] + ".jpg"
        ut.SaveImageFile(filePath, eigenNew[jj].reshape(160,160))

    print("filenames: ", filename)
    dirname = "./Yale/eigenfaces/"
    eigenname = ut.readFileLabel(dirname)
    randomImg = ut.RandomImg(eigenname)
    file70, file20, file10 = ut.McCallRuleWrap(eigenname)
    pareto90, pareto10 = ut.ParetoRule(eigenname)


    train_images = ImageSet()
    #print(test_images.GetImageCount())
    train_images.LoadFromList(file70, 'Yale/eigenfaces')
    #print(test_images.GetImageCount())
    imgs = train_images.GetImageRange(range(0, train_images.GetImageCount()))

    test_images = ImageSet()
    test_images.LoadFromList(file10, 'Yale/eigenfaces')
    testImg = test_images.GetImageRange(range(0, test_images.GetImageCount()))


    save_base = os.path.join('.', 'saves')
    nt = NeuralNet(imgs['data'].shape[1],
                   train_images.GetUniqueLabelCount())
    for epoch in [1600,1200,800,400]:
        matrixPath = "confusion/confusion" + str(epoch) + "-mccall10-4-eigen.txt"
        reportPath = "classification/classreport" + str(epoch) + "-mccall10-4-eigen.txt"
        save_path = os.path.join(save_base, f'save_{str(epoch).zfill(6)}')
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        saved_path = nt.CheckNet( testImg, imgs, epoch, os.path.join(save_path, 'yale'), matrixPath, reportPath)

        print(saved_path)




    # meta = './saves/save_001600/yale.meta'
    # path = './saves/save_001600'
    # nt = NeuralNet(testImg['data'].shape[1], test_images.GetUniqueLabelCount())
    # #nt.InitialiseSession()
    # nt.RestoreState(meta)
    # nt.TestNet(testImg, path)

    # cv2.imshow("GIF Image", img)


    ut.DisplayImage(file10, "./Yale/eigenfaces")

    #print("Display with overlay")

    #ut.DisplayWithOverlay(file20[5], "./Yale/croppedfaces", "premultiplytest.tga", "./assets")

if __name__ == '__main__':
    main()
