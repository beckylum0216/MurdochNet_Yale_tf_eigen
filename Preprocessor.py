import  cv2
import numpy as np

class ProcessImage(object):
    def __init__(self, imgWidth, imgHeight, rawImage):
        self.a = 0
        self.b = 0
        self.c = self.a + imgWidth
        self.d = self.b + imgHeight
        self.width = imgWidth
        self.height = imgHeight
        self.image = rawImage


    def DetectFace(self):

        face_cascade = cv2.CascadeClassifier("./haar_cascade/haarcascade_frontalface_default.xml")
        self.faces = face_cascade.detectMultiScale(self.image, 1.05, 3)



        for x, y, w, h in self.faces:
            cv2.rectangle(self.image, (x, y), (x + w, y + h), (0, 255, 255), 2)
            print("x: ", x, " y: ", y, " w: ", w, " h: ",h)




        #cv2.imshow("detected face", self.image)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        return self.faces



    def CropImage(self):
        self.croppedImg = np.empty([self.faces[0][2], self.faces[0][3]], np.uint8)
        self.width = self.faces[0][2]
        self.height = self.faces[0][3]

        for ii in range(self.faces[0][2]):
            for jj in range(self.faces[0][3]):
                self.croppedImg[ii][jj] = self.image[ii + self.faces[0][1]][jj + self.faces[0][0]]
                #print("cropped image: ", croppedImg[ii][jj])

        #cv2.imshow("cropped image", self.croppedImg)

        return self.croppedImg

    def ScaleImage(self, targetImage, targetWidth, targetHeight):

        dimension = (targetWidth, targetHeight)

        resizedImg = cv2.resize(targetImage, dimension, interpolation=cv2.INTER_AREA)

        # cv2.imshow("resized img", resizedImg)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return resizedImg

    def ApplyGaborFilter(self, targetImage):
        gabor_filter = cv2.getGaborKernel((self.width, self.height), 8.0, np.pi / 4, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        self.gaborImg = cv2.filter2D(targetImage, cv2.CV_8UC3, gabor_filter)

        #cv2.imshow("cropped gabor: ", self.gaborImg)

        return self.gaborImg

    def ApplyEigenFilter(self, targetImage):
        eigenMean, eigenVectors, eigenValues = cv2.PCACompute2(targetImage, mean=None)

        return eigenMean, eigenVectors, eigenValues
