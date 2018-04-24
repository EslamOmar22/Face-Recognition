import cv2
import numpy as np
import math
import skimage.feature as sk
import glob

number_of_images = 275
class FeaturesExtraction:

    def mean(self, n, X):
        s = 0
        for i in range(n):
           s += X[0][i]
        Mean = s/n
        return Mean

    def variance(self, mean, x, n):
        summ = 0
        for i in range(n):
            summ += pow((x[0][i]-mean), 2)
        var = summ/n
        return var

    def stander_deviation(self, var):
        SD = math.sqrt(var)
        return SD

    def smoothness (self, varience):
        smooth = 1-(1/(1+varience))
        return smooth

    def thirdmoment(self, s_deviation, x, mean, n):
        summ = 0
        for i in range(n):
            summ += pow((x[0][i]-mean), 3)
        temp = n * pow(s_deviation, 3)
        M3 = summ/temp
        return M3

    def fourthmoment(self, s_deviation, x, mean, n):
        summ = 0
        for i in range(n):
            summ += pow((x[0][i] - mean), 4)
        temp = n * pow(s_deviation, 4)
        M4 = summ / temp
        return M4-3

    def uniformty(self, gray_image, x):
        for i in range(256):
            U = 0
            z = x[0][i]
            hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256]).T
            U += pow(hist[0][z], 2)

        return U

    def entropy(self, gray_image, x):
        for i in range(255):
            e = 0
            z = x[0][i]
            hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256]).T
            if hist[0][z] == 0:
                e += 0
            else:
                e += (hist[0][z] * math.log(hist[0][z], 2))

        e *= -1
        return e

    def glcm(self, gray_image):
        gray = np.divide(gray_image, 255)
        gray *= 7
        glcm = sk.greycomatrix(gray.astype(int), [2], [0], levels=8, normed=True, symmetric=True)
        return glcm

    def contrast(self, glm):
        contr = 0
        for i in range(8):
            for j in range(8):
                contr += pow((i-j), 2)*glm[i][j]
        return contr

    def energy(self, glm):
        enrg = 0
        for i in range(8):
            for j in range(8):
                enrg += pow(glm[i][j], 2)
        return enrg

    def homogen(self, glm):
        homo = 0
        for i in range(8):
            for j in range(8):
                homo += glm[i][j]/(1 + abs(i-j))
        return homo

    def entrop(self, glm):
        ent = 0
        for i in range(8):
            for j in range(8):
                if glm[i][j] == 0:
                    ent += 0
                else:
                    ent += glm[i][j]*math.log(glm[i][j], 2)

        ent *= -1
        return ent

    def call_features(self, path):
        classes = 1
        j = 0
        result = np.zeros((number_of_images, 13))
        for c in range(28):
            path2 = path + str(classes) + "/*"
            for i in glob.glob(path2):
                 image = cv2.imread(i)
                 gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                 imge = cv2.resize(gray_image, (50, 50))
                 h, w = imge.shape[:2]
                 n = h * w
                 glc = self.glcm(imge)
                 X = np.reshape(imge, (1, n))
                 result[j][0] = self.mean(n, X)#mean
                 result[j][1] = self.variance(result[j][0], X, n)#var
                 result[j][2] = self.stander_deviation(result[j][1])#sd
                 result[j][3] = self.smoothness(result[j][1])#smooth
                 result[j][4] = self.thirdmoment(result[j][2], X, result[j][0], n)#third moment
                 result[j][5] = self.fourthmoment(result[j][2], X, result[j][0], n)#fourth
                 result[j][6] = self.uniformty(imge,X)#uniformty
                 result[j][7] = self.entropy(imge,X)#entropy
                 result[j][8] = self.contrast(glc)
                 result[j][9] = self.energy(glc)
                 result[j][10] = self.homogen(glc)
                 result[j][11] = self.entrop(glc)
                 result[j][12] = classes
                 print(j)
                 j += 1
            classes += 1
            np.savetxt("Features.csv", result, delimiter=",", fmt='%f')


if __name__ == '__main__':

    path = "/" # add detected faces path
    p = FeaturesExtraction()
    p.call_features(path)
