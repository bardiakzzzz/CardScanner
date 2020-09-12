import cv2
import imutils
from imutils.perspective import four_point_transform
import numpy as np
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="Path to the image to be scanned")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])

# image = cv2.imread('FirstPart/photo_2019-01-26_11-49-40.jpg')
ratio = image.shape[0] / 500.0
orig = image.copy()
image = imutils.resize(image, height=500)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)

cv2.imshow("blur", gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

edged = cv2.Canny(gray, 50, 250)

cv2.imshow("Canny", edged)
cv2.waitKey(0)
cv2.destroyAllWindows()

lines = cv2.HoughLinesP(image=edged, rho=1, theta=np.pi / 180, threshold=100,
                        minLineLength=50, maxLineGap=50)

a, b, c = lines.shape
for i in range(a):
    cv2.line(edged, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (255, 255, 255), 2,
             cv2.LINE_AA)


cv2.imshow("Edged", edged)
cv2.waitKey(0)
cv2.destroyAllWindows()

contours = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)

contours = sorted(contours, key=cv2.contourArea, reverse=True)

for c in contours:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) == 4:
        screenCnt = approx
        break

cv2.drawContours(image, [screenCnt], -1, (255, 255, 0), 5)
cv2.imshow("Outline", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

new_contours = screenCnt.reshape(4, 2) * ratio

warped = four_point_transform(orig, new_contours)

cv2.imshow("Scanned", warped)
cv2.waitKey(0)
