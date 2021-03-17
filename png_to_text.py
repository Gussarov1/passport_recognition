# импортирование необходимых библиотек
# import numpy as np
import os
import time

import cv2
import imutils
import pytesseract
from skimage.filters import threshold_local


# параметр для сканируемого изображения
args_image = "2.jpg"

# прочитать изображение
image = cv2.imread(args_image)
# image = cv2.rotate(image, rotateCode=0)
orig = image.copy()

# # показать исходное изображение
# cv2.imshow("Original Image", image)
# cv2.waitKey(0)  # press 0 to close all cv2 windows
# cv2.destroyAllWindows()

# конвертация изображения в градации серого. Это уберёт цветовой шум
grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# размытие картинки, чтобы убрать высокочастотный шум
# это помогает определить контур в сером изображении
grayImageBlur = cv2.blur(grayImage, (3, 3))

# теперь производим определение границы по методу Canny
edgedImage = cv2.Canny(grayImageBlur, 100, 100, 3)

# показать серое изображение с определенными границами
# cv2.imshow("gray", grayImage)
# cv2.imshow("grayBlur", grayImageBlur)
# cv2.imshow("Edge Detected Image", edgedImage)
# cv2.waitKey(0)  # нажать 0, чтобы закрыть все окна cv2
# cv2.destroyAllWindows()

# найти контуры на обрезанном изображении, рационально организовать область
# оставить только большие варианты


allContours = cv2.findContours(edgedImage.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

allContours = imutils.grab_contours(allContours)

# сортировка контуров области по уменьшению и сохранение топ-1
# allContours = sorted(allContours, key=cv2.contourArea, reverse=True)[:1]

# # aппроксимация контура
perimeter = cv2.arcLength(allContours[0], True)
ROIdimensions = cv2.approxPolyDP(allContours[0], 0.02 * perimeter, True)
# показать контуры на изображении
cv2.drawContours(image, [ROIdimensions], -1, (0, 255, 0), 2)
cv2.imshow("Contour Outline", image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# # изменение массива координат
# # ROIdimensions = ROIdimensions.reshape(4, 2)
#
# # список удержания координат ROI
# rect = np.zeros((4, 2), dtype="float32")
#
# # наименьшая сумма будет у верхнего левого угла,
# # наибольшая — у нижнего правого угла
# s = np.sum(ROIdimensions, axis=1)
# rect[0] = ROIdimensions[np.argmin(s)]
# rect[2] = ROIdimensions[np.argmax(s)]
#
# # верх-право будет с минимальной разницей
# # низ-лево будет иметь максимальную разницу
# diff = np.diff(ROIdimensions, axis=1)
# rect[1] = ROIdimensions[np.argmin(diff)]
# rect[3] = ROIdimensions[np.argmax(diff)]
#
# # верх-лево, верх-право, низ-право, низ-лево
# (tl, tr, br, bl) = rect
#
# # вычислить ширину ROI
# widthA = np.sqrt((tl[0] - tr[0]) ** 2 + (tl[1] - tr[1]) ** 2)
# widthB = np.sqrt((bl[0] - br[0]) ** 2 + (bl[1] - br[1]) ** 2)
# maxWidth = max(int(widthA), int(widthB))
#
# # вычислить высоту ROI
# heightA = np.sqrt((tl[0] - bl[0]) ** 2 + (tl[1] - bl[1]) ** 2)
# heightB = np.sqrt((tr[0] - br[0]) ** 2 + (tr[1] - br[1]) ** 2)
# maxHeight = max(int(heightA), int(heightB))
#
# # набор итоговых точек для обзора всего документа
# # размер нового изображения
# dst = np.array([
#     [0, 0],
#     [maxWidth - 1, 0],
#     [maxWidth - 1, maxHeight - 1],
#     [0, maxHeight - 1]], dtype="float32")
#
# # вычислить матрицу перспективного преобразования и применить её
# transformMatrix = cv2.getPerspectiveTransform(rect, dst)
#
# # преобразовать ROI
# scan = cv2.warpPerspective(orig, transformMatrix, (maxWidth, maxHeight))
#
# # давайте посмотрим на свёрнутый документ
# cv2.imshow("Scaned", scan)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# #
# # конвертация в серый
# scanGray = cv2.cvtColor(scan, cv2.COLOR_BGR2GRAY)
#
# # показать финальное серое изображение
# cv2.imshow("scanGray", scanGray)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# ------------------------------
# конвертация в черно-белое с высоким контрастом для документов

# увеличить контраст в случае с документом
scanGray = grayImageBlur.copy()
T = threshold_local(scanGray, 61, offset=17, method="gaussian")
scanBW = (scanGray > T).astype("uint8") * 255

print(pytesseract.image_to_string(scanBW, lang='rus'))
# показать финальное изображение с высоким контрастом
cv2.imshow("scanBW", scanBW)
cv2.waitKey(0)
cv2.destroyAllWindows()
