# импортирование необходимых библиотек
import numpy as np
import cv2
import imutils
import pytesseract as pytesseract
from PIL import Image
from skimage.filters import threshold_local
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
# ----------------------------------------------------------------------------------------------------------------------

# параметр для сканируемого изображения
args_image = "page0.png"
# args_image = "test1.jpg"

# прочитать изображение
image = cv2.imread(args_image)
image = cv2.resize(image, (0,0), fx=0.5, fy=0.5)
orig = image.copy()

# показать исходное изображение
cv2.imshow("Original Image", image)
cv2.waitKey(0)  # press 0 to close all cv2 windows
cv2.destroyAllWindows()

# ----------------------------------------------------------------------------------------------------------------------

# конвертация изображения в градации серого. Это уберёт цветовой шум
grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# размытие картинки, чтобы убрать высокочастотный шум
# это помогает определить контур в сером изображении
grayImageBlur = cv2.blur(grayImage, (3, 3))

# теперь производим определение границы по методу Canny
edgedImage = cv2.Canny(grayImageBlur, 10, 20)

# показать серое изображение с определенными границами
cv2.imshow("gray", grayImage)
cv2.imshow("grayBlur", grayImageBlur)
cv2.imshow("Edge Detected Image", edgedImage)
cv2.waitKey(0)  # нажать 0, чтобы закрыть все окна cv2
cv2.destroyAllWindows()

# ----------------------------------------------------------------------------------------------------------------------
hsv_min = np.array((0, 0, 200), np.uint8)
hsv_max = np.array((180, 255, 255), np.uint8)

scanGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
T = threshold_local(scanGray,  51, offset=7, method="gaussian")
scanGray = (scanGray > T).astype("uint8") * 255

cv2.imshow("Original Image", scanGray)
cv2.waitKey(0)  # press 0 to close all cv2 windows
cv2.destroyAllWindows()
thresh = scanGray.copy()
# hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# thresh = cv2.inRange(hsv, hsv_min, hsv_max)
# ----------------------------------------------------------------------------------------------------------------------

# найти контуры на обрезанном изображении, рационально организовать область
# оставить только большие варианты

allContours = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
allContours = imutils.grab_contours(allContours)
# сортировка контуров области по уменьшению и сохранение топ-1
allContours = sorted(allContours, key=cv2.contourArea, reverse=True)[:2]
# aппроксимация контура
perimeter = cv2.arcLength(allContours[1], True)
ROIdimensions = cv2.approxPolyDP(allContours[1], 0.02 * perimeter, True)

# показать контуры на изображении
cv2.drawContours(image, [ROIdimensions], -1, (0, 255, 0), 2)

cv2.imshow("Contour Outline", image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# изменение массива координат
ROIdimensions = ROIdimensions.reshape(4, 2)

# список удержания координат ROI
rect = np.zeros((4, 2), dtype="float32")

# наименьшая сумма будет у верхнего левого угла,
# наибольшая — у нижнего правого угла
s = np.sum(ROIdimensions, axis=1)
rect[0] = ROIdimensions[np.argmin(s)]
rect[2] = ROIdimensions[np.argmax(s)]

# верх-право будет с минимальной разницей
# низ-лево будет иметь максимальную разницу
diff = np.diff(ROIdimensions, axis=1)
rect[1] = ROIdimensions[np.argmin(diff)]
rect[3] = ROIdimensions[np.argmax(diff)]

# верх-лево, верх-право, низ-право, низ-лево
(tl, tr, br, bl) = rect

# вычислить ширину ROI
widthA = np.sqrt((tl[0] - tr[0]) ** 2 + (tl[1] - tr[1]) ** 2)
widthB = np.sqrt((bl[0] - br[0]) ** 2 + (bl[1] - br[1]) ** 2)
maxWidth = max(int(widthA), int(widthB))

# вычислить высоту ROI
heightA = np.sqrt((tl[0] - bl[0]) ** 2 + (tl[1] - bl[1]) ** 2)
heightB = np.sqrt((tr[0] - br[0]) ** 2 + (tr[1] - br[1]) ** 2)
maxHeight = max(int(heightA), int(heightB))

# набор итоговых точек для обзора всего документа
# размер нового изображения
dst = np.array([
    [0, 0],
    [maxWidth - 1, 0],
    [maxWidth - 1, maxHeight - 1],
    [0, maxHeight - 1]], dtype="float32")

# вычислить матрицу перспективного преобразования и применить её
transformMatrix = cv2.getPerspectiveTransform(rect, dst)

# преобразовать ROI
scan = cv2.warpPerspective(orig, transformMatrix, (maxWidth, maxHeight))

if maxHeight < maxWidth:
    scan = cv2.rotate(scan, 0)
# давайте посмотрим на свёрнутый документ
# cv2.imshow("Scaned",scan)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# ----------------------------------------------------------------------------------------------------------------------

# конвертация в серый
scanGray = cv2.cvtColor(scan, cv2.COLOR_BGR2GRAY)

# показать финальное серое изображение
# cv2.imshow("scanGray", scanGray)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# ------------------------------


oneHeight = int(maxHeight / 100)
oneWidth = int(maxWidth / 100)

# Width = x
# Height = y
# ОФМС
passport = {}
croped = scanGray[oneHeight * 11:oneHeight * 16, oneWidth * 21:oneWidth * 90]
T = threshold_local(croped, 67, offset=25, method="gaussian")
scanBW = (croped > T).astype("uint8") * 255
ofmc = pytesseract.image_to_string(scanBW, lang='rus')

croped = scanGray[oneHeight * 16:oneHeight * 20, oneWidth * 8:oneWidth * 90]
T = threshold_local(croped, 67, offset=25, method="gaussian")
scanBW = (croped > T).astype("uint8") * 255
ofmc = ofmc + pytesseract.image_to_string(scanBW, lang='rus')

passport['ofms'] = ofmc.replace('\n\x0c', ' ').strip()

# ДАТА ВЫДАЧИ
croped = scanGray[oneHeight * 23:oneHeight * 27, oneWidth * 18:oneWidth * 39]
T = threshold_local(croped, 27, offset=15, method="gaussian")
scanBW = (croped > T).astype("uint8") * 255
passport['date_issue'] = pytesseract.image_to_string(scanBW, lang='rus').replace('\n\x0c', '')

# КОД ПОДРАЗДЕЛЕНИЯ
croped = scanGray[oneHeight * 23:oneHeight * 27, oneWidth * 56:oneWidth * 72]
T = threshold_local(croped, 25, offset=15, method="gaussian")
scanBW = (croped > T).astype("uint8") * 255
passport['code'] = pytesseract.image_to_string(scanBW, lang='rus').replace('\n\x0c', '')

# Фамилия
croped = scanGray[oneHeight * 56:oneHeight * 62, oneWidth * 41:oneWidth * 83]
T = threshold_local(croped, 41, offset=15, method="gaussian")
scanBW = (croped > T).astype("uint8") * 255
passport['last_name'] = pytesseract.image_to_string(scanBW, lang='rus').replace('\n\x0c', '')

# ИМЯ
croped = scanGray[oneHeight * 65:oneHeight * 69, oneWidth * 39:oneWidth * 88]
T = threshold_local(croped, 29, offset=15, method="gaussian")
scanBW = (croped > T).astype("uint8") * 255
passport['first_name'] = pytesseract.image_to_string(scanBW, lang='rus').replace('\n\x0c', '')

# ОТЧЕСТВО
croped = scanGray[oneHeight * 69:oneHeight * 73, oneWidth * 42:oneWidth * 89]
T = threshold_local(croped, 25, offset=15, method="gaussian")
scanBW = (croped > T).astype("uint8") * 255
passport['middle_name'] = pytesseract.image_to_string(scanBW, lang='rus').replace('\n\x0c', '')

# ПОЛ
croped = scanGray[oneHeight * 73:oneHeight * 76, oneWidth * 35:oneWidth * 43]
T = threshold_local(croped, 25, offset=15, method="gaussian")
scanBW = (croped > T).astype("uint8") * 255
passport['gender'] = pytesseract.image_to_string(scanBW, lang='rus').replace('\n\x0c', '')

# Дата рождения
croped = scanGray[oneHeight * 73:oneHeight * 77, oneWidth * 55:oneWidth * 90]
T = threshold_local(croped, 41, offset=12, method="gaussian")
scanBW = (croped > T).astype("uint8") * 255
passport['birthday'] = pytesseract.image_to_string(scanBW, lang='rus').replace('\n\x0c','')

# Номер
croped = scanGray
croped = cv2.rotate(croped, 2)
croped = croped[oneHeight * 3:oneHeight * 7, oneWidth * 22:oneWidth * 60]
T = threshold_local(croped, 25, offset=23, method="gaussian")
scanBW = (croped > T).astype("uint8") * 255
passport['number'] = pytesseract.image_to_string(scanBW, lang='rus').replace('\n\x0c','')

for k in passport.keys():
    print('{0} - {1}'.format(k, passport[k]))
cv2.imshow("scanBW", scanBW)
cv2.waitKey(0)
cv2.destroyAllWindows()