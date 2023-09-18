#Программный модуль на Python, позволяющий определить погрешность изображения, размещенного на фотографии, согласно параметрам длины, ширины и площади изображения 
import numpy as np
from skimage import io
from matplotlib import pyplot as plt

def show_images(in_img_list, in_img_path_list):
    f, ax = plt.subplots(1, len(in_img_list), figsize=(13, 5), squeeze=True)    
    for index, img_zip in enumerate(zip(in_img_list, in_img_path_list)):
        img = img_zip[0]
        img_name = img_zip[1]
        ax[index].axis("off")
        ax[index].imshow(img, cmap='gray')
        ax[index].set_title(f"{index} ({img_name})")
    plt.show()

print("Исходные изображения и вид сбоку")
img_path_list = ["_images/_CalibratePic.jpg", "_images/_TestPic.jpg", "_images/_ProfilePic.jpg"]
img_list = [io.imread(img_path) for img_path in img_path_list]
show_images(img_list, img_path_list)

## Параметры калибровки камеры
# Измеренные стороны прямоугольного объекта (𝑑𝑋 и 𝑑𝑌)
REAL_CALIBRATION_WIDTH = 7.5   # см
REAL_CALIBRATION_HEIGHT = 11.2 # см

# Измеренные стороны (высота и ширина) в пикселях (𝑑𝑥 и 𝑑𝑦)
IMG_CALIBRATION_WIDTH = 192    # px
IMG_CALIBRATION_HEIGHT = 284   # px

# Измеренное расстояние от камеры до объекта (𝑑𝑍)
CAM_CALIBRATION_DISTANCE = 40  # см

# _1_ Описание класса камеры c внутренними (𝐾) и внешними ([𝑅|𝑡]) параметры камеры.
class Camera:
    def __init__(self):
        self.cx = 0                             # Оптического центра по y
        self.cy = 0                             # Оптического центра по x
        self.fx = 0                             # Фокусное расстояние 𝑓 по x — расстояние между плоскостью изображения и центром камеры.
        self.fy = 0                             # Фокусное расстояние 𝑓 по y — расстояние между плоскостью изображения и центром камеры.
        self.K = np.array(0)                    # Калибровочная матрица
        
    def calibrate(self, w_px, h_px, w_sm, h_sm, dist_sm):
        self.fx = (w_px * dist_sm) / w_sm       # fx = (dx/dX)*dZ 
        self.fy = (h_px * dist_sm) / h_sm       # fy = (dy/dY)*dZ
        # Калибровочная матрица
        self.K = np.array([[self.fx, 0,       self.cx], 
                           [0,       self.fy, self.cy], 
                           [0,       0,       1]])
        
    # _2_ Метод для проецирования трехмерных точек на плоскость изображения.
    def get_roi(self, img, x, y, w, h):
        Yc, Xc, _ = img.shape
        return [x - Xc / 2, y - Yc / 2, w, h]
    
    # _3_ Метод, рассчитывающий трехмерные координаты точки в системе координат камеры, учитывая ее двухмерные координаты в плоскости изображения.
    def get_coordinates(self, rect):
        # np.matmul - Матричное произведение двух массивов; np.linalg.inv - Вычисление (мультипликативной) обратной величины матрицы;
        p0 = np.matmul(np.linalg.inv(self.K), np.array([[rect[0]],           [rect[1]],           [1]]))
        p1 = np.matmul(np.linalg.inv(self.K), np.array([[rect[0] + rect[2]], [rect[1]],           [1]]))
        p2 = np.matmul(np.linalg.inv(self.K), np.array([[rect[0]],           [rect[1] + rect[3]], [1]]))
        p3 = np.matmul(np.linalg.inv(self.K), np.array([[rect[0] + rect[2]], [rect[1] + rect[3]], [1]]))
        return [p0, p1, p2, p3]
    
    def get_real_size(self, coordinates, in_scale):
        calc_w = (coordinates[3][0] - coordinates[0][0]) * in_scale
        calc_h = (coordinates[3][1] - coordinates[0][1]) * in_scale
        return calc_w, calc_h

    def print_stats(self, in_calc_w, in_calc_h, in_real_w, in_real_h):
        print(f'Полученная ширина: {np.round(in_calc_w, 5)} см, реальная ширина: {np.round(in_real_w, 5)} см, погрешность: {np.round(in_real_w - in_calc_w, 5)}')
        print(f'Полученная длина: {np.round(in_calc_h, 5)} см, реальная длина: {np.round(in_real_h, 5)} см, погрешность: {np.round(in_real_h - in_calc_h, 5)}')
        print(f'Полученная площадь: {np.round(in_calc_w * in_calc_h, 5)} м², реальная площадь {np.round((in_real_w * in_real_h), 5)} м², погрешность: {np.round((in_real_w * in_real_h) - (in_calc_w * in_calc_h), 5)}м²')

# Инициализация класса камеры и калибровка камеры
camera = Camera()
camera.calibrate(w_px=IMG_CALIBRATION_WIDTH, h_px=IMG_CALIBRATION_HEIGHT, w_sm=REAL_CALIBRATION_WIDTH, h_sm=REAL_CALIBRATION_HEIGHT, dist_sm=CAM_CALIBRATION_DISTANCE)

# _4_ Получение фотографии плоского тест-объекта и измерить все необходимые параметры
CALIBRATION_IMG_START_X = 332 # px
CALIBRATION_IMG_START_Y = 544 # px
calibration_img = img_list[0]
coordinates = camera.get_coordinates(camera.get_roi(calibration_img, CALIBRATION_IMG_START_X, CALIBRATION_IMG_START_Y, IMG_CALIBRATION_WIDTH, IMG_CALIBRATION_HEIGHT))

w, h = camera.get_real_size(coordinates, CAM_CALIBRATION_DISTANCE)
print(f'\n**Калибровочное изображение**')
camera.print_stats(w, h, REAL_CALIBRATION_WIDTH, REAL_CALIBRATION_HEIGHT)

# _5_ Составление необходимых матриц, для расчета площади изображения и оценки погрешности.
IMG_TEST_WIDTH = 172   # px
IMG_TEST_HEIGHT = 280  # px
IMG_TEST_START_X = 323 # px
IMG_TEST_START_Y = 539 # px
CAM_TEST_DISTANCE = 37 # см
test_img = img_list[1]
coordinates = camera.get_coordinates(camera.get_roi(test_img, IMG_TEST_START_X, IMG_TEST_START_Y, IMG_TEST_WIDTH, IMG_TEST_HEIGHT))
w, h = camera.get_real_size(coordinates, CAM_TEST_DISTANCE)
print(f'\n**Тестовое изображение**')
camera.print_stats(w, h, 6, 10.8)