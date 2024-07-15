import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

def find_angle(image):
    dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
    magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX)
    magnitude_spectrum = np.uint8(magnitude_spectrum)

    magnitude_spectrum = cv2.resize(magnitude_spectrum, (720,720))
    magnitude_spectrum = cv2.GaussianBlur(magnitude_spectrum, (7, 7), 3)

    binary_spectrum = cv2.adaptiveThreshold(cv2.bitwise_not(magnitude_spectrum), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                cv2.THRESH_BINARY, 11, 2)
    binary_spectrum = cv2.medianBlur(binary_spectrum, 5)
    
    # kernel = np.ones((5, 5), np.uint8)
    # binary_spectrum = cv2.erode(binary_spectrum, kernel, iterations=1)

    binary_spectrum = cv2.bitwise_not(binary_spectrum)
   
    linesP = cv2.HoughLinesP(binary_spectrum, 3,  np.pi / 180, 300, minLineLength=10, maxLineGap=50)

    white_image = np.ones_like(binary_spectrum) * 255

    # Desenhar as linhas detectadas na imagem branca
    angles = []
    for line in linesP:
        x1, y1, x2, y2 = line[0]
        cv2.line(white_image, (x1, y1), (x2, y2), (0, 255, 0), 2)   
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if angle > 45:
            angle -= 90
        elif angle < -45:
            angle += 90

        angles.append(angle)

    mean = np.mean(angles)
    std = np.std(angles)
    threshold = 1
    outlier = []
    if float(std) != 0:
        for j,i in enumerate(angles):
            z = (i-mean)/std
            if abs(z) > threshold:
                outlier.append(j)
    else:
        pass

    without_outliers = [tupla for j, tupla in enumerate(angles) if j not in outlier]

    return np.mean(without_outliers), magnitude_spectrum, binary_spectrum, white_image 

if __name__ == '__main__':
    image = cv2.imread('cemig1.png', cv2.IMREAD_GRAYSCALE)

    image = rotate_image(image, 10)

    angle, magnitude_spectrum, binary_spectrum, white_image = find_angle(image)
    print(angle)
    cv2.imwrite('Espectro de Frequencia.jpg', magnitude_spectrum)
    cv2.imwrite('Espectro de Frequencia Binarizado.jpg', binary_spectrum)
    cv2.imwrite('Espectro.jpg', white_image)

