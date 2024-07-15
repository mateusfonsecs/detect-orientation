import numpy as np
import cv2
from src.tf import find_angle

def spin(img, angle):
    bgr = cv2.split(img)
    contagem_b, contagem_g, contagem_r = np.bincount(bgr[0].ravel(), minlength=256), np.bincount(bgr[1].ravel(), minlength=25), np.bincount(bgr[2].ravel(), minlength=256)

    valor_predominante = (int(np.argmax(contagem_b)), int(np.argmax(contagem_g)), int(np.argmax(contagem_r)))
    altura, largura = img.shape[:2]
    nova_altura, nova_largura = altura * 1.2, largura * 1.2
    dx, dy = int((nova_largura - largura) / 2), int((nova_altura - altura) / 2)
    img = cv2.copyMakeBorder(img, dy, dy, dx, dx, cv2.BORDER_CONSTANT, value=valor_predominante)
    
    center = tuple(np.array(img.shape[1::-1]) / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    image = cv2.warpAffine(img, M, img.shape[1::-1], flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cv2.imwrite(f"results_2{angle}.png", image)
    
    return image

def generate_angles(initial_angle, final_angle, step_angle):
    angles = []

    # Gerando os ângulos no intervalo correto
    current_angle = initial_angle
    while current_angle <= final_angle:
        angles.append(current_angle)
        current_angle += step_angle

    # Convertendo a lista de ângulos em um array numpy, se desejado
    angles_array = np.array(angles)
        
    return angles_array

def generate_image(image, angles):
    list_images = []
    for angle in angles:
        image_copy = image.copy()
        image_res = spin(image_copy, -angle)
        list_images.append(image_res)

    return list_images

def read_images(list_images, angles, list2 = None):

    list_res = []
    for i, image in enumerate(list_images):
        # cv2.imshow("Rotated Image", image)
        # cv2.waitKey(50)
        angle, _, _, _ = find_angle(image)
        list_res.append((angle, angles[i], abs(angle - angles[i])))

    if list2 is not None:
        for i, image in enumerate(list2):
            # cv2.imshow("Rotated Image", image)
            # cv2.waitKey(50)
            angle, _, _, _ = find_angle(image)
            list_res.append((angle, angles[i], abs(angle - angles[i])))

    return list_res

def process_res(list_res):
    sum_squared_error = 0
    sum_error = 0

    for dado in list_res:
        real_value, predicted_value, error = dado
        sum_squared_error += error ** 2  # Somando o quadrado do erro
        sum_error += error

    mean_squared_error = sum_squared_error / len(list_res)  # Média dos quadrados dos erros
    rmse = np.sqrt(mean_squared_error)  # Raiz quadrada da média dos quadrados dos erros
    
    mean_error = sum_error / len(list_res)  # Erro médio

    return rmse, mean_error

if __name__ == "__main__":
    image_dir = "cemig1.png"
    image_dir2 = "cemig2.jpg"
    image = cv2.imread(image_dir)
    image2 = cv2.imread(image_dir2)
    angles = generate_angles(-30, 30, 0.2)
    list_images_1 = generate_image(image, angles)
    list_images_2 = generate_image(image2, angles)
    list_res = read_images(list_images_1, angles, list_images_2)
    rmse, mean_error = process_res(list_res)
    print(rmse, mean_error)

