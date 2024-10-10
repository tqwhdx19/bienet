# -*- coding: utf-8 -*-

import os
import numpy as np
import cv2
import copy


def enhancement_strategy_building_integrity(input_path, input_label_path, output_aug_path):

    files = os.listdir(input_path)
    number = len(files)

    # enhance the building integrity
    alpha_1 = 1.2  # enhanced contrast ratio
    alpha_2 = 0.8  # diminished contrast ratio
    beta = 0   uminance factor
    gamma = 1  

    for index in range(number):
        name = files[index]

        print("index:", index, name)

        full_image_path = input_path + "/" + name
        full_label_path = input_label_path + "/" + name
        full_output_aug_path = output_aug_path + "/" + name

        img = cv2.imread(full_image_path)

        # 读取单通道图像
        gray_image = cv2.imread(full_label_path, cv2.IMREAD_GRAYSCALE)
        background = 255 - gray_image

        # Example: Thresholding (adjust threshold values based on your image)
        _, thresholded = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

        # Find contours of objects
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        image_foreground = np.zeros_like(img)

        # Calculate the average pixel value for each object
        for i, contour in enumerate(contours):
            # Create a mask for each object
            mask = np.zeros_like(gray_image)
            cv2.drawContours(mask, [contour], -1, 255, -1)

            # Apply the mask to the original image to extract the object
            masked_image = cv2.bitwise_and(img, img, mask=mask)
            masked_image_new = copy.deepcopy(masked_image)

            # Calculate the average pixel value of the object
            average_x, average_y, average_z = np.mean(masked_image[:, :, 0][mask != 0]), np.mean(
                masked_image[:, :, 1][mask != 0]), \
                                              np.mean(masked_image[:, :, 2][mask != 0])
            masked_image[mask != 0] = average_x, average_y, average_z

            masked_image_out = (masked_image / 2 + masked_image_new / 2).astype(np.uint8)
            # masked_image_out = (0.6 * masked_image + 0.4 * masked_image_new).astype(np.uint8)

            image_foreground = (image_foreground + masked_image_out).astype(np.uint8)
            # print(f"contour {i + 1}")

        # 将单通道图像转换为三通道的RGB图像
        color_background_image = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)

        # 将图像从BGR格式转换为RGB格式
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_background = img * ((color_background_image / 255).astype(np.uint8))
        f_enhanced = cv2.convertScaleAbs(image_foreground, alpha=alpha_1, beta=beta)
        b_diminished = cv2.convertScaleAbs(img_background, alpha=alpha_2, beta=beta)
        img_enhanced = f_enhanced + b_diminished

        img_enhanced = cv2.cvtColor(img_enhanced, cv2.COLOR_RGB2BGR)
        cv2.imwrite(full_output_aug_path, img_enhanced)


if __name__ == '__main__':

    input_path = "../train/image"
    input_label_path = "../train/label"
    output_aug_path = "../train/aug"

    enhancement_strategy_building_integrity(input_path, input_label_path, output_aug_path)
