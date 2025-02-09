import cv2
import numpy as np
import os
import sys
#import matplotlib.pyplot as plt

def load_image(path):
    return cv2.imread(path)

#kropujem sliku, da se fokusiram na deo sa objektima
def preprocess_image(image):
    height, width = image.shape[:2]
    crop_y1, crop_y2 = int(height * 0.2), int(height * 0.7)  #kropujem samo po vertikali, horizontala mi ne treba
    #crop_x1, crop_x2 = int(width * 0.05), int(width * 0.95)
    cropped_image = image[crop_y1:crop_y2, 0:width]
    return cropped_image

#kreiram maske za prepoznavanje likova
def create_character_mask(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # konvertovanje slike u hsv
    
    #crvena (toad) --> 2 maske (oko 0 i oko 150-180)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([2, 255, 255])
    lower_red2 = np.array([150, 100, 100])
    upper_red2 = np.array([200, 210, 240])
    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)

    #boo (belo-siva kombinacija)
    lower_white_gray = np.array([0, 0, 210])
    upper_white_gray = np.array([180, 20, 255])
    mask_white_gray = cv2.inRange(hsv, lower_white_gray, upper_white_gray)

    #bobomb maska (crno-zuta kombinacija)
    lower_dark = np.array([0, 0, 0])
    upper_dark = np.array([180, 255, 40])
    lower_yellow = np.array([20, 150, 100])
    upper_yellow = np.array([39, 180, 180])
    mask_dark_yellow = cv2.inRange(hsv, lower_dark, upper_dark) | cv2.inRange(hsv, lower_yellow, upper_yellow)

    #kombinujem maske
    combined_mask = cv2.bitwise_or(mask_red, mask_white_gray)
    combined_mask = cv2.bitwise_or(combined_mask, mask_dark_yellow)
    
    #zatvaranje
    kernel_close = np.ones((5, 5), np.uint8)
    #opened_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_open, iterations=0)
    closed_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_close, iterations=3)
    
    return closed_mask

#prilagodjen kod sa vežbi
def watershed_segmentation(image, mask):

    kernel = np.ones((3, 3), np.uint8)
    #opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=0)

    sure_bg = cv2.dilate(mask, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 0)  #0,3,5; najb rezultati sa 0

    ret, sure_fg = cv2.threshold(dist_transform, 0.55 * dist_transform.max(), 255, 0)  #(0.7-0.5)
    #treshold 0.55 najb rez
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    #markiranje
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    #primena
    markers = cv2.watershed(image, markers)
    unique_colors = len(np.unique(markers)) - 2 
    return unique_colors, markers

def run_and_display(image_path, actual_count):
    image = load_image(image_path)
    cropped_image = preprocess_image(image)
    mask = create_character_mask(cropped_image)

    count, markers = watershed_segmentation(cropped_image, mask)

    # prikazivanje koraka obrade slike (zakomentarisano)
    # fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    # axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # axs[0].set_title("Original Image")
    # axs[1].imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
    # axs[1].set_title("Cropped Image")
    # axs[2].imshow(mask, cmap="gray")
    # axs[2].set_title("Binary Mask (Characters Only)")
    # axs[3].imshow(markers, cmap="jet")
    # axs[3].set_title(f"Detected Characters = {count}")
    # for ax in axs:
    #     ax.axis("off")
    # plt.show()

    return count

def calculate_mae(image_dir, object_counts_csv):
    image_paths = [os.path.join(image_dir, f'picture_{i + 1}.png') for i in range(len(object_counts_csv))]
    predicted_counts = []

    for i, image_path in enumerate(image_paths):
        predicted_count = run_and_display(image_path, object_counts_csv[i])
        predicted_counts.append(predicted_count)
    
    mae = np.mean(np.abs(np.array(predicted_counts) - np.array(object_counts_csv)))
    print(mae)

#Predicted Counts: [10, 24, 15, 9, 3, 11, 3, 14, 17, 7]
#Mean Absolute Error (MAE): 0.5 

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("You must enter the path to the image folder!")
        sys.exit(1)

    image_directory = sys.argv[1]
    object_counts_csv = [10, 23, 16, 9, 3, 10, 3, 13, 17, 6]

    calculate_mae(image_directory, object_counts_csv)

