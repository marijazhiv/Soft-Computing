import os
import sys
import cv2
import numpy as np
import pandas as pd

#load ground truth from CSV
def load_ground_truth(csv_path):
    return pd.read_csv(csv_path, index_col=0).iloc[:, 0].to_dict()

#preprocess video frames
def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return processed

#counting trucks bz finding contours
def count_trucks_in_frame(frame, visualize=False):
    processed_frame = preprocess_frame(frame)
    contours, _ = cv2.findContours(processed_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    count = 0

    # output_frame = frame.copy()

## Heuristic for truck-like shape; CHAT GPT
    for contour in contours:
        if cv2.contourArea(contour) > 230:  # filter based on area
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)    #odnos sirine i visine
            if 1.5 < aspect_ratio < 3.5: 
                count += 1
    #             if visualize:
    #                 # draw contour
    #                 cv2.drawContours(output_frame, [contour], -1, (0, 255, 0), 2)  # green
    #                 # draw bounding box
    #                 cv2.rectangle(output_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  #blue box
    #                 cv2.putText(output_frame, "", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # if visualize:
    #     cv2.imshow("Detected Trucks and Contours", output_frame)
    #     cv2.waitKey(1) 

    return count

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py <data_folder>")
        sys.exit(1)

    data_folder = sys.argv[1]
    if not os.path.isdir(data_folder):
        print(f"Error: {data_folder} is not a valid directory.")
        sys.exit(1)

    # load ground truth
    csv_path = os.path.join(data_folder, "counts.csv")
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} does not exist.")
        sys.exit(1)

    ground_truth = load_ground_truth(csv_path)

    # collect video files
    video_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith(".mp4")]
    if not video_files:
        print(f"No video files found in {data_folder}.")
        sys.exit(1)

    #analyze each video
    results = {}
    for video_file in video_files:
        cap = cv2.VideoCapture(video_file)  
        frame_count = 0
        truck_count = 0
        frame_index = 0 

        while cap.isOpened():
            ret, frame = cap.read()   #cita jedan po jedan frejm; kod sa vezbi
            if not ret:
                break

            if frame_index % 5 == 0:  # analyze every fifth frame
                truck_count += count_trucks_in_frame(frame, visualize=False)
                frame_count += 1

            frame_index += 1 

        cap.release()
        if frame_count > 0:
            average_trucks = truck_count / frame_count
        else:
            average_trucks = 0
        video_name = os.path.basename(video_file)
        results[video_name] = round(average_trucks)

    #cv2.destroyAllWindows()

    #MAE
    ground_truth_df = pd.DataFrame.from_dict(ground_truth, orient='index', columns=['Ground Truth'])
    results_df = pd.DataFrame.from_dict(results, orient='index', columns=['Detected'])
    comparison_df = pd.concat([ground_truth_df, results_df], axis=1)
    comparison_df['Absolute Error'] = abs(comparison_df['Ground Truth'] - comparison_df['Detected'])
    mae = comparison_df['Absolute Error'].mean()

    
    #print(comparison_df)
    print(mae)
