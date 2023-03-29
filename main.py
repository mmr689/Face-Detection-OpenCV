import os
import cv2


def main(img_path, model):
    architecture, weights = model
    # ------- LOAD THE MODEL -------
    net = cv2.dnn.readNetFromCaffe(architecture, weights)

    # ------- READ THE IMAGE AND PREPROCESSING -------
    img = cv2.imread(img_path)
    h, w, _ = img.shape
    img_resized = cv2.resize(img, (300, 300))
    # Create a blob
    blob = cv2.dnn.blobFromImage(img_resized, 1.0, (300, 300), (104, 117, 123))
    
    # ------- DETECTIONS AND PREDICTIONS ----------
    net.setInput(blob)
    detections = net.forward()
    # Work with all object detections
    for detection in detections[0][0]:
        # Only show detection with 90% confidence is a face
        if detection[2] > 0.9:
            box = detection[3:7] * [w, h, w, h]
            x_start, y_start, x_end, y_end = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            cv2.rectangle(img, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
            cv2.putText(img, "Conf: {:.2f}".format(detection[2] * 100), (x_start, y_start - 5), 1, 1.2, (0, 255, 255), 2)

    # ------- SHOW RESULTS -------
    cv2.imshow("Image", img)

    # ------- IF SOME KEY PRESSED CLOSE IMAGE -------
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":

    # ------- WORK PATHS -------
    current_dir = os.getcwd()
    # DNN
    architecture = os.path.join(current_dir, 'myFiles/models', 'deploy.prototxt')
    weights = os.path.join(current_dir, 'myFiles', 'models', 'res10_300x300_ssd_iter_140000.caffemodel')
    # Images
    img_list = ['all.png', 'all2.png', 'dwight.png']
    path_in_list = [os.path.join(current_dir, 'myFiles', 'imgs', file_name) for file_name in img_list]

    # -------  MAIN -------
    for path in path_in_list:
        main(img_path=path, model=(architecture, weights))

    print(' *** FIN PROGRAMA ***\n')