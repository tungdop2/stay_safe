import os
import numpy as np
import cv2
import argparse

def load_top_view_config(file):
    with open(file, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        M = np.zeros((3, 3))
        for i in range(3):
            # print(lines[i].split(' '))
            M[i] = np.array(lines[i].split(' '))

        w_scale = float(lines[3])
        h_scale = float(lines[4])
    
    # print('M:', M)
    # print('w_scale:', w_scale)
    # print('h_scale:', h_scale)
    return M, w_scale, h_scale

def four_point_transform(image, pts):
    (tl, tr, br, bl) = pts[:4]

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([[0, 0],
                    [maxWidth - 1, 0],
                    [maxWidth - 1, maxHeight - 1],
                    [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(pts[:4], dst)
    # or use this function
    # M = cv2.findHomography(pts, dst)[0]
    
    w, h = pts[4:6]
    bl = cv2.perspectiveTransform(np.array([[bl]]), M)[0][0]
    h = cv2.perspectiveTransform(np.array([[h]]), M)[0][0]
    w = cv2.perspectiveTransform(np.array([[w]]), M)[0][0]
    h = np.sqrt(((h[0] - bl[0]) ** 2) + ((h[1] - bl[1]) ** 2))
    w = np.sqrt(((w[0] - bl[0]) ** 2) + ((w[1] - bl[1]) ** 2))

    return M, w, h

# define 4 points for ROI
def selectROI(event, x, y, flags, param):
    global imagetmp, roiPts, ct

    if event == cv2.EVENT_LBUTTONDOWN:
        roiPts[ct] = np.array([x, y], dtype="float32")
        ct += 1
        cv2.circle(imagetmp, (x, y), 2 if ct < 4 else 4, (0, 255, 0) if ct < 4 else (0, 0, 255), -1)
        # cv2.imshow("image", imagetmp)
        if ct == 4:
            cv2.line(imagetmp, (int(roiPts[0][0]), int(roiPts[0][1])), (int(roiPts[1][0]), int(roiPts[1][1])), (0, 255, 0), 2)
            cv2.line(imagetmp, (int(roiPts[1][0]), int(roiPts[1][1])), (int(roiPts[2][0]), int(roiPts[2][1])), (0, 255, 0), 2)
            cv2.line(imagetmp, (int(roiPts[2][0]), int(roiPts[2][1])), (int(roiPts[3][0]), int(roiPts[3][1])), (0, 255, 0), 2)
            cv2.line(imagetmp, (int(roiPts[3][0]), int(roiPts[3][1])), (int(roiPts[0][0]), int(roiPts[0][1])), (0, 255, 0), 2)

        cv2.imshow("image", imagetmp)

def printROI(M, w, h):
    with open('distance.txt', 'w') as f:
        for i in range(3):
            f.write('{} {} {}\n'.format(M[i][0], M[i][1], M[i][2]))
        f.write(str(w) + '\n')
        f.write(str(h) + '\n')

def selectROIfromvideo(video_path):
    global imagetmp, roiPts, ct
    roiPts = np.zeros((6, 2), dtype="float32")
    ct = 0

    vid = cv2.VideoCapture(video_path)
    image = vid.read()[1]
    # image = cv2.imread("1.jpg")
    cv2.imwrite("2.jpg", image)
    imagetmp = cv2.resize(image, (1600, 1600 * image.shape[0] // image.shape[1]))
    # imagetmp = image.copy()
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", selectROI)

    M = None
    while ct < 6:
        cv2.imshow("image", imagetmp)
        cv2.waitKey(500)
    cv2.imwrite("tmp.jpg", imagetmp)
    cv2.destroyAllWindows()

    M, w, h = four_point_transform(imagetmp, roiPts)
    printROI(M, w, h)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Select ROI from video')
    parser.add_argument('--video_path', type=str, default='../../videos/test.mp4', help='video path')
    args = parser.parse_args()
    print(args.video_path)
    selectROIfromvideo(args.video_path)
    
