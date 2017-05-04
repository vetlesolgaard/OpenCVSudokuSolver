
import cv2
import numpy as np
from copy import copy, deepcopy


class SudokuSolver:
    def __init__(self):
        print('init')
        self.img_list = []
        self.video_capture()


    def video_capture(self):
        cap = cv2.VideoCapture(0)
        while(True):
            # Capture frame-by-frame
            ret, orig_img = cap.read()

            ''' --- Image transform --- '''
            #gray_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
            board_img = self.find_sudoku_board(deepcopy(orig_img))

            ''' --- Show --- '''
            self.img_list.append(orig_img)
            self.img_list.append(board_img)
            self.display_images()
            self.img_list = [] # Need to clear image_list before next run

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.quit_program(cap)

    def find_sudoku_board(self, orig_img):
        canny_img = self.canny_edge_detector(orig_img)
        contour = self.find_contours(canny_img)
        contour_img, box_points = self.draw_contours(deepcopy(orig_img), contour)
        self.img_list.append(contour_img)
        cropped = self.crop_image(deepcopy(orig_img), box_points)
        return cropped


    def find_contours(self, img):
        mode = cv2.RETR_TREE
        method = cv2.CHAIN_APPROX_SIMPLE
        img = deepcopy(img)
        return_img, contours, hierarchy = cv2.findContours(img, mode, method)
        '''
        Right now it finds the largest area, needs to find the
        square that resembles a sudokuboard. Feature points maybe?
        '''
        best_contour = 0
        area = 0
        for cont in contours:
            area_sample = cv2.contourArea(cont)
            if area < area_sample:
                area = area_sample
                best_contour = cont
        return best_contour


    def draw_contours(self, orig_img, contour):
        #contour_img = cv2.drawContours(orig_img, contour, -1, (0,255,0), 3)
        #perimeter = cv2.arcLength(contour, True)
        #epsilon = 0.1*cv2.arcLength(contour, True)
        #approx = cv2.approxPolyDP(contour, epsilon, True)
        
        x,y,w,h = cv2.boundingRect(contour)
        contour_img = cv2.rectangle(orig_img, (x,y),(x+w,y+h),(0,255,0),2)
        box_points = np.array([[x, x+w], [y, y+h]])

        #box_points = np.int0(box_points)
        # contour_img = cv2.drawContours(orig_img,[box_points], 0, (0,255,0), 3)

        return contour_img, box_points


    def find_corner_features(self, gray_img):
        print('find_corner_features()')
        #sift = cv2.xfeatures2d.SIFT_create()
        #kp = sift.detect(gray_img, None)
        #sift_img = cv2.drawKeypoints(gray_img, kp)
        #c_features_img = cv2.drawKeypoints(gray_img, kp, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        return gray_img


    def canny_edge_detector(self, img):
        apertureSize = 3
        return cv2.Canny(img, 50, 150, apertureSize)


    def crop_image(self, orig_img, box_points):
        print(box_points)
        start_vertical = box_points[0][0]
        end_vertical = box_points[0][1]
        start_horizontal = box_points[1][0]
        end_horizontal = box_points[1][1]

        cropped = orig_img[start_horizontal:end_horizontal, start_vertical:end_vertical]

        return cropped


    def sobel(self, img):
        sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
        sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
        return np.sqrt(sobelx**2 + sobely**2).astype(np.uint8)


    def hough_lines(self, orig_img, edges):
        minLineLength = 0
        maxLineGap = 40
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength, maxLineGap)
        if(lines != None):
            for x in range(0, len(lines)):
                for x1,y1,x2,y2 in lines[x]:
                    cv2.line(orig_img, (x1,y1), (x2,y2), (0,255,0), 2)
        return orig_img


    def show_image(self, img):
        cv2.imshow('img', img)
        cv2.waitKey(0)


    def display_images(self):
        for idx, img in enumerate(self.img_list):
            cv2.imshow('img'+str(idx),img)


    def quit_program(self, cap):
        cap.release()
        cv2.destroyAllWindows()
        exit(0)


x = SudokuSolver()
