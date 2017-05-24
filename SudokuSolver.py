
import cv2
import numpy as np
from copy import copy, deepcopy
import scipy.misc as scipy
from NumberClassification import *
from Board import *

class SudokuSolver:
    def __init__(self):
        print('init')
        self.nc = NumberClassification()
        self.img_list = []

        self.video_capture()

    def video_capture(self):
        cap = cv2.VideoCapture(0)
        while(True):
            a = raw_input('.')

            # Capture frame-by-frame
            ret, orig_img = cap.read()
            self.img_list.append(orig_img)

            ''' Finding board, return contour and coordinates to box'''
            box_points, contour_img, processed_img = self.find_sudoku_board(deepcopy(orig_img))
            board_img = self.crop_image(orig_img, box_points)
            gray_board_img = cv2.cvtColor(board_img, cv2.COLOR_RGB2GRAY)
            board_processed_img = self.crop_image(processed_img, box_points)
            self.img_list.append(board_processed_img)

            ''' We have a board_img '''
            if len(board_img > 0):

                ''' Computing hough lines in board '''
                # Find HoughLines, merges and return #
                merged_lines = self.hough_lines(deepcopy(board_img), board_processed_img)
                self.img_list.append(board_img)
                if len(merged_lines) > 0:
                    print(merged_lines)
                    print('lines ->', len(merged_lines))
                self.visualize_grid_lines(board_img, merged_lines)
                if len(merged_lines) == 20:
                    print('Correct grid detected!')
                    #self.visualize_grid(board_img, merged_lines)
                    # Extract grid coordinates #
                    grid_points = self.extract_grid(board_img, merged_lines)
                    # # Maps the grid points to cells #
                    mapped_grid = self.map_grid(board_img, grid_points)
                    # ''' We have a confirmed grid '''
                    print(mapped_grid)
                    if mapped_grid is not None:
                        print('map_grid ->', len(mapped_grid))
                        prefilled = self.classify_cells(gray_board_img, mapped_grid)
                        sudoku_to_solve = self.create_array_with_prefilled(prefilled)
                        print(sudoku_to_solve)
                        self.solve_sudoku_board(sudoku_to_solve)



            ''' --- Show --- '''
            self.display_images()
            self.img_list = [] # Need to clear image_list before next run
            if cv2.waitKey(1) & 0xFF == ord('q') or a=='q':
                self.quit_program(cap)


    def solve_sudoku_board(self, board_to_solve):
        b = Board(array, 9,3,3)
        b.createBoard()
        solved_board = b.solveBoard()
        print(solved_board)


    def create_array_with_prefilled(self, prefilled):
        sudoku_to_solve = np.zeros(81)
        for i in range(0, len(prefilled[0])):
            sudoku_to_solve[prefilled[1][i]] = prefilled[0][i]
        return np.reshape(sudoku_to_solve, (9,9))

    def classify_cells(self, board_img, mapped_grid):
        cells = np.asarray(self.crop_grid(board_img, mapped_grid))
        cells = cv2.bitwise_not(cells) / 255.0
        cl_cells = []
        idx_list = []
        print('Success')
        for idx, c in enumerate(cells):
            self.img_list.append(c)
            c[c<0.5] = 0.0
            c[:5,:] = 0.0
            c[:,:5] = 0.0
            c[25:,:] = 0.0
            c[:,25:] = 0.0
            c = c*1.4
            print(np.sum(c[5:20,10:18]))
            if np.sum(c[5:20,10:18]) > 10.0:
                cl_cells.append(c)
                idx_list.append(idx)
        pred = np.argmax(self.nc.classify_images(np.asarray(cl_cells)), axis=1)
        y_label = [2, 5, 8, 4, 3, 3, 6, 1, 9, 2, 7, 1]
        for i in range(0, len(pred)):
            print(y_label[i], pred[i])
        prefilled = []
        prefilled.append(pred)
        prefilled.append(idx_list)
        print('pred -> ', pred)
        print('idx list -> ', idx_list)
        return prefilled

    def find_sudoku_board(self, orig_img):
        processed_img = self.canny_edge_detector(orig_img)
        #gray_board_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
        #processed_img = self.preprocess_for_grid_detection(gray_board_img)
        contour = self.find_contours(processed_img)
        contour_img, box_points = self.draw_contours(orig_img, contour)
        #cropped = self.crop_image(deepcopy(orig_img), box_points)
        return box_points, contour_img, processed_img


    def crop_grid(self, img, mapped_grid):
        cells = []
        for i in range(0, len(mapped_grid)-1):
            for j in range(0, len(mapped_grid[i])-1):
                topl = mapped_grid[i][j]
                topr = mapped_grid[i][j+1]
                botl = mapped_grid[i+1][j]
                botr = mapped_grid[i+1][j+1]
                start_vert = int((topl[1]+topr[1])/2)
                end_vert = int((botl[1]+botr[1])/2)
                start_horiz = int((topl[0]+botl[0])/2)
                end_horiz = int((topr[0]+botr[0])/2)
                if start_vert>0 and end_vert>0 and start_horiz>0 and end_horiz>0:
                    cells.append(scipy.imresize(img[start_vert:end_vert, start_horiz:end_horiz], (28,28)))
        return cells


    def map_grid(self, img, grid_points):
        width = img.shape[0]
        height = img.shape[1]
        mapped_grid = None
        print('grid_points before -> ', len(grid_points))
        grid_points = self.cleanup_grid_points(grid_points)
        grid_points = sorted(grid_points,key=lambda x: (x[1],x[0]))
        print('grid_points -> ', len(grid_points))
        if len(grid_points) == 100: # Only 9x9 sudokuboard
            grid_points = np.asarray(grid_points).reshape((10,10,2))
            mapped_grid = np.zeros_like(grid_points)
            for i in range(0, len(grid_points)):
                mapped_grid[i] = sorted(grid_points[i],key=lambda x: (x[0],x[1]))
        return mapped_grid

    ''' Not needed '''
    def cleanup_grid_points(self, grid_points):
        clean_list = []
        def append_to_clean_list(grid_points):
            if len(clean_list) <=0:
                clean_list.append(grid_points)
            x1 = grid_points[0]
            y1 = grid_points[1]
            for i in range(0, len(clean_list)):
                x2, y2 = clean_list[i]
                if x1==x2 and y1==y2: return
                elif x1>x2-20 and x1<x2+20 and y1>y2-20 and y1<y2+20: return
            clean_list.append(grid_points)
        for i in range(0, len(grid_points)):
            append_to_clean_list(grid_points[i])
        return clean_list



    def intersection(self, l1p1, l1p2, l2p1, l2p2):
        da = l1p2-l1p1
        db = l2p2-l2p1
        dp = l1p1-l2p1
        def perp(a):
            b = np.empty_like(a)
            b[0] = -a[1]
            b[1] = a[0]
            return b
        dap = perp(da)
        denom = np.dot(dap, db)
        if denom==0:
            return None
        num = np.dot(dap, dp)
        return (num / denom.astype(float)*db + l2p1)


    def extract_grid(self, img, point_list):
        all_lines = []
        width = img.shape[0]
        height = img.shape[1]
        for i in range(0, len(point_list)):
            lines = []
            rho = point_list[i][0]
            theta = point_list[i][1]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            lines.append([x1, y1])
            lines.append([x2, y2])
            all_lines.append(lines)
        intersect_points = []
        for i in range(0, len(all_lines)):
            for j in range(0, len(all_lines)):
                if all_lines[i]==all_lines[j]: continue
                l1p1 = np.asarray(all_lines[i][0])
                l1p2 = np.asarray(all_lines[i][1])
                l2p1 = np.asarray(all_lines[j][0])
                l2p2 = np.asarray(all_lines[j][1])
                #cv2.line(img, (l1p1[0], l1p1[1]), (l1p2[0], l1p2[1]), (0,255,0), 1)
                nparr = self.intersection(l1p1, l1p2, l2p1, l2p2)
                temp_intersect = []
                if nparr is not None:
                    if np.abs(nparr[1]) > height or np.abs(nparr[0]) > width:
                        continue
                    x = int(nparr[0])
                    y = int(nparr[1])
                    temp_intersect.append(x)
                    temp_intersect.append(y)
                if len(temp_intersect) > 0:
                    intersect_points.append(temp_intersect)
        return intersect_points


    def find_contours(self, img):
        mode = cv2.RETR_TREE
        method = cv2.CHAIN_APPROX_SIMPLE
        img = deepcopy(img)
        return_img, contours, hierarchy = cv2.findContours(img, mode, method)
        '''
        Right now it finds the largest area, needs to find the
        square that resembles a sudokuboard. Feature points maybe?
        '''
        best_contour = None
        area = None
        for cont in contours:
            area_sample = cv2.contourArea(cont)
            if area < area_sample:
                area = area_sample
                best_contour = cont
        return best_contour


    def preprocess_for_grid_detection(self, orig_img):
        gaus_img = cv2.GaussianBlur(orig_img, (11,11), 0)
        thresh_img = cv2.adaptiveThreshold(gaus_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        return thresh_img


    def draw_contours(self, orig_img, contour):
        ''' Can be improved, still a little unstable '''
        #contour_img = cv2.drawContours(orig_img, contour, -1, (0,255,0), 3)
        #perimeter = cv2.arcLength(contour, True)
        #epsilon = 0.1*cv2.arcLength(contour, True)
        #approx = cv2.approxPolyDP(contour, epsilon, True)
        x,y,w,h = cv2.boundingRect(contour)
        contour_img = cv2.rectangle(orig_img, (x,y),(x+w,y+h),(0,255,0),2)
        box_points = np.array([[y, y+h], [x, x+w]])
        return contour_img, box_points
        #box_points = np.int0(box_points)
        # contour_img = cv2.drawContours(orig_img,[box_points], 0, (0,255,0), 3)


    def gaussian_blur(self, orig_img):
        gaus_img = cv2.GaussianBlur(orig_img, (11,11), 0)
        return gaus_img

    def crop_image(self, orig_img, box_points):
        start_vertical = box_points[0][0]
        end_vertical = box_points[0][1]
        start_horizontal = box_points[1][0]
        end_horizontal = box_points[1][1]
        # print(start_vertical)
        # print(end_vertical)
        # print(start_horizontal)
        # print(end_horizontal)
        cropped = orig_img[start_vertical:end_vertical, start_horizontal:end_horizontal]
        return cropped


    def hough_lines(self, orig_img, edges):
        point_list = []
        def add_to_list(rho, theta):
            temp_list = []
            temp_list.append(rho)
            temp_list.append(theta)
            if len(point_list) <= 0:
                point_list.append(temp_list)
            else:
                for x in range(0, len(point_list)):
                    l_rho = point_list[x][0]
                    l_theta = point_list[x][1]
                    if l_rho == 0 or l_theta == -100: continue #Impossible values
                    elif rho == l_rho and theta == l_theta: continue
                    elif l_rho > rho-20.0 and l_rho < rho+20.0 and l_theta > theta-(np.pi/4) and l_theta < theta+(np.pi/4):
                        point_list[x][0] = (rho+l_rho)/2
                        point_list[x][1] = (theta+l_theta)/2
                        return
                point_list.append(temp_list)

        lines = cv2.HoughLines(image=edges, rho=1, theta=1*np.pi/180, threshold=150)
        if lines is not None:
            lines = abs(lines)
            for x in range(0, len(lines)):
                f_rho = lines[x][0][0]
                f_theta = lines[x][0][1]
                a = np.cos(f_theta)
                b = np.sin(f_theta)
                x0 = a*f_rho
                y0 = b*f_rho
                x1 = int(x0 + 1000*(-b))
                y1 = int(y0 + 1000*(a))
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*(a))
                if x < 3 or x > 18:
                    cv2.line(orig_img, (x1, y1), (x2, y2), (0,255,0), 1)
                if f_theta > 3.0 or f_rho == 0.0: continue
                add_to_list(f_rho, f_theta)
        self.img_list.append(orig_img)
        return sorted(point_list,key=lambda x: x[0])


    def visualize_grid_lines(self, img, point_list):
        img = deepcopy(img)
        for i in range(0, len(point_list)):
            rho = point_list[i][0]
            theta = point_list[i][1]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            cv2.line(img, (x1, y1), (x2, y2), (0,255,0), 1)
            # if i > 18:
            #     print('rho -> ', rho)
            #    print('theta -> ', theta)
        self.img_list.append(img)
        return

    def visualize_grid_points(self, img, point_list):
        img = deepcopy(img)
        for i in range(0, len(point_list)):
            x = grid_points[i][0]
            y = grid_points[i][1]
            cv2.circle(img, (x,y), 5, (0,255,0), 1)
        self.img_list.append(img)
        return

    def find_corner_features(self, gray_img):
        corners = cv2.goodFeaturesToTrack(gray_img,100,0.01,30)
        corners = np.int0(corners)
        for i in corners:
            x,y = i.ravel()
            cv2.circle(gray_img,(x,y),3,255,-1)
        return gray_img

    def harris_corner_detection(self, gray_img):
        gray_img = np.float32(gray_img)
        harris_dst = cv2.cornerHarris(gray_img,5,3,0.04)
        dst = cv2.dilate(harris_dst,None)
        return dst

    def canny_edge_detector(self, img):
        apertureSize = 3
        canny_img = cv2.Canny(img, 150, 250, apertureSize)
        canny_img = cv2.dilate(canny_img, (13,13))
        return canny_img

    def sobel(self, img):
        sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
        sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
        return np.sqrt(sobelx**2 + sobely**2).astype(np.uint8)

    def show_image(self, img):
        cv2.imshow('img', img)
        cv2.waitKey(0)

    def display_images(self):
        for idx, img in enumerate(self.img_list):
            if len(img) > 0:
                cv2.imshow('img'+str(idx),img)

    def quit_program(self, cap):
        cap.release()
        cv2.destroyAllWindows()
        exit(0)

if __name__ == "__main__":
    x = SudokuSolver()
