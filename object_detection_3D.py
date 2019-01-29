"""
    Functionality : Detect smartphone from the video feed and augment animation

    Main Functions:
    detect_objects : Localize smartphone in a frame with the help of trained Tensorflow graph
    detect_marker_orientation : Fine tune position of detected smartphone
    render_object : Calculate transformation paramteres to correctlty orient 3D model
"""


import cv2
import time
import argparse
import numpy as np
import tensorflow as tf
import os, glob
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import OpenGL.GLUT.freeglut
from PIL import Image
import numpy as np
from utils.objloader import *
import math

from utils.app_utils import FPS, WebcamVideoStream
from object_detection.utils import label_map_util

CWD_PATH = os.getcwd()

# Path to frozen detection graph. This is the actual model that is used for the object detection.
MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
PATH_TO_CKPT = os.path.join(CWD_PATH, 'object_detection', MODEL_NAME, 'frozen_inference_graph.pb')

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(CWD_PATH, 'object_detection', 'data', 'object-detection.pbtxt')

NUM_CLASSES = 1
WIDTH = 480
HEIGHT = 360

# Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)


class Char:
    """
        Functionality : Load and store animation
    """
    def __init__(self):
        self.frames = []
        self.frames_length = 0
        self.frame_index = 0
                  
    # load frames from directory
    def load(self, directory):
        os.chdir(directory)
        for file in glob.glob("*.OBJ"):
            self.frames.append(OBJ(file))
        
        os.chdir('..')
        self.frames_length = len(self.frames)
 
    # get next frame
    def next_frame(self):
        self.frame_index += 1
 
        if self.frame_index >= self.frames_length:
            self.frame_index = 0
 
        return self.frames[self.frame_index].gl_list

class AR:
    """
        Functionality : Detect marker and its orientation
                        Overlay animation
    """
        
    def __init__(self):
        self.prev_points = []
        self.mean_points = []

        self.INVERSE_MATRIX = np.array([[ 1.0, 1.0, 1.0, 1.0],
                                   [-1.0,-1.0,-1.0,-1.0],
                                   [-1.0,-1.0,-1.0,-1.0],
                                   [ 1.0, 1.0, 1.0, 1.0]])
        

        self.points = []
        self.img = []
        
        self.marker_flag = False
        self.cup = Char()
        self.texture_background = None
        self.animation_path = args.animation_path

        glutInit()
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
        glutInitWindowSize(args.WIDTH, args.HEIGHT)
        glutInitWindowPosition(0, 0)
        self.window_id = glutCreateWindow("Smartphone Detection")
        
        glutDisplayFunc(detect_objects)
        
        glutIdleFunc(detect_objects)
        
        self._init_gl(args.WIDTH, args.HEIGHT)

        
    def set_mean(self, points):
        """
            Functionality : Calculate current co-ordinates of the marker based on the current and prevision positions of the marker
                            Mitigates jitter
        """
        if self.mean_points == []:
            self.mean_points = points
            return
    
        for i in range(4):
            self.mean_points[i] = np.array([(10*self.mean_points[i][j] + 90*points[i][j])/100 for j in range(2)])

       
    def get_mean(self):
        return self.mean_points
    
        
    def clear(self):
        self.points = []
        self.img = []
        self.marker_flag = False

    def _init_gl(self, width, height):
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClearDepth(1.0)
        glDepthFunc(GL_LESS)
        glEnable(GL_DEPTH_TEST)
        glShadeModel(GL_SMOOTH)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(33.7, 1.3, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)

        if self.animation_path == "animation_full":
            print("Loading animation.. This might take a few of minutes! You can change to static animation by passing argument : '-a-path = \"animaiton\"'")
        self.cup.load(self.animation_path)

        glEnable(GL_TEXTURE_2D)
        self.texture_background = glGenTextures(1)

        self.rvecs = []
        self.tvecs = []
        
    def _draw_background(self):
        # draw background
        glBegin(GL_QUADS)
        glTexCoord2f(0.0, 1.0); glVertex3f(-1.0, -1.0, 0.0)
        glTexCoord2f(1.0, 1.0); glVertex3f( 1.0, -1.0, 0.0)
        glTexCoord2f(1.0, 0.0); glVertex3f( 1.0,  1.0, 0.0)
        glTexCoord2f(0.0, 0.0); glVertex3f(-1.0,  1.0, 0.0)
        glEnd( )

    def _draw_scene(self,box, image):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
 
        # convert image to OpenGL texture format
        bg_image = cv2.flip(image, 0)
        bg_image = Image.fromarray(bg_image)     
        ix = bg_image.size[0]
        iy = bg_image.size[1]
        bg_image = bg_image.tobytes("raw", "BGRX", 0, -1)
  
        # create background texture
        glBindTexture(GL_TEXTURE_2D, self.texture_background)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexImage2D(GL_TEXTURE_2D, 0, 3, ix, iy, 0, GL_RGBA, GL_UNSIGNED_BYTE, bg_image)
         
        # draw background
        glBindTexture(GL_TEXTURE_2D, self.texture_background)
        glPushMatrix()
        glTranslatef(0.0,0.0,-10.0)
        self._draw_background()
        glPopMatrix()
        if box is not None:
            #Marker detected by Tensorflow model
            image = self.detect_marker_orientation(box, image)
        else:
            image = None
        glutSwapBuffers()
        
    def _draw_background(self):
        # draw background
        glBegin(GL_QUADS)
        glTexCoord2f(0.0, 1.0); glVertex3f(-4.0, -3.0, 0.0)
        glTexCoord2f(1.0, 1.0); glVertex3f( 4.0, -3.0, 0.0)
        glTexCoord2f(1.0, 0.0); glVertex3f( 4.0,  3.0, 0.0)
        glTexCoord2f(0.0, 0.0); glVertex3f(-4.0,  3.0, 0.0)
        glEnd( )
    
    def render_object(self,image, points):
    	# load calibration data
        with np.load('webcam_calibration_ouput1.npz') as X:
            mtx, dist, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]

	# set up criteria, image, points and axis
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
        imgp = np.array(points, dtype="float32")
 
        objp = np.array([[0.,0.,0.],[1.,0.,0.],
                         [1.,1.,0.],[0.,1.,0.]], dtype="float32")  
 
        axis = np.float32([[0,0,0], [0,1,0], [1,1,0], [1,0,0],
                           [0,0,-1],[0,1,-1],[1,1,-1],[1,0,-1] ])
 
        # project 3D points to image plane
        imgp = cv2.cornerSubPix(gray,imgp,(11,11),(-1,-1),criteria)
        _,rvecs, tvecs = cv2.solvePnP(objp, imgp, mtx, dist)
       
        return rvecs, tvecs


    def detect_marker_orientation(self, box, img):
        """
            Functionality : Detect marker's orientation:
            Hint: All mobile phones have a black screen. Detect bounding rectangle around this screen.
                  If no such rectangle can be found, then may be it is a False Positive
        """
        
        self.marker_flag = False 
        img_height, img_width, chanel = img.shape
        points = []
        ymin, xmin, ymax, xmax = box

        #retrive the box returned by Tensorflow model
        box_ymin = int((ymin-0.1)*img_height)
        box_ymax = int((ymax+0.1)*img_height)
        box_xmin = int((xmin-0.1)*img_width)
        box_xmax = int((xmax+0.1)*img_width)
        
        if box_xmin <0:
            box_xmin = 0
        if box_ymin <0:
            box_ymin = 0
        if box_xmax > img_width:
            box_xmax = img_width
        if box_ymax > img_height:
            box_ymax = img_height

        #Calculate height, width and area of the box
        box_height = box_ymax-box_ymin
        box_width = box_xmax-box_xmin
        box_img = img[box_ymin:box_ymax, box_xmin:box_xmax]
        box_area = (box_height)*(box_width)

        #Thresholding -> Canny Edge Detection -> Contouring
        img_gray = cv2.cvtColor(box_img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.GaussianBlur(img_gray,(3,3),0)
        ret3,thresh = cv2.threshold(thresh,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        thresh = cv2.Canny(thresh, 50,60)
                
        im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_area = 0
        
        for contour in contours:
            #Approximate each contour with a rectangle. find a rectangle with the largest area
            epsilon = 0.08*cv2.arcLength(contour,True)
            result = cv2.approxPolyDP(contour,epsilon, True)
            area = cv2.contourArea(contour)

            if len(result) == 4 and math.fabs(area)>=0.1*box_area and  math.fabs(area)> max_area:
                    self.marker_flag = True #Rectangle found
                    points = result.reshape(4, 2)
            
        if self.marker_flag is True:
            #Order points in clockwise direction starting from the top-left point
            ordered_points = self.order_points(points)

            for i in range(len(ordered_points)):
                ordered_points[i,0] += box_xmin
                ordered_points[i,1] += box_ymin
                    
            self.set_mean(ordered_points)
            ordered_points = self.get_mean()
            self.points = ordered_points
            
        else:
            return
        if self.points != []:
            #Render object
            img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            rvecs, tvecs = self.render_object(img, self.points)
            
            rmtx = cv2.Rodrigues(rvecs)[0]
     
            view_matrix = np.array([[rmtx[0][0],rmtx[0][1],rmtx[0][2],tvecs[0]],
                                    [rmtx[1][0],rmtx[1][1],rmtx[1][2],tvecs[1]],
                                    [rmtx[2][0],rmtx[2][1],rmtx[2][2],tvecs[2]],
                                    [0.0       ,0.0       ,0.0       ,1.0    ]])
     
            view_matrix = view_matrix * self.INVERSE_MATRIX
     
            view_matrix = np.transpose(view_matrix)
     
            # load view matrix and draw shape
            glPushMatrix()
            glLoadMatrixd(view_matrix)
     
            glCallList(self.cup.next_frame())     
            glPopMatrix()


    def order_points(self, points):
        s = points.sum(axis=1)
        diff = np.diff(points, axis=1)
                     
        ordered_points = np.zeros((4,2), dtype="float32")
                 
        ordered_points[0] = points[np.argmin(s)]
        ordered_points[2] = points[np.argmax(s)]
        ordered_points[1] = points[np.argmin(diff)]
        ordered_points[3] = points[np.argmax(diff)]
        return ordered_points
    
    
def detect_objects():
    """
        Functionality : Detect marker in a frame with the help of Tensorflow's object detection API 
    """
    frame = video_capture.read()
    image_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)      
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]

    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Each box represents a part of the image where a particular object was detected.
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Actual detection.
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})
    cl = np.squeeze(classes)
    sc = np.squeeze(scores)
    bx = np.squeeze(boxes)
    ind = sc.argmax()
    img_contour = image_np.copy()
    box = None
    img_output = image_np.copy()

    if sc[ind] >= 0.95 and cl[ind] == 1:
        box = tuple(bx[ind].tolist())
    ar._draw_scene(box, cv2.cvtColor(img_contour, cv2.COLOR_BGR2RGB))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-src', '--source', dest='video_source', type=int,
                        default=0, help='Device index of the camera.')
    parser.add_argument('-wd', '--width', dest='WIDTH', type=int,
                        default=WIDTH, help='Width of the frames in the video stream.')
    parser.add_argument('-ht', '--height', dest='HEIGHT', type=int,
                        default=HEIGHT, help='Height of the frames in the video stream.')
    parser.add_argument('-a-path', '--animation-path', dest='animation_path', type=str,
                        default="animation_full", help='Path of folder containing animation frames.')
    
    args = parser.parse_args()

    ar = AR()
    
    video_capture = WebcamVideoStream(src=args.video_source,
                                      width=WIDTH,
                                      height=HEIGHT).start()

    fps = FPS().start()
    

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)

    while True:  
        t = time.time()
        
        glutMainLoopEvent()
        detect_objects()
        fps.update()
        print('[INFO] elapsed time: {:.2f}'.format(time.time() - t))
        ar.clear()
    glutLeaveMainLoop();
    sess.close()
    fps.stop()
    video_capture.stop()
    cv2.destroyAllWindows()



