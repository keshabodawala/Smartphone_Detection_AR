import os
import cv2
import time
import argparse
import multiprocessing
import numpy as np
import tensorflow as tf
from numba import jit,prange

from utils.app_utils import FPS, WebcamVideoStream
from multiprocessing import Queue, Pool
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
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

def arrange_points(points):
    #Arrange co-ordimates in clockwise order, starting from top-left co-ordinate 
    p = sorted(points,key=lambda x: x[0][0])
    sorted_points = p.copy()
    if p[0][0][1] > p[1][0][1]:
        sorted_points[0] = p[1]
        sorted_points[1] = p[0]
    if p[2][0][1] < p[3][0][1]:
        sorted_points[2] = p[3]
        sorted_points[3] = p[2]   
    return sorted_points


@jit(nopython=True, parallel = True)
def display(img, img_warp):
    #Replace green background with live feed
    h, w, c = img.shape
    for i in prange(0, h):
        for j in prange(0, w):
            if img_warp[i,j,0] == 0 and img_warp[i,j,1] == 0 and img_warp[i,j,2] == 0:
                continue
            else:
                if img_warp[i,j, 0] <= 100 and img_warp[i,j,1] >= 220 and img_warp[i,j,2] <= 100:
                    continue
                else:
                    img[i,j,:] = img_warp[i,j,:]
    return img


class AR:
    def __init__(self, video):
        self.video = video #Green Screen video used to overlay a 2D object
        self.cap = cv2.VideoCapture(video)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,320);
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,240);
        self.prev_points = []
        self.mean_points = []
        
        self.step = 0
        self.points = []
        self.img = []
        self.counter = 0
        self.marker_flag = False

    def set_mean(self, points):
        """
            Functionality : Calculate current co-ordinates of the marker based on the current and prevision positions of the marker
                            Mitigates jitter
        """
        if self.mean_points == []:
            self.mean_points = points
            return
    
        if self.step >= 100:
            self.mean_points = points
            self.step = 0
            return

        for i in range(4):
            self.mean_points[i] = np.array([[(5*self.mean_points[i][0,j] + 95*points[i][0,j])/100 for j in range(2)]])
        self.step += 1

       

    def get_mean(self):
        return self.mean_points
    
    def release(self):
        self.cap.release()
        
    def clear(self):
        self.points = []
        self.img = []
        self.marker_flag = False
        self.prev_points = self.points
    



    def detect_marker(self, box, img):
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
        thresh = cv2.Canny(img_gray, 50,120)
        img_output = img.copy()
        im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = []
        for contour in contours:
            #Approximate each contour with a rectangle. find a rectangle with the largest area
            epsilon = 0.08*cv2.arcLength(contour,True)
            result = cv2.approxPolyDP(contour,epsilon, True)
            area = cv2.contourArea(contour)
            
            if area >= 0.1*box_area:
                cnt = cnt + [contour]
                if len(result) == 4:
                    self.marker_flag = True
                    points = result
        
        if self.marker_flag is True:
            points = arrange_points(points)
                    
            for i in range(len(points)):
                points[i][0][0] += box_xmin
                points[i][0][1] += box_ymin

            self.set_mean(points)
            points = self.get_mean()
            self.points = points

        if self.points != []:
            img_output = self.superimpose(img)
            return img_output

        return img
        
    
    def superimpose(self, img):
        ret, frame = self.cap.read()
        
        if frame is None:
            self.cap = cv2.VideoCapture(self.video)
            ret, frame = self.cap.read()
        h, w, c = img.shape
        img_coord = np.float32([[0,0], [0, h-1], [w-1, h-1], [w-1, 0]])
        pts = np.float32(np.squeeze(self.points))

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(frame, (w,h))
        M = cv2.getPerspectiveTransform(img_coord, pts)
        img_warp = cv2.warpPerspective(img_resized, M, img.shape[1::-1])
        img_output = display(img, img_warp)
        
        return img_output
    
def detect_objects(image_np, sess, detection_graph, ar):
    """
        Functionality : Detect marker in a frame with the help of Tensorflow's object detection API 
    """
    
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

    img_output = image_np.copy()
    for i in range(sc.shape[0]):
        if sc[i] >= 0.65 and cl[i] == 1:
            box = tuple(bx[i].tolist())
            img_output = ar.detect_marker(box, img_contour)
    return image_output


def worker(input_q, output_q):
    # Load a (frozen) Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)
        
    
    ar = AR("greenscreen.mp4")
    fps = FPS().start()
    
    while True:
        fps.update()
        frame = input_q.get()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output_q.put(detect_objects(frame_rgb, sess, detection_graph, ar))
        ar.clear()
    ar.release()
    fps.stop()
    sess.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-src', '--source', dest='video_source', type=int,
                        default=0, help='Device index of the camera.')
    parser.add_argument('-wd', '--width', dest='width', type=int,
                        default=480, help='Width of the frames in the video stream.')
    parser.add_argument('-ht', '--height', dest='height', type=int,
                        default=360, help='Height of the frames in the video stream.')
    parser.add_argument('-num-w', '--num-workers', dest='num_workers', type=int,
                        default=2, help='Number of workers.')
    parser.add_argument('-q-size', '--queue-size', dest='queue_size', type=int,
                        default=5, help='Size of the queue.')
    args = parser.parse_args()

    logger = multiprocessing.log_to_stderr()
    logger.setLevel(multiprocessing.SUBDEBUG)

    input_q = Queue(maxsize=args.queue_size)
    output_q = Queue(maxsize=args.queue_size)
    pool = Pool(args.num_workers, worker, (input_q, output_q))

    video_capture = WebcamVideoStream(src=0,
                                      width=WIDTH,
                                      height=HEIGHT).start()
    fps = FPS().start()
    
    
    while True:  
        frame = video_capture.read()
        input_q.put(frame)
        
        t = time.time()        
        output_rgb = cv2.cvtColor(output_q.get(), cv2.COLOR_RGB2BGR)
        cv2.imshow('Video', output_rgb)
        fps.update()
        
        print('[INFO] elapsed time: {:.2f}'.format(time.time() - t))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    fps.stop()
    print('[INFO] elapsed time (total): {:.2f}'.format(fps.elapsed()))
    print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))

    pool.terminate()
    video_capture.stop()
    cv2.destroyAllWindows()
