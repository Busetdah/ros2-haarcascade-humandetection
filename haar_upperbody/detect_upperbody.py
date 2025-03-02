import rclpy
from rclpy.node import Node
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import mediapipe as mp
import time
import numpy as np
from scipy.optimize import linear_sum_assignment
from ament_index_python.packages import get_package_share_directory
import os

class FusedDetectorMultiAngle(Node):
    def __init__(self):
        super().__init__('fused_detector_multi_angle')
        self.image_pub = self.create_publisher(Image, 'fused_detection_image', 10)
        self.bridge = CvBridge()
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.get_logger().error("Kamera tidak dapat dibuka!")
        
        # Cascade dari cv2.data.haarcascades untuk frontal face, upper body, dan full body
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.upperbody_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_upperbody.xml')
        self.fullbody_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
        if self.face_cascade.empty():
            self.get_logger().error("Gagal memuat Haar Cascade wajah dari cv2.data.haarcascades")
        if self.upperbody_cascade.empty():
            self.get_logger().error("Gagal memuat Haar Cascade upper body dari cv2.data.haarcascades")
        if self.fullbody_cascade.empty():
            self.get_logger().error("Gagal memuat Haar Cascade full body dari cv2.data.haarcascades")
        
        # Untuk cascade tambahan (back, front, side) diambil dari folder "cascades" di package
        package_share = get_package_share_directory('haar_upperbody')
        back_path = os.path.join(package_share, 'cascades', 'cascade_back.xml')
        front_path = os.path.join(package_share, 'cascades', 'cascade_front.xml')
        side_path = os.path.join(package_share, 'cascades', 'cascade_side.xml')
        self.back_cascade = cv2.CascadeClassifier(back_path)
        self.front_cascade = cv2.CascadeClassifier(front_path)
        self.side_cascade = cv2.CascadeClassifier(side_path)
        if self.back_cascade.empty():
            self.get_logger().error("Gagal memuat Cascade Back dari " + back_path)
        if self.front_cascade.empty():
            self.get_logger().error("Gagal memuat Cascade Front dari " + front_path)
        if self.side_cascade.empty():
            self.get_logger().error("Gagal memuat Cascade Side dari " + side_path)
        
        # Inisialisasi MediaPipe Hands untuk deteksi gesture tangan
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        self.mp_draw = mp.solutions.drawing_utils

        # State machine: "standby" atau "follow"
        self.mode = "standby"
        self.open_hand_start_time = None
        self.locked_target_bbox = None  # Format: (x, y, w, h)

        # Tracker (KCF) dan Kalman Filter (KF) untuk mode follow
        self.tracker = None
        self.kalman = None

        self.timer = self.create_timer(0.033, self.timer_callback)

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("Frame tidak terbaca")
            return

        # Flip frame agar mirror
        frame = cv2.flip(frame, 1)
        display_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Deteksi menggunakan semua cascade
        detections = []  # List of tuples (x, y, w, h)

        # Frontal face (warna hijau)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0,255,0), 2)
            detections.append((x, y, w, h))
        
        # Upper body (warna cyan)
        upperbodies = self.upperbody_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        for (x, y, w, h) in upperbodies:
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0,200,200), 2)
            detections.append((x, y, w, h))
        
        # Full body (warna ungu)
        fullbodies = self.fullbody_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        for (x, y, w, h) in fullbodies:
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (200,0,200), 2)
            detections.append((x, y, w, h))
        
        # Cascade tambahan: Back (warna biru)
        backs = self.back_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        for (x, y, w, h) in backs:
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (255,0,0), 2)
            detections.append((x, y, w, h))
        
        # Front (warna kuning)
        fronts = self.front_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        for (x, y, w, h) in fronts:
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0,255,255), 2)
            detections.append((x, y, w, h))
        
        # Side (warna biru muda)
        sides = self.side_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        for (x, y, w, h) in sides:
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (255,255,0), 2)
            detections.append((x, y, w, h))
        
        human_detected = len(detections) > 0
        if not human_detected:
            self.open_hand_start_time = None

        # Deteksi gesture tangan dengan MediaPipe
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        open_hand_detected = False
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(display_frame, handLms, self.mp_hands.HAND_CONNECTIONS)
                count = self.count_fingers(handLms, frame.shape)
                if count == 5:
                    open_hand_detected = True
                    cv2.putText(display_frame, "Open Hand Detected", (10,80),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                    break

        current_time = time.time()
        if self.mode == "standby":
            cv2.putText(display_frame, "Mode: Standby", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            if human_detected and open_hand_detected:
                if self.open_hand_start_time is None:
                    self.open_hand_start_time = current_time
                    self.get_logger().info("Open hand mulai terdeteksi...")
                elif current_time - self.open_hand_start_time >= 5:
                    # Lock target: prioritaskan dengan urutan: face > upper body > full body > back > front > side
                    if len(faces) > 0:
                        self.locked_target_bbox = faces[0]
                    elif len(upperbodies) > 0:
                        self.locked_target_bbox = upperbodies[0]
                    elif len(fullbodies) > 0:
                        self.locked_target_bbox = fullbodies[0]
                    elif len(backs) > 0:
                        self.locked_target_bbox = backs[0]
                    elif len(fronts) > 0:
                        self.locked_target_bbox = fronts[0]
                    elif len(sides) > 0:
                        self.locked_target_bbox = sides[0]
                    else:
                        self.locked_target_bbox = detections[0]
                    self.get_logger().info("Open hand 5 detik. Mode FOLLOW aktif.")
                    self.mode = "follow"
                    self.initialize_tracker(frame, self.locked_target_bbox)
                    self.initialize_kalman(self.locked_target_bbox)
            else:
                self.open_hand_start_time = None

        elif self.mode == "follow":
            cv2.putText(display_frame, "Mode: Follow", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            measurement_available = False

            # Gunakan gabungan deteksi Haar (dari semua cascade) sebagai measurement jika tersedia
            if human_detected:
                best_det = None
                best_iou = 0
                for det in detections:
                    iou_val = self.iou(self.locked_target_bbox, det)
                    if iou_val > best_iou:
                        best_iou = iou_val
                        best_det = det
                if best_det is not None and best_iou > 0.3:
                    self.locked_target_bbox = best_det
                    cx = best_det[0] + best_det[2] / 2.0
                    cy = best_det[1] + best_det[3] / 2.0
                    measurement = np.array([[np.float32(cx)],
                                            [np.float32(cy)]])
                    self.kalman.correct(measurement)
                    measurement_available = True

            # Update tracker (KCF)
            if self.tracker is not None:
                success, tracked_bbox = self.tracker.update(frame)
                if success:
                    x, y, w, h = [int(v) for v in tracked_bbox]
                    if w > 0 and h > 0:
                        measurement = np.array([[np.float32(x + w/2)],
                                                 [np.float32(y + h/2)]])
                        measurement_available = True
                        self.get_logger().info(f"Tracker measurement: x={x}, y={y}, w={w}, h={h}")
                        self.kalman.predict()
                        corrected = self.kalman.correct(measurement)
                        center_x, center_y = corrected[0,0], corrected[1,0]
                    else:
                        self.get_logger().warn("Tracker bounding box tidak valid (w atau h = 0)")
                        measurement_available = False
                else:
                    self.get_logger().warn("Tracker update gagal, gunakan prediksi KF")
                    self.kalman.predict()
                    corrected = self.kalman.statePost
                    center_x, center_y = corrected[0,0], corrected[1,0]
                    measurement_available = False
            else:
                self.kalman.predict()
                corrected = self.kalman.statePost
                center_x, center_y = corrected[0,0], corrected[1,0]
                measurement_available = False

            # Jika tidak ada measurement, gunakan prediksi KF saja dan cek apakah target sudah keluar frame
            if not measurement_available:
                frame_h, frame_w = frame.shape[:2]
                if center_x < 0 or center_y < 0 or center_x > frame_w or center_y > frame_h:
                    self.get_logger().info("Predicted target keluar frame. Reset ke mode STANDBY.")
                    self.mode = "standby"
                    self.locked_target_bbox = None
                    self.tracker = None
                    self.kalman = None
                else:
                    _, _, orig_w, orig_h = self.locked_target_bbox
                    fused_bbox = (int(center_x - orig_w/2), int(center_y - orig_h/2), orig_w, orig_h)
                    cv2.rectangle(display_frame, (fused_bbox[0], fused_bbox[1]),
                                  (fused_bbox[0]+orig_w, fused_bbox[1]+orig_h), (255,0,0), 3)
            else:
                _, _, orig_w, orig_h = self.locked_target_bbox
                fused_bbox = (int(center_x - orig_w/2), int(center_y - orig_h/2), orig_w, orig_h)
                cv2.rectangle(display_frame, (fused_bbox[0], fused_bbox[1]),
                              (fused_bbox[0]+orig_w, fused_bbox[1]+orig_h), (255,0,0), 3)

        image_msg = self.bridge.cv2_to_imgmsg(display_frame, encoding="bgr8")
        self.image_pub.publish(image_msg)
        cv2.imshow("Fused Detector with KF MultiAngle", display_frame)
        cv2.waitKey(1)

    def initialize_tracker(self, frame, bbox):
        self.get_logger().info(f"Initialize tracker dengan bbox: {bbox}")
        self.tracker = cv2.TrackerKCF_create()
        self.tracker.init(frame, tuple(bbox))

    def initialize_kalman(self, bbox):
        x, y, w, h = bbox
        center_x = x + w/2.0
        center_y = y + h/2.0
        self.get_logger().info(f"Initialize Kalman Filter dengan center: ({center_x}, {center_y})")
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                  [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                                 [0, 1, 0, 1],
                                                 [0, 0, 1, 0],
                                                 [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kalman.statePre = np.array([[np.float32(center_x)],
                                         [np.float32(center_y)],
                                         [0],
                                         [0]], np.float32)

    def count_fingers(self, handLms, image_shape):
        landmarks = handLms.landmark
        h, w, _ = image_shape
        count = 0
        finger_tips = [self.mp_hands.HandLandmark.INDEX_FINGER_TIP,
                       self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                       self.mp_hands.HandLandmark.RING_FINGER_TIP,
                       self.mp_hands.HandLandmark.PINKY_TIP]
        finger_pip = [self.mp_hands.HandLandmark.INDEX_FINGER_PIP,
                      self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
                      self.mp_hands.HandLandmark.RING_FINGER_PIP,
                      self.mp_hands.HandLandmark.PINKY_PIP]
        for tip, pip in zip(finger_tips, finger_pip):
            if landmarks[tip].y < landmarks[pip].y:
                count += 1
        if landmarks[self.mp_hands.HandLandmark.THUMB_TIP].x < landmarks[self.mp_hands.HandLandmark.THUMB_IP].x:
            count += 1
        return count

    def iou(self, bbox1, bbox2):
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1+w1, x2+w2)
        yi2 = min(y1+h1, y2+h2)
        inter_width = max(0, xi2 - xi1)
        inter_height = max(0, yi2 - yi1)
        inter_area = inter_width * inter_height
        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - inter_area
        return inter_area / union_area if union_area > 0 else 0

    def destroy_node(self):
        self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = FusedDetectorMultiAngle()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
