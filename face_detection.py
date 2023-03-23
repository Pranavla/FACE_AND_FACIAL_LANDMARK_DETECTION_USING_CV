#IMPORTING THE NECESSARY LIBRARIES
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt

#Importing and displaying the image

sample_img = cv2.imread('F:\Computer_vision\Open cv\Project\WhatsApp Image 2023-03-23 at 13.04.31.jpeg')
plt.figure(figsize = [15, 15])
plt.title("REAL IMAGE");plt.axis('off');plt.imshow(sample_img[:,:,::-1]);plt.show()

#Drawing Rectangle bounding box on face

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
face_detection_results = face_detection.process(sample_img[:,:,::-1])
    
img_copy = sample_img[:,:,::-1].copy()

if face_detection_results.detections:
    for face_no, face in enumerate(face_detection_results.detections):
        mp_drawing.draw_detection(image=img_copy, detection=face, keypoint_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2))
fig = plt.figure(figsize = [15, 15])
plt.title("FACE DETECTED IMAGE");plt.axis('off');plt.imshow(img_copy);plt.show()

#Drawing facial landmark on the face

mp_face_mesh = mp.solutions.face_mesh
face_mesh_images = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=2,min_detection_confidence=0.5)
mp_drawing_styles = mp.solutions.drawing_styles
face_mesh_results = face_mesh_images.process(sample_img[:,:,::-1])

img_copy = sample_img[:,:,::-1].copy()

if face_mesh_results.multi_face_landmarks:
    for face_landmarks in face_mesh_results.multi_face_landmarks:
        mp_drawing.draw_landmarks(image=img_copy, landmark_list=face_landmarks,connections=mp_face_mesh.FACEMESH_TESSELATION,landmark_drawing_spec=None, connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())       
        mp_drawing.draw_landmarks(image=img_copy, landmark_list=face_landmarks,connections=mp_face_mesh.FACEMESH_CONTOURS,landmark_drawing_spec=None, connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
fig = plt.figure(figsize = [15, 15])
plt.title("IMAGE WITH FACE LANDMARKS");plt.axis('off');plt.imshow(img_copy);plt.show()           