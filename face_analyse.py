from retinaface import RetinaFace
from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt
from IPython.display import display


# test
def face_analyse(img, db_path, draw=True):
	return img, {"blbl": 42}, [img]*10



# def face_analyse(img, db_path, draw=True):
# 	# find faces in img
# 	faces = RetinaFace.detect_faces(img)
# 	drawn_img = img.copy()

# 	# stock face informations
# 	data = {}
# 	i = 1
# 	for key in faces.keys():
# 		# for each face found in img
# 		identity = faces[key]

# 		facial_area = identity["facial_area"]
# 		right_eye = identity["landmarks"]["right_eye"]
# 		left_eye = identity["landmarks"]["left_eye"]
# 		nose = identity["landmarks"]["nose"]
# 		mouth_right = identity["landmarks"]["mouth_right"]
# 		mouth_left = identity["landmarks"]["mouth_left"]

# 		width = facial_area[3] - facial_area[1]
# 		height = facial_area[2] - facial_area[0]
# 		percent = 5
# 		percent_threshold = 100
# 		while percent <= percent_threshold:
# 			crop_img = img[facial_area[1]-int(width*percent/100):facial_area[3]+int(width*percent/100),
# 						   facial_area[0]-int(height*percent/100):facial_area[2]+int(height*percent/100)]
# 			try:
# 				demography = DeepFace.analyze(crop_img, detector_backend="retinaface")
# 				df = DeepFace.find(img_path=crop_img, db_path=db_path, model_name="Facenet", detector_backend="retinaface")
# 				break
# 			except:
# 				percent += 5
# 		tmp = {}
# 		if percent > percent_threshold:
# 			tmp["emotion"] = None
# 			tmp["age"] = None
# 			tmp["gender"] = None
# 			tmp["race"] = None
# 			tmp["recognition"] = None
# 		else:
# 			tmp["emotion"] = demography["dominant_emotion"]
# 			tmp["age"] = demography["age"]
# 			tmp["gender"] = demography["gender"]
# 			tmp["race"] = demography["dominant_race"]
# 			if df.shape[0] > 0:
# 				tmp["recognition"] = df["identity"][0].split(".")[0].split("/")[-1]
# 			else:
# 				tmp["recognition"] = None

# 		if draw:
# 			thickness = 2
# 			cv2.rectangle(drawn_img, (facial_area[2], facial_area[3]), (facial_area[0], facial_area[1]), (0, 255, 255), thickness)
# 			cv2.circle(drawn_img, (int(right_eye[0]), int(right_eye[1])), radius=thickness, color=(0, 0, 255), thickness=-1)
# 			cv2.circle(drawn_img, (int(left_eye[0]), int(left_eye[1])), radius=thickness, color=(0, 0, 255), thickness=-1)
# 			cv2.circle(drawn_img, (int(nose[0]), int(nose[1])), radius=thickness, color=(0, 0, 255), thickness=-1)
# 			cv2.line(drawn_img, (int(mouth_right[0]), int(mouth_right[1])), (int(mouth_left[0]), int(mouth_left[1])), (0, 0, 255), thickness=thickness)
# 			drawn_img = cv2.putText(drawn_img, str(i), (facial_area[0], facial_area[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness, cv2.LINE_AA, False)

# 		# save this face data in data dict
# 		data[i] = tmp
# 		i += 1

# 	# return drawn_img, data
# 	# test
# 	return drawn_img, data, crop_img