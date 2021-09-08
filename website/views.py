from flask import Blueprint, render_template, request, flash, make_response, Response
from werkzeug.utils import secure_filename
import numpy as np
import cv2
import base64
from face_analyse import face_analyse
from .models import User, Image
from . import db
import sqlalchemy
import os
import re
import time
import moviepy.editor as mp
from tqdm import tqdm
from datetime import datetime


views = Blueprint('views', __name__)

@views.route('/')
def home():
	return render_template("home.html")


"""
face-recognition
"""

def cv2_to_str(cv2_img):
	is_success, img_buf_arr = cv2.imencode(".jpg", cv2_img)
	if not is_success:
		return None

	byte_img = img_buf_arr.tobytes()
	byte_img = base64.b64encode(byte_img)
	byte_img = byte_img.decode("utf-8")
	return byte_img

@views.route("/face-recognition", methods=["GET", "POST"])
def face_recognition():
	if request.method == "POST":

		# request for file
		pic = request.files['pic']
		if not pic:
			flash("Image not valid.", category="error")
			return render_template("face_recognition.html", page=1)

		# check file type
		filename = secure_filename(pic.filename)
		mimetype = pic.mimetype
		if not filename or not mimetype or (mimetype != "image/jpeg" and mimetype != "image/png"):
			flash("Type image not valid.", category="error")
			return render_template("face_recognition.html", page=1)

		# str to cv2
		img = pic.read()
		img = np.fromstring(img, np.uint8)
		img = cv2.imdecode(img, 1)

		# resize
		scale_percent = 50
		width = int(img.shape[1] * scale_percent / 100)
		height = int(img.shape[0] * scale_percent / 100)
		dim = (width, height)
		img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

		# face recognition
		img, data, crop_imgs = face_analyse(img, "known", draw=True)

		# convert face_recognition ouput from cv2 to str
		img = cv2_to_str(img)
		if img is None:
			flash("Conversion failed.", category="error")
			return render_template("face_recognition.html", page=1)
		for i in range(len(crop_imgs)):
			crop_imgs[i] = cv2_to_str(crop_imgs[i])
			if crop_imgs[i] is None:
				flash("Conversion failed.", category="error")
				return render_template("face_recognition.html", page=1)

		
		return render_template("face_recognition.html", page=2, drawn_img=img, crop_img=crop_imgs)

	else:
		return render_template("face_recognition.html", page=1)

"""
upload
"""

@views.route("/upload", methods=["GET", "POST"])
def upload():
	if request.method == "POST":

		name = request.form.get("name")
		user_id = request.form.get("user_id")
		pic = request.files['pic']
		if not pic:
			flash("Image file not valid.", category="error")
			return render_template("upload.html", page=1)
		filename = secure_filename(pic.filename)
		mimetype = pic.mimetype
		if not filename or not mimetype or (mimetype != "image/jpeg" and mimetype != "image/png"):
			flash("File type not valid.", category="error")
			return render_template("upload.html", page=1)

		img = pic.read()

		#new_image = Image(user_id=1, data=img.tobytes())
		new_image = Image(user_id=1, data=img)
		db.session.add(new_image)
		db.session.commit()
		flash(filename + " added to db.", category="success")

		img = np.fromstring(img, np.uint8)
		img = cv2.imdecode(img, 1)
		img = cv2_to_str(img)

		return render_template("upload.html", page=2, img=img)
	else:
		return render_template("upload.html", page=1)

"""
video
"""

@views.after_request
def after_request(response):
	response.headers.add('Accept-Ranges', 'bytes')
	return response

def get_chunk(video_path, byte1=None, byte2=None):
	file_size = os.stat(video_path).st_size
	start = 0
	
	if byte1 < file_size:
		start = byte1
	if byte2:
		length = byte2 + 1 - byte1
	else:
		length = file_size - start

	with open(video_path, 'rb') as f:
		f.seek(start)
		chunk = f.read(length)
	return chunk, start, length, file_size

@views.route('/video-stream/<video_path>')
def video_stream(video_path):
	range_header = request.headers.get('Range', None)
	byte1, byte2 = 0, None
	if range_header:
		match = re.search(r'(\d+)-(\d*)', range_header)
		groups = match.groups()

		if groups[0]:
			byte1 = int(groups[0])
		if groups[1]:
			byte2 = int(groups[1])
	
	chunk, start, length, file_size = get_chunk(video_path, byte1=byte1, byte2=byte2)
	resp = Response(chunk, 206, mimetype='video/mp4',
					  content_type='video/mp4', direct_passthrough=True)
	resp.headers.add('Content-Range', 'bytes {0}-{1}/{2}'.format(start, start + length - 1, file_size))
	return resp

def write_video(filename, output_name, seconds_max=30, func=None):
	cap = cv2.VideoCapture(filename)
	if not cap.isOpened():
		print("File Cannot be Opened")
		return False
	fps = cap.get(cv2.CAP_PROP_FPS)
	frame_max = int(seconds_max * fps)
	size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
			int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
	length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	if seconds_max > 0 and length > frame_max:
		length = frame_max
	if length < 3:
		length = 3
	# fourcc = cv2.VideoWriter_fourcc(*'XVID')
	out = cv2.VideoWriter("tmp_video_no_sound.mp4", -1, fps, size)
	for i in tqdm(range(length)):
		if not cap.isOpened():
			print("cap not opened")
			break
		ret, frame = cap.read()
		if not ret:
			print("Can't receive frame (stream end?). Exiting ...")
			if i == 0:
				return False
			break
		if func is not None:
			frame = func(frame)
		out.write(frame)
	cap.release()
	out.release()
	cv2.destroyAllWindows()
	print("done")
	return True

def mp4_to_mp3(filename, output_name, seconds_max=30):
	audio = mp.VideoFileClip(filename).audio
	if seconds_max > 0 and audio.duration > seconds_max:
		audio = audio.set_duration(seconds_max)
	audio.write_audiofile(output_name)

def add_sound_on_mp4(video_name, audio_name, output_name):
	video = mp.VideoFileClip(video_name)
	video.write_videofile(output_name, audio=audio_name)

def cv2_edge(frame):
	frame = cv2.Canny(frame, 0, 255)
	return frame

def cv2_contours(frame):
	gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	retval, thresh = cv2.threshold(gray_img, 127, 255, 0)
	img_contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	cv2.drawContours(frame, img_contours, -1, (0, 255, 0), 3)
	return frame

def transparentOverlay(src, overlay, pos=(0, 0), scale=1):
	overlay = cv2.resize(overlay, (0, 0), fx=scale, fy=scale)
	h, w, _ = overlay.shape  # Size of foreground
	rows, cols, _ = src.shape  # Size of background Image
	y, x = pos[0], pos[1]  # Position of foreground/overlay image

	for i in range(h):
		for j in range(w):
			if x + i >= rows or y + j >= cols:
				continue
			alpha = float(overlay[i][j][3] / 255.0)  # read the alpha channel
			src[x + i][y + j] = alpha * overlay[i][j][:3] + (1 - alpha) * src[x + i][y + j]
	return src

def cv2_snap_filter(frame):
	face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
	specs_ori = cv2.imread('glass.png', -1)
	cigar_ori = cv2.imread('cigar.png', -1)
	mus_ori = cv2.imread('mustache.png', -1)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(frame, 1.2, 5, 0, (120, 120), (350, 350))
	for (x, y, w, h) in faces:
		if h > 0 and w > 0:
			glass_symin = int(y + 1.5 * h / 5)
			glass_symax = int(y + 2.5 * h / 5)
			sh_glass = glass_symax - glass_symin

			cigar_symin = int(y + 4 * h / 6)
			cigar_symax = int(y + 5.5 * h / 6)
			sh_cigar = cigar_symax - cigar_symin

			mus_symin = int(y + 3.5 * h / 6)
			mus_symax = int(y + 5 * h / 6)
			sh_mus = mus_symax - mus_symin

			face_glass_roi_color = frame[glass_symin:glass_symax, x:x + w]
			face_cigar_roi_color = frame[cigar_symin:cigar_symax, x:x + w]
			face_mus_roi_color = frame[mus_symin:mus_symax, x:x + w]

			specs = cv2.resize(specs_ori, (w, sh_glass), interpolation=cv2.INTER_CUBIC)
			cigar = cv2.resize(cigar_ori, (w, sh_cigar), interpolation=cv2.INTER_CUBIC)
			mustache = cv2.resize(mus_ori, (w, sh_mus), interpolation=cv2.INTER_CUBIC)

			transparentOverlay(face_glass_roi_color, specs)
			transparentOverlay(face_cigar_roi_color, cigar, (int(w / 2), int(sh_cigar / 2)))
			transparentOverlay(face_mus_roi_color, mustache)
	return frame

@views.route("/video", methods=["GET", "POST"])
def video():
	if request.method == "POST":
		file = request.files['file']
		seconds_max = request.form.get("seconds_max")
		preprocess = request.form.get("preprocess")
		try:
			seconds_max = int(seconds_max)
		except ValueError:
			seconds_max = -1
		filename = secure_filename(file.filename)
		# mimetype = file.mimetype
		# print(mimetype, filename)
		file.save(filename)

		cv2_funcs = {"edge": cv2_edge, "contours": cv2_contours, "snap_filter": cv2_snap_filter}
		if preprocess is not None:
			ret = write_video(filename, "tmp_video_no_sound.mp4", seconds_max=seconds_max, func=cv2_funcs[preprocess])
			print(preprocess)
		else:
			ret = write_video(filename, "tmp_video_no_sound.mp4", seconds_max=seconds_max)
		if not ret:
			flash("Video not valid.", category="error")
			return render_template("video.html", page=1)
		mp4_to_mp3(filename, "tmp_audio.mp3", seconds_max=seconds_max)
		output = "result" + datetime.now().strftime("%d-%m-%Y_%H:%M:%S").replace(":", "_") + ".mp4"
		add_sound_on_mp4("tmp_video_no_sound.mp4", "tmp_audio.mp3", output)
		os.system("rm -f tmp_video_no_sound.mp4 tmp_audio.mp3")

		return render_template('video.html', page=2, video_path=output)
	else:
		return render_template('video.html', page=1)

# def generate_frames(file_path):
# 	cap = cv2.VideoCapture(file_path)
# 	fps = cap.get(cv2.CAP_PROP_FPS)
# 	print("fps =", fps)

# 	while True:
# 		# get frame
# 		ret, frame = cap.read()
# 		if frame is None:
# 			break

# 		# apply transformations
# 		#gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
# 		frame = cv2.Canny(frame, 0, 255)
# 		# contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# 		# cv2.drawContours(edged, contours, -1, (0, 255, 0), 3)

# 		# return frame
# 		ret,buffer=cv2.imencode('.jpg',frame)
# 		frame=buffer.tobytes()
# 		yield(b'--frame\r\n'
# 				   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# @views.route('/video')
# def index():
# 	return render_template('video.html')

# @views.route('/video-stream')
# def video():
# 	return Response(generate_frames("src/video/touhou-bad-apple-amv.mp4"), mimetype='multipart/x-mixed-replace; boundary=frame')


# @auth.route('/sign-up', methods=["GET", "POST"])
# def sign_up():
# 	if request.method == "POST":
# 		email = request.form.get("email")
# 		pseudo = request.form.get("pseudo")
# 		password1 = request.form.get("password1")
# 		password2 = request.form.get("password2")

# 		if len(email) < 4:
# 			flash("Email must be greater than four characters", category="error")
# 		elif len(pseudo) < 2:
# 			flash("Pseudo must be greater than two characters", category="error")
# 		elif password1 != password2:
# 			flash("Passwords don't match", category="error")
# 		elif len(password1) < 7:
# 			flash("Firstname must be greater than seven characters", category="error")
# 		else:
# 			flash("Account created !", category="success")

# 	return render_template("sign_up.html")