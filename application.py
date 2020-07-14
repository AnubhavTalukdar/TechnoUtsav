# Importing "flask" class from "Flask" package
from flask import Flask, render_template as rt, url_for, request, Response, jsonify, send_from_directory, abort, send_file, redirect
from flask_cors import CORS
import os
# from flask_ngrok import run_with_ngrok
import numpy as np
import imutils
import time
from gevent.pywsgi import WSGIServer
import re


import dlib
import tensorflow as tf
from PIL import Image
import cv2
import io
import ffmpy
from absl import app, logging

from tools import label_map_util
from tools import visualization_utils as vis_util
from tools import visualization_utils_og as vis_util_og
from tools.centroidtracker import CentroidTracker
from tools.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS


application = app = Flask(__name__)
CORS(app)

# run_with_ngrok(app)
print("v10.25")


MEDIA_ADDR = os.path.join(os.getcwd(), "static", "assets", "media")
MEDIA_IP = os.path.join(MEDIA_ADDR, "input")
MEDIA_OP = os.path.join(MEDIA_ADDR, "output")
SAMPLE_MEDIA_ADDRESS = ""

# Path to frozen detection graph. This is the actual model that is used for the weapons detection.
PATH_TO_CKPT = "./graph/weapons_v2.pb"
PATH_TO_LABELS = "./static/assets/weapons_label_map.pbtxt"

NUM_CLASSES = 4+1
TEST_IMAGE_PATHS = []
input_type = ""
IMAGE_ALLOWED_EXTENSIONS = set(['jpg', 'jpeg'])
VIDEO_ALLOWED_EXTENSIONS = set(['mp4'])
i = 0
people_count = []
sample = False
incompat_file_err = ""



# load in weights and classes
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


# Creating "app" variable and setting this to an instance of class "Flask". Basically, we're initialising the Flask application
#  __name__  : special variable in Python that is just the name of the variable



# AUXILIARY FUNCTIONS

def allowed_file(filename, endpoint):
	if endpoint=="/image":
		return '.' in filename and filename.rsplit('.', 1)[1].lower() in IMAGE_ALLOWED_EXTENSIONS
	elif endpoint=="/video":
		return '.' in filename and filename.rsplit('.', 1)[1].lower() in VIDEO_ALLOWED_EXTENSIONS
	else:
		return False

def video_converter():
    ff = ffmpy.FFmpeg(
     inputs={os.path.join(MEDIA_OP, 'output_temp.mp4'): None},
     outputs={os.path.join(MEDIA_OP, 'output.mp4'): None})
    ff.run()
    os.remove(os.path.join(MEDIA_OP, 'output_temp.mp4'))

# FLASK FUNCTIONS

@app.after_request
def add_header(response):
	# response.cache_control.no_store = True
	response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
	response.headers['Pragma'] = 'no-cache'
	response.headers['Expires'] = '-1'
	return response

# GET REQUESTS

@app.route("/")
@app.route("/home")
def home_get_handler():
	global MEDIA_OP, sample, incompat_file_err
	sample = False
	MEDIA_OP = os.path.join(MEDIA_ADDR, "output")
	return rt("home.html", err=incompat_file_err)

@app.route('/features', methods= ['GET'])
def features_get_handler():
	global incompat_file_err
	incompat_file_err = ""
	return rt("features.html")

@app.route('/image', methods= ['GET'])
def image_get_handler():
	global incompat_file_err
	incompat_file_err = ""
	return redirect("/", code=302)

@app.route('/aboutus', methods= ['GET'])
def aboutus_get_handler():
	global incompat_file_err
	incompat_file_err = ""
	return rt("aboutus.html")

@app.route('/sample', methods= ['GET'])
def sample_get_handler():
	global incompat_file_err
	incompat_file_err = ""
	return redirect("/", code=302)

@app.route('/video', methods= ['GET'])
def video_get_handler():
	global incompat_file_err
	incompat_file_err = ""
	return redirect("/", code=302)

@app.route("/results")
def results():
	global incompat_file_err
	incompat_file_err = ""
	if len(os.listdir(MEDIA_OP))==0:
		print(MEDIA_OP)
		print(people_count)
		return redirect("/", code=302)
	elif os.listdir(MEDIA_OP)[0].endswith(".mp4"):
		input_type = "video"
	elif os.listdir(MEDIA_OP)[0].endswith(".jpg"):
		input_type = "image"
	else:
		return redirect(url_for('/'))

	print(people_count)

	if sample == False:
		return rt("results.html", input_type=input_type, i=len(os.listdir(MEDIA_OP)),
	counter = people_count, sample=sample, sample_media_address=SAMPLE_MEDIA_ADDRESS)
	else:
		return rt("results.html", input_type=input_type, i=int(len(os.listdir(MEDIA_OP))/2),
	counter = people_count, sample=sample, sample_media_address=SAMPLE_MEDIA_ADDRESS)



# POST REQUESTS

@app.route('/image', methods= ['POST'])
def image_route_handler():
    global incompat_file_err
    detection_mode1 = request.form.get('iradio1')
    detection_mode2 = request.form.get('iradio2')

	# Removing previous files
    for file in os.listdir(MEDIA_OP):
        os.remove(os.path.join(MEDIA_OP, file))
    for file in os.listdir(MEDIA_IP):
        os.remove(os.path.join(MEDIA_IP, file))

	#Code to accept multiple files and securely save them
	# Includes file extension check
    i = 0
    uploaded_images = request.files.getlist("images")

    for image in uploaded_images:
        if image and allowed_file(image.filename, request.url_rule.rule):
            i = i+1
            image.save(os.path.join(MEDIA_IP, "input{}.jpg".format(i)))


    if(i==0):
        incompat_file_err = "Upload .JPG or .JPEG images only"
        print(incompat_file_err)
        return redirect("/", code=302)


    TEST_IMAGE_PATHS = []

	# Code to store the imput images in the list TEST_IMAGE_PATHS

    for image in os.listdir(MEDIA_IP):
        if(image.endswith(".jpg") or image.endswith(".jpeg")):
            TEST_IMAGE_PATHS.append(os.path.join(MEDIA_IP, image))

    i = 0

    if detection_mode1 == "wd":
        def load_image_into_numpy_array(image):
            (im_width, im_height) = image.size
            return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

        file_object = []


        people_count.clear()

		# ## Loading label map
        PATH_TO_LABELS = "./static/assets/weapons_label_map.pbtxt"
        PATH_TO_CKPT = "./graph/weapons_v2.pb"

        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
        category_index = label_map_util.create_category_index(categories)

        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        with detection_graph.as_default():
            with tf.Session(graph=detection_graph) as sess:
                for image_path in TEST_IMAGE_PATHS:
                    image = Image.open(image_path)
                    # the array based representation of the image will be used later in order to prepare the
                    # result image with boxes and labels on it.
                    image_np = load_image_into_numpy_array(image)
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
                    # Visualization of the results of a detection.
                    print("Classes: ", classes)
                    print("Scores: ", scores)
                    vis_util_og.visualize_boxes_and_labels_on_image_array(image_np,np.squeeze(boxes), np.squeeze(classes).astype(np.int32), np.squeeze(scores), category_index,
                    use_normalized_coordinates=True, line_thickness=5)
                    print(image_path)
                    im = Image.fromarray(image_np)
                    i= re.findall(r'\d+',image_path)[-1]
                    print("i= {}".format(i))
                    im.save(os.path.join(MEDIA_OP, "processed{}.jpg".format(i)))


        try:
            return redirect("results", code=302)
        except FileNotFoundError:
            abort(404)

    elif detection_mode2 == "pc":
        PATH_TO_CKPT = "./graph/ssd.pb"

        PATH_TO_LABELS = "./static/assets/people_label_map.pbtxt"
        people_count.clear()


        num_classes = 90
        detection_graph = tf.Graph()

        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')


        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_classes, use_display_name=True)
        category_index = label_map_util.create_category_index(categories)

        is_color_recognition_enabled = 0

        with detection_graph.as_default():
            with tf.Session(graph=detection_graph) as sess:
                print("Checkpoint 3")
                for image_path in TEST_IMAGE_PATHS:
                    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

                    # Each box represents a part of the image where a particular object was detected.
                    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

                    # Each score represent how level of confidence for each of the objects.
                    # Score is shown on the result image, together with the class label.
                    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
                    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
                    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

                    input_frame = cv2.imread(image_path)

                    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                    image_np_expanded = np.expand_dims(input_frame, axis=0)

                    # Actual detection.
                    (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})

                    # insert information text to video frame
                    font = cv2.FONT_HERSHEY_SIMPLEX

                    # Visualization of the results of a detection.
                    counter, csv_line, counting_mode = vis_util.visualize_boxes_and_labels_on_single_image_array(1,input_frame,
                    																					1, is_color_recognition_enabled, np.squeeze(boxes), np.squeeze(classes).astype(np.int32), np.squeeze(scores), category_index, use_normalized_coordinates=True, line_thickness=8)

                    if(len(counting_mode) == 0):
                        cv2.putText(input_frame, "Person: 0", (10, 35), font,   0.8, (0,255,255),2,cv2.FONT_HERSHEY_SIMPLEX)
                    else:
                        cv2.putText(input_frame, counting_mode, (10, 35), font, 0.8, (0,255,255),2,cv2.FONT_HERSHEY_SIMPLEX)

                    i= re.findall(r'\d+',image_path)[-1]
                    print("i= {}".format(i))
                    cv2.imwrite(os.path.join(MEDIA_OP,"processed{}.jpg".format(i)), input_frame)

                    if counting_mode!="":
                        people_count.append(int(counting_mode.split(" ")[1]))
                    elif counting_mode=="":
                        people_count.append(int("0"))
        try:
            print("No. of people= {}".format(people_count))
            return redirect("results", code=302)
        except FileNotFoundError:
            abort(404)
    else:
    	return redirect("/", code=302)

# API that returns video with detections on it
@app.route('/video', methods= ['POST'])
def video_route_handler():
	global incompat_file_err
	detection_mode1 = request.form.get('vradio1')
	detection_mode2 = request.form.get('vradio2')

	FILE_OUTPUT = os.path.join(MEDIA_OP,'output_temp.mp4')
	for file in os.listdir(MEDIA_OP):
		os.remove(os.path.join(MEDIA_OP, file))
	for file in os.listdir(MEDIA_IP):
		os.remove(os.path.join(MEDIA_IP, file))


	image = request.files["videos"]
	image_name = image.filename


	if not allowed_file(image.filename, request.url_rule.rule):
		incompat_file_err = "Upload .MP4 videos only."
		print(incompat_file_err)
		return redirect("/", code=302)

	image.save(MEDIA_IP+"/input.mp4")

	if detection_mode2 == "pc":
		print("PC mode entered")

		confidence_ip = 0.4
		prototxt = "./graph/MobileNetSSD_deploy.prototxt"
		model = "./graph/MobileNetSSD_deploy.caffemodel"

		# initialize the list of class labels MobileNet SSD was trained to
		# detect
		CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
			"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
			"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
			"sofa", "train", "tvmonitor"]

		# load our serialized model from disk
		print("[INFO] loading model...")
		net = cv2.dnn.readNetFromCaffe(prototxt, model)


		print("[INFO] opening video file...")
		vs = cv2.VideoCapture(os.path.join(MEDIA_IP, "input.mp4"))


		# initialize the video writer (we'll instantiate later if need be)
		writer = None

		skip_frames = 30


		# initialize the frame dimensions (we'll set them as soon as we read
		# the first frame from the video)
		W = None
		H = None

		# instantiate our centroid tracker, then initialize a list to store
		# each of our dlib correlation trackers, followed by a dictionary to
		# map each unique object ID to a TrackableObject
		ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
		trackers = []
		trackableObjects = {}

		# initialize the total number of frames processed thus far, along
		# with the total number of objects that have moved either up or down
		totalFrames = 0
		totalDown = 0
		totalUp = 0


		# start the frames per second throughput estimator
		fps = FPS().start()

		# loop over frames from the video stream
		while True:
			# grab the next frame and handle if we are reading from either
			# VideoCapture or VideoStream
			frame = vs.read()
			frame= frame[1]


			# if we are viewing a video and we did not grab a frame then we
			# have reached the end of the video
			if frame is None:
				break

			# resize the frame to have a maximum width of 500 pixels (the
			# less data we have, the faster we can process it), then convert
			# the frame from BGR to RGB for dlib
			frame = imutils.resize(frame, width=500)
			rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


			# if the frame dimensions are empty, set them
			if W is None or H is None:
				(H, W) = frame.shape[:2]


			# W = int(vs.get(3))
			# H = int(vs.get(4))


			# if we are supposed to be writing a video to disk, initialize
			# the writer
			if writer is None:
				input_fps = int(vs.get(cv2.CAP_PROP_FPS))
				fourcc = cv2.VideoWriter_fourcc(*"mp4v")
				writer = cv2.VideoWriter(FILE_OUTPUT, fourcc, input_fps, (W, H), True)

			# initialize the current status along with our list of bounding
			# box rectangles returned by either (1) our object detector or
			# (2) the correlation trackers
			status = "Waiting"
			rects = []

			# check to see if we should run a more computationally expensive
			# object detection method to aid our tracker
			if totalFrames % skip_frames == 0:
				# set the status and initialize our new set of object trackers
				status = "Detecting"
				trackers = []

				# convert the frame to a blob and pass the blob through the
				# network and obtain the detections
				blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
				net.setInput(blob)
				detections = net.forward()

				# loop over the detections
				for i in np.arange(0, detections.shape[2]):
					# extract the confidence (i.e., probability) associated
					# with the prediction
					confidence = detections[0, 0, i, 2]

					# filter out weak detections by requiring a minimum
					# confidence
					if confidence > confidence_ip:
						# extract the index of the class label from the
						# detections list
						idx = int(detections[0, 0, i, 1])

						# if the class label is not a person, ignore it
						if CLASSES[idx] != "person":
							continue

						# compute the (x, y)-coordinates of the bounding box
						# for the object
						box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
						(startX, startY, endX, endY) = box.astype("int")

						# construct a dlib rectangle object from the bounding
						# box coordinates and then start the dlib correlation
						# tracker
						tracker = dlib.correlation_tracker()
						rect = dlib.rectangle(startX, startY, endX, endY)
						tracker.start_track(rgb, rect)

						# add the tracker to our list of trackers so we can
						# utilize it during skip frames
						trackers.append(tracker)

			# otherwise, we should utilize our object *trackers* rather than
			# object *detectors* to obtain a higher frame processing throughput
			else:
				# loop over the trackers
				for tracker in trackers:
					# set the status of our system to be 'tracking' rather
					# than 'waiting' or 'detecting'
					status = "Tracking"

					# update the tracker and grab the updated position
					tracker.update(rgb)
					pos = tracker.get_position()

					# unpack the position object
					startX = int(pos.left())
					startY = int(pos.top())
					endX = int(pos.right())
					endY = int(pos.bottom())

					# add the bounding box coordinates to the rectangles list
					rects.append((startX, startY, endX, endY))

			# draw a horizontal line in the center of the frame -- once an
			# object crosses this line we will determine whether they were
			# moving 'up' or 'down'
			cv2.line(frame, (0, H // 2), (W, H // 2), (0, 255, 255), 2)

			# use the centroid tracker to associate the (1) old object
			# centroids with (2) the newly computed object centroids
			objects = ct.update(rects)

			# loop over the tracked objects
			for (objectID, centroid) in objects.items():
				# check to see if a trackable object exists for the current
				# object ID
				to = trackableObjects.get(objectID, None)

				# if there is no existing trackable object, create one
				if to is None:
					to = TrackableObject(objectID, centroid)

				# otherwise, there is a trackable object so we can utilize it
				# to determine direction
				else:
					# the difference between the y-coordinate of the *current*
					# centroid and the mean of *previous* centroids will tell
					# us in which direction the object is moving (negative for
					# 'up' and positive for 'down')
					y = [c[1] for c in to.centroids]
					direction = centroid[1] - np.mean(y)
					to.centroids.append(centroid)

					# check to see if the object has been counted or not
					if not to.counted:
						# if the direction is negative (indicating the object
						# is moving up) AND the centroid is above the center
						# line, count the object
						if direction < 0 and centroid[1] < H // 2:
							totalUp += 1
							to.counted = True

						# if the direction is positive (indicating the object
						# is moving down) AND the centroid is below the
						# center line, count the object
						elif direction > 0 and centroid[1] > H // 2:
							totalDown += 1
							to.counted = True

				# store the trackable object in our dictionary
				trackableObjects[objectID] = to

				# draw both the ID of the object and the centroid of the
				# object on the output frame
				text = "ID {}".format(objectID)
				cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
				cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

			# construct a tuple of information we will be displaying on the
			# frame
			info = [
				("Up", totalUp),
				("Down", totalDown),
				("Status", status),
			]

			# loop over the info tuples and draw them on our frame
			for (i, (k, v)) in enumerate(info):
				text = "{}: {}".format(k, v)
				cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
					cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

			# check to see if we should write the frame to disk
			if writer is not None:
				writer.write(frame)

			# show the output frame
		# 	cv2.imshow("Frame", frame)
		# 	key = cv2.waitKey(1) & 0xFF

		# 	# if the `q` key was pressed, break from the loop
		# 	if key == ord("q"):
		# 		break

			# increment the total number of frames processed thus far and
			# then update the FPS counter
			totalFrames += 1
			fps.update()

		# stop the timer and display FPS information
		fps.stop()
		print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
		print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

		# check to see if we need to release the video writer pointer
		if writer is not None:
			writer.release()



		# otherwise, release the video file pointer
		else:
			vs.release()

		# close any open windows
		cv2.destroyAllWindows()
		try:
			video_converter()
			return redirect("results", code=302)
		except FileNotFoundError:
			abort(404)

	elif detection_mode1 == "wd":
		print("Weapons mode entered")
		def load_image_into_numpy_array(image):
			(im_width, im_height) = image.size
			return np.array(image.getdata()).reshape(
			  (im_height, im_width, 3)).astype(np.uint8)
		# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
		# PATH_TO_TEST_IMAGES_DIR = "/gdrive/My Drive/object_detection/OIDv4_ToolKit/OID/video"
		# i = 0

		# Playing video from file
		cap = cv2.VideoCapture(os.path.join(MEDIA_IP, "input.mp4"))

		frame_width = int(cap.get(3))
		frame_height = int(cap.get(4))
		input_fps = int(cap.get(cv2.CAP_PROP_FPS))
		# input_fps = 30
		# codec = 0X00000021
		# Use above code for Linux
		codec = cv2.VideoWriter_fourcc(*'mp4v')

		# Define the codec and create VideoWriter object.The output is stored in 'output.avi' file.
		out = cv2.VideoWriter(FILE_OUTPUT, codec , input_fps, (frame_width, frame_height))


		NUM_CLASSES = 4+1


		# Load a Tensorflow model into memory.
		detection_graph = tf.Graph()
		with detection_graph.as_default():
			od_graph_def = tf.GraphDef()
			with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
				serialized_graph = fid.read()
				od_graph_def.ParseFromString(serialized_graph)
				tf.import_graph_def(od_graph_def, name='')

		# Loading label map
		label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
		categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
		category_index = label_map_util.create_category_index(categories)

		with detection_graph.as_default():
			with tf.Session(graph=detection_graph) as sess:
				# Definite input and output Tensors for detection_graph
				image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

				# Each box represents a part of the image where a particular object was detected.
				detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

				# Each score represent how level of confidence for each of the objects.
				detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
				detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
				num_detections = detection_graph.get_tensor_by_name('num_detections:0')

				while(cap.isOpened()):
					ret, frame = cap.read()

					if not ret:
						break;

					image_np_expanded = np.expand_dims(frame, axis=0)

					(boxes, scores, classes, num) = sess.run(
						[detection_boxes, detection_scores, detection_classes, num_detections],
						feed_dict={image_tensor: image_np_expanded})

					# Visualization of the results of a detection.
					vis_util_og.visualize_boxes_and_labels_on_image_array(
						frame,
						np.squeeze(boxes),
						np.squeeze(classes).astype(np.int32),
						np.squeeze(scores),
						category_index,
						use_normalized_coordinates=True,
						line_thickness=4   )

					if ret == True:
						# Saves for video
						out.write(frame)
					if cv2.waitKey(1) == ord('q'):
						break


			# When everything done, release the video capture and video write objects
			print("Exiting successfully")
			cap.release()
			out.release()

			try:
				video_converter()
				return redirect("results", code=302)
			except FileNotFoundError:
				abort(404)
	else:
		return redirect("/", code=302)



@app.route("/sample", methods= ['POST'])
def sample():
    global MEDIA_OP, sample, SAMPLE_MEDIA_ADDRESS
    sample_criteria = request.form.get('sample_criteria')
    MEDIA_OP = os.path.join(os.getcwd(), "static", "assets")
    sample = True

    people_count.clear()

    if(sample_criteria=="image_wd"):
        SAMPLE_MEDIA_ADDRESS = os.path.join("test_images", "wd")

    elif(sample_criteria=="video_wd"):
        SAMPLE_MEDIA_ADDRESS = os.path.join("test_videos", "wd")

    elif(sample_criteria=="image_pc"):
        people_count.append(2)
        people_count.append(2)
        people_count.append(2)
        people_count.append(1)
        people_count.append(4)
        people_count.append(1)
        SAMPLE_MEDIA_ADDRESS = os.path.join("test_images", "pc")

    elif(sample_criteria=="video_pc"):
        SAMPLE_MEDIA_ADDRESS =  os.path.join("test_videos", "pc")

    else:
        return redirect("/", code=302)


    MEDIA_OP = os.path.join(MEDIA_OP, SAMPLE_MEDIA_ADDRESS)
    return redirect("/results", code=302)






# if __name__ == '__main__':
# 	from gevent.pywsgi import WSGIServer
# 	app.debug = True
# 	http_server = WSGIServer(('', 5000), app)
# 	http_server.serve_forever()


# if __name__ == '__main__':
#     app.run()

# if __name__ == '__main__':
#     app.run(host = '0.0.0.0', port=80)

# if __name__ == '__main__':
#     app.run(debug=True, host = '0.0.0.0', port=5000)

# FOR FINAL DEPLOYMENT
if __name__ == "__main__":
    app.run(host='0.0.0.0')
