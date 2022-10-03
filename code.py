import face_recognition
import argparse
import pickle
import cv2
import random
# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-e", "--encodings", required=True, help="path to serialized db of facial encodings")
# ap.add_argument("-i", "--image", required=True, help="path to input image")
# ap.add_argument("-d", "--detection-method", type=str, default="hog", help="face detection model to use: either `hog` or `cnn`")
# args = vars(ap.parse_args())


print("Loading Encodings.....")
data = pickle.loads(open("./Jurassic-Park-characters-Face-recognition-master/encodings.pickle", "rb").read())

image = cv2.imread('./Jurassic-Park-characters-Face-recognition-master/examples/example_1.jpg')
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

print("Recognising Faces.....")
boxes = face_recognition.face_locations(rgb, model="hog")
encodings = face_recognition.face_encodings(rgb, boxes)
# initialize the list of names for each face detected
names= []

# loop over the facial embeddings
for encoding in encodings:
  matches= face_recognition.compare_faces(data["encodings"], encoding)
  name="Unknown"

  # check to see if we have found a match
  if True in matches:
    matchedIdxs = [i for (i, b) in enumerate(matches) if b]
    counts = {}
    for i in matchedIdxs:
      name = data["names"][i]
      counts[name] = counts.get(name, 0) + 1
      name = max(counts, key=counts.get)
      # update the list of names
    names.append(name)
  

# loop over the recognized faces
for ((top, right, bottom, left), name) in zip(boxes, names):
  cv2.rectangle(image, (left, top), (right, bottom), (255, 255, 0), 2)
  y = (top - 15 if top - 15 > 15 else top + 15)
  cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,0.75, (255, 255, 0), 2)
# show the output image
cv2.imshow("frame",image)
k=cv2.waitKey(0)



#if k == ord('s'): # wait for 's' key to save and exit
a=random.randint(0,99)
cv2.imwrite(f"./Jurassic-Park-characters-Face-recognition-master/output.jpg", image)
cv2.destroyAllWindows()