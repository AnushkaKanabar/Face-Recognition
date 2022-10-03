import face_recognition
import argparse
import pickle
import cv2
import random

print("Loading Encodings.....")
data = pickle.loads(open("./Jurassic-Park-characters-Face-recognition-master/encodings.pickle", "rb").read())

#!curl -o lunch_scene.mp4 https://www.youtube.com/watch?v=0Nz8YrCC9X8

#cap = cv2.VideoCapture('./Jurassic-Park-characters-Face-recognition-master/lunch scene video.mp4')
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, image = cap.read()

    if not ret:
      break

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

    cv2.imshow("frames", image)
    if cv2.waitKey(1) & 0xff==ord('q'):
      break

cv2.destroyAllWindows()
cap.release()

#image = cv2.imread('./Jurassic-Park-characters-Face-recognition-master/examples/example_1.jpg')
#rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


  
cv2.destroyAllWindows()