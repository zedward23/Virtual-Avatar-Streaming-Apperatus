import cv2
import dlib
import math

def vec2DDist(p1, p2):
    return math.sqrt((float(p1[0])-float(p2[0])) ** 2. + (float(p1[1]-p2[1])) ** 2.)

def EAR(points):
    p1 = points[0]
    p2 = points[1]
    p3 = points[2]
    p4 = points[3]
    p5 = points[4]
    p6 = points[5]

    return (vec2DDist(p2, p6) + vec2DDist(p3, p5)) / 2. * vec2DDist(p1, p4)

def main():
    blink_thresh = 284

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    hog_face_detector = dlib.get_frontal_face_detector()

    dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    points = [(0,0)]*6
    while True:
        _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        l_eye_min = [999999.,999999.]
        l_eye_max = [0,0]
        r_eye_min = [999999.,999999.]
        r_eye_max = [0,0]


        faces = hog_face_detector(gray)
        for face in faces:
            l_eye_min = [999999.,999999.]
            l_eye_max = [0,0]
            r_eye_min = [999999.,999999.]
            r_eye_max = [0,0]

            face_landmarks = dlib_facelandmark(gray, face)

            for n in range(36, 42):
                x = face_landmarks.part(n).x

                if (x < l_eye_min[0]):
                    l_eye_min[0] = x

                if (x > l_eye_max[0]):
                    l_eye_max[0] = x

                y = face_landmarks.part(n).y

                if (y < l_eye_min[1]):
                    l_eye_min[1] = y

                if (y > l_eye_max[1]):
                    l_eye_max[1] = y

                points[n-36] = (x,y)
                cv2.circle(frame, (x, y), 1, (0, 255, 255), 1)

            ear1 = EAR(points)

            #for n in range(43, 48):
            #    x = face_landmarks.part(n).x
            #    
            #    if (x < r_eye_min[0]):
            #        r_eye_min[0] = x
    #
            #    if (x > r_eye_max[0]):
            #        r_eye_max[0] = x
            #    
            #    y = face_landmarks.part(n).y
    #
            #    if (y < r_eye_min[1]):
            #        r_eye_min[1] = y
    #
            #    if (y > r_eye_max[1]):
            #        r_eye_max[1] = y
            #    
            #    points[n-43][0] = x
            #    points[n-43][1] = y
            #    cv2.circle(frame, (x, y), 1, (0, 255, 255), 1)
    #
            #ear2 = EAR(points)
        l_eye = frame[l_eye_min[1]-20:l_eye_max[1]+20, l_eye_min[0]-20:l_eye_max[0]+20]
        l_eye = cv2.resize(l_eye, (200,200))

        #r_eye = frame[r_eye_min[1]-20:r_eye_max[1]+20, r_eye_min[0]-20:r_eye_max[0]+20]
        #r_eye = cv2.resize(r_eye, (200,200))
        #print(ear1)
        if (ear1 < blink_thresh):
            print("left blink")
        print("")
        #if (ear2 > blink_thresh):
        #    print("right blink")

        cv2.imshow("LeftEye", l_eye)
        #cv2.imshow("RightEye", r_eye)


        key = cv2.waitKey(1)
        if key == 27:
            break
    cap.release()
    cv2.destroyAllWindows()