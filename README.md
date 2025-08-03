# Classroom_AutoAttend_Systemt 
I choose to do this type of project to be able to dive in Computer Vision world and also for my class project.
In this project i used self collected data in the prefered environment and on the cam that used to detect and recognize too
I used CVzone and Ultralytics to help collecting data 
MTCNN and ArcFace from DeepFace for recognition
Ultralytics for antispoofing(YoLov8)
i collected about 12k for spoof and 12k for real for anti spoofing detection(including labels, so basically 6k each), Yolo basically predict which is which based on the pictrue texture and lightning, anti spoofing would work better if you have a special equipment to analyze face depth and such. I divide the trainning set into 70% train 20% val 10% test. Yolo wouldn't work well if two or many object close or overlap on each other.
In this project i encounnter a few problem which is real time system is kinda slow and laggy because i have to run it through YOLO first to be able to identify spoof/real so before send through the detect and recognize system, so it would be 2 time cropping face for a single frame since YoLo and Arcface has different input it would be better if you just use YOLO + ArcFace than YOLO+ MTCNN + ARCFACE but i added MTCNN in so it would be able to locate face feature with parameters to adjust base on Arcface template to embed vectors and by embedding it can be free from retrain.

