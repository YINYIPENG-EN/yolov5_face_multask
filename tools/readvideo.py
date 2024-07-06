import cv2

cap = cv2.VideoCapture('F:/yolov5_face/runs/demo.avi')
fps = 25
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('res.mp4', fourcc, fps, (width, height))
while True:
    _, frame = cap.read()
    out.write(frame)
    cv2.imshow("res", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
out.release()
cv2.destroyAllWindows()
