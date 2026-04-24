import cv2

camera_source = '0'
if camera_source.isdigit():
    camera_source = int(camera_source)

print(f'Camera source type: {type(camera_source)}')
print(f'Camera source value: {camera_source}')

cap = cv2.VideoCapture(camera_source)
print(f'Camera opened: {cap.isOpened()}')

ret, frame = cap.read()
print(f'Frame captured: {ret}')

cap.release()
