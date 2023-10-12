import cv2

# Input video file path
input_video_path = "./test_videos/test_1.mp4"

# Output video file path
output_video_path = "./test_videos/output.mp4"

# Open the input video
video_capture = cv2.VideoCapture(input_video_path)

# Get video properties
frame_width = int(video_capture.get(3))
frame_height = int(video_capture.get(4))
fps = int(video_capture.get(5))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

frame_count = 0
frame_skip = fps // 4  # Capture 4 frames per second

while True:
    ret, frame = video_capture.read()
    if not ret:
        break
    
    # Capture and write frames
    if frame_count % frame_skip == 0:
        output_video.write(frame)
    
    frame_count += 1

    # Break if all frames have been processed
    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

# Release the video objects
video_capture.release()
output_video.release()

# Close all OpenCV windows
cv2.destroyAllWindows()

