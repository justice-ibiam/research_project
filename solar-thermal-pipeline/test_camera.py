import subprocess
import numpy as np
import cv2
import sys
import os
import time

OUTPUT_DIR = "data/incoming"
AUTO_CAPTURE = True          # True = save continuously every X seconds, False = manually press 's'
CAPTURE_INTERVAL_SECS = 2.0  # Saves a frame every 2.0 seconds
SAVE_GRAYSCALE = True        # True for 1-channel grayscale; False for 3-channel BGR

# Create dataset path
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Perfected FFmpeg parameters
ffmpeg_command = [
    'ffmpeg',
    '-f', 'avfoundation',
    '-pix_fmt', 'uyvy422',
    '-framerate', '9',
    '-video_size', '160x120',
    '-i', '0',
    '-f', 'image2pipe',
    '-pix_fmt', 'bgr24',     
    '-vcodec', 'rawvideo', '-'
]

print(f" Thermal dataset pipeline initialized. Target: '{OUTPUT_DIR}'")
print(" PRESS [S] (while selecting the preview window) to force a frame save.")
print(" PRESS [Q] (while selecting the preview window) to cleanly exit.")

# Open pipe and make sure stderr is suppressed so it doesn't crowd out script prints
pipe = subprocess.Popen(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

last_saved_time = time.time()
frame_idx = 0

try:
    while True:
        # Read exact packet buffer (160 * 120 * 3 = 57,600 bytes)
        raw_bytes = pipe.stdout.read(57600)
        if not raw_bytes or len(raw_bytes) < 57600:
            continue
            
        # Reshape flat binary array back to 2D image matrix
        frame = np.frombuffer(raw_bytes, dtype='uint8').reshape((120, 160, 3))
        
        # Prepare live preview screen (Upscale to 640x480 for humans)
        preview = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_NEAREST)
        cv2.putText(preview, f"Logged Images: {frame_idx}", (15, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('Pure Thermal FLIR Pipeline', preview)

        # KEYBOARD LOGIC: Must call cv2.waitKey to let macOS draw the window frame
        key = cv2.waitKey(10) & 0xFF  # Wait 10ms for snappy processing
        
        current_time = time.time()
        trigger_save = False
        
        if AUTO_CAPTURE and (current_time - last_saved_time >= CAPTURE_INTERVAL_SECS):
            trigger_save = True
            last_saved_time = current_time
        elif key == ord('s'):
            trigger_save = True

        if trigger_save:
            processed_frame = frame.copy()
            if SAVE_GRAYSCALE:
                processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
                
            timestamp = int(time.time() * 1000)
            filename = os.path.join(OUTPUT_DIR, f"solar_frame_{timestamp}_{frame_idx:05d}.png")
            cv2.imwrite(filename, processed_frame)
            print(f" Saved image: {filename} | Layout: {processed_frame.shape}")
            frame_idx += 1

        if key == ord('q'):
            break

except KeyboardInterrupt:
    print("\n Interrupted via terminal command.")

finally:
    # Safely close everything down so the camera device isn't locked up next run
    pipe.terminate()
    pipe.wait()
    cv2.destroyAllWindows()
    # Explicitly flush window events for macOS
    for _ in range(5):
        cv2.waitKey(1)
    print(f"\n Pipeline Closed. Total files saved to '{OUTPUT_DIR}': {frame_idx}")
