# Yolov5-Object-Detection-and-Counting-Based-on-Hikvision-Camera
## Usage
This project is based on a Hikvision Camera, make sure you have connected to the camera before running the codes.
1. Create a virtual environment according to "requiremnets"
2. Open the "default.html" file, change the "href" in line 23 and line 47 to the url of your index.html and Cameras.html, respectively
3. Alter the "source" in "yolov5_config.json" file into “rtsp://” + ”your hikvision user name” + ”:” + ”your hikvision password” + ”@” + ”your hikvision camera’s IP adress” + ”/Streaming/Channels/” + ”your hikvision camera’s main stream number”. For example: "rtsp://admin:123456@192.168.0.147/Streaming/Channels/1"
4. Run "web_main.py"
5. Click on "Cameras Management", then click on "view" in the new page
6. Make some movements before the camera. The results are saved in the "runs/detect/exp" files
