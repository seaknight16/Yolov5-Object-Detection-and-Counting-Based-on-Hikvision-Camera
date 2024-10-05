# import the necessary packages
import sys
from yolov5 import Darknet
from camera import LoadStreams, LoadImages
from utils.general import non_max_suppression, scale_boxes
from flask import Response
from flask import Flask
from flask import render_template
import json
import cv2
import os
from pathlib import Path
from utils.general import increment_path
import numpy as np
import time
from ultralytics.utils.plotting import Annotator, colors

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT_ = ROOT
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
project = ROOT / "runs/detect"
name = "exp",  # save results to project/name

# initialize a flask object
app = Flask(__name__)

# initialize the video stream and allow the camera sensor to warmup
with open('yolov5_config.json', 'r', encoding='utf8') as fp:
    opt = json.load(fp)
    print('[INFO] YOLOv5 Config:', opt)
darknet = Darknet(opt)
if darknet.webcam:
    # cudnn.benchmark = True  # set True to speed up constant image size inference
    dataset = LoadStreams(darknet.source, img_size=opt["imgsz"], stride=darknet.stride)
else:
    dataset = LoadImages(darknet.source, img_size=opt["imgsz"], stride=darknet.stride)
time.sleep(2.0)

# define a dictionary contains the class numbers and their corresponding class names
dict={
    0:'person',
    1:'bicycle',
    2:'car',
    3:'motorcycle',
    4:'airplane',
    5:'bus',
    6:'train',
    7:'truck',
    8:'boat',
    9:'traffic light',
    10:'fire hydrant',
    11:'stop sign',
    12:'parking meter',
    13:'bench',
    14:'bird',
    15:'cat',
    16:'dog',
    17:'horse',
    18:'sheep',
    19:'cow',
    20:'elephant',
    21:'bear',
    22:'zebra',
    23:'giraffe',
    24:'backpack',
    25:'umbrella',
    26:'handbag',
    27:'tie',
    28:'suitcase',
    29:'frisbee',
    30:'skis',
    31:'snowboard',
    32:'sports ball',
    33:'kite',
    34:'baseball bat',
    35:'baseball glove',
    36:'skateboard',
    37:'surfboard',
    38:'tennis racket',
    39:'bottle',
    40:'wine glass',
    41:'cup',
    42:'fork',
    43:'knife',
    44:'spoon',
    45:'bowl',
    46:'banana',
    47:'apple',
    48:'sandwich',
    49:'orange',
    50:'broccoli',
    51:'carrot',
    52:'hot dog',
    53:'pizza',
    54:'donut',
    55:'cake',
    56:'chair',
    57:'couch',
    58:'potted plant',
    59:'bed',
    60:'dining table',
    61:'toilet',
    62:'tv',
    63:'laptop',
    64:'mouse',
    65:'remote',
    66:'keyboard',
    67:'cell phone',
    68:'microwave',
    69:'oven',
    70:'toaster',
    71:'sink',
    72:'refrigerator',
    73:'book',
    74:'clock',
    75:'vase',
    76:'scissors',
    77:'teddy bear',
    78:'hair drier',
    79:'toothbrush'
}

@app.route("/")
def index():
    # return the rendered template
    return render_template("default.html")
@app.route("/detection")
def result():
    # return the rendered template
    return render_template("detection.html")

def detect_gen(dataset):
    judge_arr = [True]*10
    judge_=True
    cnt=0
    num=0
    vid_writer = None
    name="exp"
    name2="video"
    names = darknet.names
    save_dir = increment_path(Path(project) / name, exist_ok=False)
    (save_dir / "labels").mkdir(parents=True, exist_ok=True)
    op = 1
    file_path = str(ROOT_ /save_dir / (name2 + str(0)))+'.mp4'
    for path, img, img0s, vid_cap, judge in dataset:
        if judge is False:
            judge_arr[cnt] = False
            cnt+=1
        else:
            judge_arr[cnt] = True
            cnt += 1
        if cnt==9:
            print(judge_arr)
            if np.sum(judge_arr)<=3:
                if judge_ is True:
                    op=1
                judge_=False
            else:
                if judge_ is False:
                    op=1
                judge_ = True
            judge_arr = [True] * 10
            cnt = 0
        # use "judge_" to determine whether there's a frame change or not
        if judge_ is True:
            img = darknet.preprocess(img)
            t1 = time.time()
            pred = darknet.model(img, augment=darknet.opt["augment"])[0]
            pred = pred.float()
            pred = non_max_suppression(pred, darknet.opt["conf_thres"], darknet.opt["iou_thres"])
            t2 = time.time()
            txt_path = str(save_dir / "labels" / "results")
            feed_type_curr, p, s, im0, frame = "Camera_%s" % str(0), path[0], '%g: ' % 0, img0s[0].copy(), dataset.count
            for i, det in enumerate(pred):
                arr=[0]*80
                fps, w, h = 8.8, im0.shape[1], im0.shape[0]
                p = Path(p)  # to Path
                save_path = str(save_dir / (name2+str(num)))  # path of saving video results
                save_path = str(Path(save_path).with_suffix(".mp4"))  # force *.mp4 suffix on results videos
                annotator = Annotator(im0, line_width=3, example=str(names))
                if op==1:
                    op=0
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter.fourcc(*"mp4v"), fps, (w, h))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], im0.shape).round()
                    for *xyxy, conf, cls in reversed(det):
                        # Write results
                        arr[int(cls)] += 1
                        line = dict[int(cls)]  # label format
                        with open(f"{txt_path}.txt","a") as f:  #write results to a text file
                            f.write(line + ' ' + time.strftime('%Y-%m-%d %X',time.localtime()) + "\n")
                        c = int(cls)  # integer class
                        label = f"{names[c]} {conf:.2f}"
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    with open(f"{txt_path}.txt", "a") as f:
                        for j in range(0,len(arr)):
                            if arr[j]!=0:
                                f.write("the number of "+dict[j]+" is: "+str(arr[j])+"\n")
                        f.write("\n")
                im0 = annotator.result()
                vid_writer.write(im0)
                if num!=0:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                s += '%gx%g ' % img.shape[2:]  # print string
                if det is not None and len(det):
                    det[:, :4] = scale_boxes(
                        img.shape[2:], det[:, :4], im0.shape).round()
                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {darknet.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                print(f'{s}Done. ({t2 - t1:.3f}s)')
                frame = cv2.imencode('.jpg', im0)[1].tobytes()
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            if vid_writer is not None:
                if op==1:
                    op=0
                    vid_writer.release()
                    num+=1
            img = darknet.preprocess(img)
            t1 = time.time()
            pred = darknet.model(img, augment=darknet.opt["augment"])[0]
            pred = pred.float()
            pred = non_max_suppression(pred, darknet.opt["conf_thres"], darknet.opt["iou_thres"])
            t2 = time.time()
            feed_type_curr, p, s, im0, frame = "Camera_%s" % str(0), path[0], '%g: ' % 0, img0s[0].copy(), dataset.count
            for i, det in enumerate(pred):
                fps, w, h = 8.8, im0.shape[1], im0.shape[0]
                p = Path(p)  # to Path
                save_path = str(save_dir / (name2 + str(num)))  # path of saving video results
                save_path = str(Path(save_path).with_suffix(".mp4"))  # force *.mp4 suffix on results videos
                if op==1:
                    op=0
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter.fourcc(*"mp4v"), fps, (w, h))
                s += '%gx%g ' % img.shape[2:]  # print string
                if det is not None and len(det):
                    det[:, :4] = scale_boxes(
                        img.shape[2:], det[:, :4], im0.shape).round()
                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {darknet.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                print(f'{s}Done. ({t2 - t1:.3f}s)')
                frame = cv2.imencode('.jpg', im0)[1].tobytes()
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/hikvision_stream')
def hikvision_stream():
    return Response(detect_gen(dataset=dataset),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8081, threaded=True)

