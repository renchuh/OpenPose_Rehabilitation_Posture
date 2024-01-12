import os
import sys
import argparse
import cv2
import time
import os
import sys
from playsound import playsound

sys.path.append(os.path.dirname(__file__) + "/../")

import imageio

from util.config import load_config
from nnet import predict
from util import visualize
from dataset.pose_dataset import data_to_input
import time
from  PIL import ImageFont,Image,ImageDraw
import numpy as np
import math
from threading import Thread
def playmuic1(name):
    playsound(name)

def playmuic2(name):
    playsound(name)

def playmuic3(name):
    playsound(name)

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

currentDT = time.localtime()
start_datetime = time.strftime("-%m-%d-%H-%M-%S", currentDT)
font = ImageFont.truetype(font='models/simhei.ttf', size=25)
result_map ={"ts":False,"zd":False,"qd":False}

def compute_angle(cord):
    lef1, lef2, lef3, right3, right2, right1 = cord# 计算抬起角度
    #left_distance = ((lef2[0] - lef3[0]) ** 2 + (lef2[1] - lef3[1]) ** 2) ** 0.5
    left_widht = lef3[0] - lef2[0]
    left_height = math.fabs(lef3[1] - lef2[1])
    left_angle = 0
    #print("angle:", left_widht,lef3[1] - lef2[1])
    if left_height!= 0 and left_widht !=0:
        if (lef2[1] - lef3[1]) < 0:
            left_angle = 90 + np.arctan(left_height/left_widht) * 180 / math.pi
        else:
            left_angle = np.arctan(left_widht/left_height) * 180 / math.pi

    #right_distance = ((right2[0] - right3[0]) ** 2 + (right2[1] - right3[1]) ** 2) ** 0.5
    right_widht = right2[0] - right3[0]
    right_height = math.fabs(right3[1] - right1[1])
    right_angle = 0
    if right_height != 0 and right_widht != 0:
        if (right2[1] - right3[1]) < 0:
            right_angle = 90 + np.arctan(right_height/right_widht) * 180 / math.pi
        else:
            right_angle = np.arctan(right_widht/right_height) * 180 / math.pi

    return left_angle,right_angle

def show_tip(res,text):
    res = Image.fromarray(res)
    draw = ImageDraw.Draw(res)
    label = text.encode('utf-8')
    draw.text(np.array([50, 50]), str(label, 'UTF-8'), fill=(255, 0, 0), font=font)
    del draw
    return res

def pose_confirm(cord,res):
    left_angle, right_angle = compute_angle(cord)
    if left_angle == 0 or right_angle ==0 :
        return res
    #print("angle:",left_angle)#,right_angle)
    if left_angle< 30 and right_angle <30 :
        if result_map["qd"]:# playsound.py 31行 修改 command = ' '.join(command).encode("gbk"),最后正价：
            # leep(float(durationInMS) / 1000.0)
            # winCommand('close',alias)
            p1 = Thread(target=playmuic1, args=('qd.mp3',))
            p1.start()
            print("輕度患者") # 語音
        elif result_map["zd"]:
            p2 = Thread(target=playmuic2, args=('zd.mp3',))
            p2.start()
            print( "重度患者")# 語音
        else:
            pass
            # p = Thread(target=playmuic3, args=('ts.mp3',))
            # p.start()
        result_map["ts"] = False
        result_map["zd"] = False
        result_map["qd"] = False
        return show_tip(res,"請抬手臂")
    elif (left_angle <120 and left_angle >80) or (right_angle <120 and right_angle >80):
        result_map["zd"] = True
    elif left_angle >120  or right_angle > 120:
        result_map["qd"] = True

    if result_map["qd"] :
        return show_tip(res, "輕度患者")
    elif result_map["zd"] :
        return show_tip(res, "重度患者")

    return res

if __name__ == '__main__':
    cfg = load_config("demo/pose_cfg.yaml")

    # Load and setup CNN part detector
    sess, inputs, outputs = predict.setup_pose_prediction(cfg)
    tic = time.time()

    # Video input
    video = "video.mp4"
    video_path = 'videos/'
    video_file = video_path + video

    # Output location
    output_path = 'videos/outputs/'
    output_format = '.mp4'
    video_output = output_path + video + str(start_datetime) + output_format


    # Video reader
    cam = cv2.VideoCapture(0)
    input_fps = cam.get(cv2.CAP_PROP_FPS)
    ret_val, orig_image = cam.read()
    video_length = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
    ending_frame = video_length
    frame_rate_ratio = 1
    process_speed = 1
    # Video writer
    output_fps = input_fps / frame_rate_ratio
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_output, fourcc, output_fps,
                          (orig_image.shape[1], orig_image.shape[0]))

    scale_search = [1, .5, 1.5, 2]  # [.5, 1, 1.5, 2]
    scale_search = scale_search[0:process_speed]


    i = 0  # default is 0
    while(cam.isOpened()) and ret_val is True :#and i < ending_frame:
        if i % frame_rate_ratio == 0:
            input_image = cv2.cvtColor(orig_image, cv2.COLOR_RGB2BGR)
            if input_image is None:
                out.release()
                cam.release()
                break
            tic = time.time()
            image_batch = data_to_input(input_image)

            # Compute prediction with the CNN
            outputs_np = sess.run(outputs, feed_dict={inputs: image_batch})
            scmap, locref, _ = predict.extract_cnn_output(outputs_np, cfg)

            # Extract maximum scoring location from the heatmap, assume 1 person
            pose = predict.argmax_pose_predict(scmap, locref, cfg.stride)
            toc = time.time()
            #print('processing time is %.5f' % (toc - tic))
            # Visualise
            res,cord = visualize.visualize_joints(input_image, pose)
            res = pose_confirm(cord,res)
            #visualize.show_heatmaps(cfg, input_image, scmap, pose)
        ret_val, orig_image = cam.read()
        i += 1
        res = cv2.cvtColor(np.asarray(res), cv2.COLOR_BGR2RGB)
        out.write(res)
        cv2.imshow('output', res)
        if cv2.waitKey(1) == ord('q'):
            out.release()
            cam.release()
            break
    out.release()
    cam.release()
    visualize.waitforbuttonpress()