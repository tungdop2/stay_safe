import re
from tabnanny import check
from loguru import logger
import numpy as np

import cv2

import torch

from utils.visualize import plot_tracking
from tracker.byte_tracker import BYTETracker
from tracking_utils.timer import Timer
from utils.boxes import postprocess
import argparse
import os
import time
from models.common import DetectMultiBackend
from utils.augmentations import letterbox

from facemask.resnet import model
from distance.distance import load_top_view_config


IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]
# device = "cpu"

def make_parser():
    parser = argparse.ArgumentParser("3K Demo!")
    parser.add_argument(
        "demo", default="image", help="demo type, eg. video and webcam"
    )
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        "--path", default="./videos/test.mp4", help="path to images or video"
    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_name",
        type=str,
        default="test",
        help="save name for results txt/video",
    )

    parser.add_argument("-c", "--ckpt", default="weights/crowdhuman_yolov5s.pt", type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=0.1, type=float, help="test conf")
    parser.add_argument("--nms", default=0.45, type=float, help="test nms threshold")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument("--track_thresh", type=float, default=0.6, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument('--min-box-area', type=float, default=400,  help='filter out tiny boxes')
    parser.add_argument(
        "--aspect_ratio_thresh", type=float, default=1.6,
        help="threshold for filtering out boxes of which aspect ratio are above the given value."
    )
    parser.add_argument('--limit', type=int, default=10, help='limit people number')
    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


class Predictor(object):
    def __init__(
        self,
        model,
        # exp,
        num_classes, conf_thresh, nms_thresh, test_size,
        device="cpu",
        fp16=False
    ):
        self.model = model
        self.num_classes = num_classes  # 1
        self.confthre = conf_thresh
        self.nmsthre = nms_thresh
        self.test_size = test_size
        self.device = device
        self.fp16 = fp16

    def inference(self, img, timer):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        img = letterbox(img, self.test_size)[0]
        # print('letterbox:', img.shape)
        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to('cuda' if self.device == "gpu" else 'cpu')
        img = img.half() if self.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        # print(img)
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim

        with torch.no_grad():
            timer.tic()
            # print('image shape:', img.size())
            # print(img)
            outputs = self.model(img, False, False)
            # if self.decoder is not None:
            #     outputs = self.decoder(outputs, dtype=outputs.type())
            # print('before nms:', outputs.size())
            outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
            # print('after nms:', outputs[0].size())
            outputs = outputs[0].cpu().numpy()
            heads = []
            faces = []
            for output in outputs:
                if output[6] == 0:
                    heads.append(output)
                else:
                    faces.append(output)
            # heads = [output for output in outputs if output[6] == 0]
            heads = np.array(heads)
            heads = torch.from_numpy(heads).to('cuda' if self.device == 'gpu' else 'cpu')
            # faces = [output for output in outputs if output[6] == 1]
            faces = np.array(faces)
            faces = torch.from_numpy(faces).to('cuda' if self.device == 'gpu' else 'cpu')

            outputs = [heads, faces]
            timer.toc()
        return outputs, img_info


def imageflow_demo(predictor, vis_folder, current_time, args, test_size):
    cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    save_name = args.save_name
    save_folder = os.path.join(
        vis_folder, save_name
    )
    os.makedirs(save_folder, exist_ok=True)
    if args.demo == "video":
        save_path = os.path.join(save_folder, args.path.split("/")[-1])
    else:
        save_path = os.path.join(save_folder, save_name + ".mp4")
    logger.info(f"video save_path is {save_path}")
    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )
    person_tracker = BYTETracker(args, frame_rate=30)
    face_tracker = BYTETracker(args, frame_rate=30)
    face_model, transform = model(64)
    face_model.to('cuda' if args.device == 'gpu' else 'cpu')
    face_model.eval()
    M, w_scale, h_scale = load_top_view_config('distance/distance.txt')
    timer = Timer()
    frame_id = 0
    while True:
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
        ret_val, frame = cap.read()
        # frame = cv2.resize(frame, (1280, 720))
        if ret_val:
            outputs, img_info = predictor.inference(frame, timer)

            if outputs[0] is not None:
                # people
                online_people = person_tracker.update(outputs[0], [img_info['height'], img_info['width']], test_size)
                print('People count:', len(online_people))
                people_tlwhs = []
                for t in online_people:
                    tlwh = t.tlwh
                    vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                    if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                        people_tlwhs.append([tlwh[0], tlwh[1], tlwh[2], tlwh[3], -1])
                
                for i in range(len(people_tlwhs) - 1):
                    for j in range(i + 1, len(people_tlwhs)):
                        bc1 = [people_tlwhs[i][0] + people_tlwhs[i][2] / 2, people_tlwhs[i][1] + people_tlwhs[i][3]]
                        bc2 = [people_tlwhs[j][0] + people_tlwhs[j][2] / 2, people_tlwhs[j][1] + people_tlwhs[j][3]]

                        bc1_ = cv2.perspectiveTransform(np.array([[bc1]]), M)[0][0]
                        bc2_ = cv2.perspectiveTransform(np.array([[bc2]]), M)[0][0]
                        dw = np.abs(bc1_[0] - bc2_[0]) / w_scale
                        dh = np.abs(bc1_[1] - bc2_[1]) / h_scale
                        dist = np.sqrt(dw * dw + dh * dh)
                        if dist <= 1:
                            people_tlwhs[i][4] = 0
                            people_tlwhs[j][4] = 0

            if outputs[1] is not None:
                # face
                online_faces = face_tracker.update(outputs[1], [img_info['height'], img_info['width']], test_size)
                print('Face count:', len(online_faces))
                faces_tlwhs = []
                for i, t in enumerate(online_faces):
                    tlwh = t.tlwh
                    if tlwh[2] * tlwh[3] > args.min_box_area and tlwh[0] > 0 and tlwh[1] > 0 and tlwh[0] + tlwh[2] < img_info['width'] and tlwh[1] + tlwh[3] < img_info['height']:
                        tlwh[0] = max(0, tlwh[0] - int(256 / tlwh[2]))
                        tlwh[1] = max(0, tlwh[1] - int(256 / tlwh[3]))
                        tlwh[2] = min(img_info['width'] - tlwh[0], tlwh[2] + int(512 / tlwh[2]))
                        tlwh[3] = min(img_info['height'] - tlwh[1], tlwh[3] + int(512 / tlwh[3]))
                        face = frame[int(tlwh[1]):int(tlwh[1] + tlwh[3]), int(tlwh[0]):int(tlwh[0] + tlwh[2])]
                        # cv2.imwrite(f'{frame_id}_{i}.jpg', face)
                        face = transform(face)
                        face = face.unsqueeze(0)
                        face = face.to('cuda' if args.device == 'gpu' else 'cpu')
                        prob = face_model(face)
                        # prob = prob.detach().cpu().numpy()[0][0]
                        prob = torch.softmax(prob, dim=1)[0][0].item()
                        faces_tlwhs.append([tlwh[0], tlwh[1], tlwh[2], tlwh[3], prob])
                timer.toc()
                online_im = plot_tracking(img_info['raw_img'],
                                          heads=people_tlwhs, faces=faces_tlwhs, limit=args.limit,
                                          frame_id=frame_id + 1, fps=1. / timer.average_time)
            else:
                timer.toc()
                online_im = img_info['raw_img']
            vid_writer.write(online_im)
            # cv2.imshow("online_im", online_im)
            # ch = cv2.waitKey(1)
            # if ch == 27 or ch == ord("q") or ch == ord("Q"):
            #     break
        else:
            break
        frame_id += 1


def main(args):
    file_name = os.path.join('runs', '')
    os.makedirs(file_name, exist_ok=True)
    vis_folder = os.path.join(file_name, "track")
    os.makedirs(vis_folder, exist_ok=True)

    logger.info("Args: {}".format(args))

    conf_thresh = args.conf
    nms_thresh = args.nms

    ckpt_file = args.ckpt
    model = DetectMultiBackend(ckpt_file, device='cuda' if args.device == 'gpu' else 'cpu')
    if args.fp16:
        model.model.half()
    model.eval()

    test_size = (608, 1088)
    predictor = Predictor(model, 2, conf_thresh, nms_thresh, test_size, args.device, args.fp16)
    current_time = time.localtime()
    imageflow_demo(predictor, vis_folder, current_time, args, test_size)


if __name__ == "__main__":
    args = make_parser().parse_args()

    main(args)
