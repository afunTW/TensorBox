"""[summary] 
github repo - https://github.com/Russell91/TensorBox/tree/df3fca93f8eedf8314772f28bd0b17f3d8bc6b7a

* Make sure you are running by python2.7 + Opencv 2.4

test usage:
python generate_bbox.py \
    --gpu 0 \
    --weights output/lstm_resnet_beetle_rezoom_2018_05_18_17.33/save.ckpt-1300000 \
    --test-boxes data/images/val1.json \
    --video-root data/test/ \
    --video-type mp4 \
    --skip-nframe 60

Returns:
    [type] -- [description]
"""

import argparse
import json
import logging
import os
import sys
from tqdm import tqdm
from glob import glob
from datetime import datetime
from pprint import pformat

import tensorflow as tf

import cv2
from train import build_forward
from utils.annolist import AnnotationLib as al
from utils.train_utils import add_rectangles, rescale_boxes


def argparser():
    """[summary]
        --weights default used to
            output/lstm_resnet_rezoom_beetle_2017_10_11_14.38/save.ckpt-1000000
        --test_boxes default used to
            data/beetle/val_boxes.json or data/images/val1.json
        --video-root defautl used to
            /data/put_data/kai/beetle or data/video/
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', dest='weights')
    parser.add_argument('--expname', dest='expname', default='')
    parser.add_argument('--test-boxes', dest='test_boxes', required=True)
    parser.add_argument('--gpu', dest='gpu', required=True)
    parser.add_argument('--gpu-fraction', dest='gpu_fraction', default=0.45, type=float)
    parser.add_argument('--output-dir', dest='outputdir', default='output_video')
    parser.add_argument('--video-type', dest='video_type', default='avi')
    parser.add_argument('--skip-nframe', dest='skip_nframe', default=1, type=int)
    parser.add_argument('--frame-count', dest='frame_count', type=int)
    parser.add_argument('--iou-threshold', dest='iou_threshold', default=0.5, type=float)
    parser.add_argument('--tau', dest='tau', default=0.25, type=float)
    parser.add_argument('--min-conf', dest='min_conf', default=0.7, type=float)
    parser.add_argument('--suppressed', dest='suppressed', action='store_true')
    parser.add_argument('--no-suppressed', dest='suppressed', action='store_false')
    parser.set_defaults(suppressed=True)
    parser.add_argument('--video-root', dest='video_root', required=True)
    return parser

def get_image_dir(W_path, expname, test_boxes_path):
    weights_iteration = int(W_path.split('-')[-1])
    expname = '_' + expname if expname else ''
    image_dir = '{}/images_{}_{}{}'.format(
        os.path.dirname(W_path),
        os.path.basename(test_boxes_path)[:-5],
        weights_iteration,
        expname
    )
    return image_dir

def main(args, logger):
    # setup
    logger.info(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = args.gpu_fraction

    # path
    path_hypes_file = '{}/hypes.json'.format(os.path.dirname(args.weights))
    with open(path_hypes_file, 'r') as f:
        H = json.load(f)
    expname = args.expname + '_' if args.expname else ''
    pred_boxed = '{}.{}{}'.format(args.weights, expname, os.path.basename(args.test_boxes))
    test_boxed = '{}.gt_{}{}'.format(args.weights, expname, os.path.basename(args.test_boxes))

    # graph
    tf.reset_default_graph()
    H['grid_width'] = H['image_width'] / H['region_size']
    H['grid_height'] = H['image_height'] / H['region_size']
    X = tf.placeholder(tf.float32, name='input', shape=(H['image_height'], H['image_width'], 3))
    if H['use_rezoom']:
        (pred_boxes,
         pred_logits,
         pred_confidences,
         pred_confs_deltas,
         pred_boxes_deltas) = build_forward(H, tf.expand_dims(X, 0), 'test', reuse=None)
        grid_area = H['grid_height'] * H['grid_width']
        reshape_shape = [grid_area * H['rnn_len'], 2]
        pred_confidences = tf.reshape(
            tf.nn.softmax(
                tf.reshape(pred_confs_deltas, reshape_shape
            )),
            reshape_shape
        )
        pred_boxes = pred_boxes + pred_boxes_deltas if H['reregress'] else pred_boxes
    else:
        (pred_boxes,
         pred_logits,
         pred_confidences) = build_forward(H, tf.expand_dims(X, 0), 'test', reuse=None)
    
    # load checkopint
    saver = tf.train.Saver()
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, args.weights)

        pred_annolist = al.AnnoList()
        data_dir = os.path.dirname(args.test_boxes)
        image_dir = get_image_dir(args.weights, args.expname, args.test_boxes)
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)

        video_paths = []
        for d in os.listdir(args.video_root):
            pattern = os.path.join(args.video_root, d, '*.{}'.format(args.video_type))
            logger.info(pattern)
            video_paths.extend(glob(pattern))

        for v in video_paths:
            txtname = '.'.join(v.split('.')[:-1]) + '.txt'
            if os.path.isfile(txtname):
                logger.info('{} existed, pass'.format(txtname))
                continue
            
            logger.info('Predicting {}'.format(os.path.basename(v)))
            outputdir = os.path.join(
                args.outputdir,
                '{}-skip-{}-count-{}'.format(
                    datetime.now().strftime('%Y%m%d'),
                    args.skip_nframe,
                    args.frame_count
                )
            )
            if not os.path.exists(outputdir):
                os.makedirs(outputdir)
            
            # video operation
            cap = cv2.VideoCapture(v)
            total_frame = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
            fourcc = cv2.cv.CV_FOURCC(*'XVID')
            filename = 'detected_{}'.format(os.path.basename(v))
            out = cv2.VideoWriter(os.path.join(outputdir, filename), fourcc, 15, (640, 480))
            
            data = []
            logger.info('total {} skip {}'.format(total_frame, args.skip_nframe))
            for frame_idx in tqdm(range(0, total_frame, args.skip_nframe)):
                if args.frame_count and len(data) > args.frame_count:
                    break
                if not cap.isOpened():
                    logger.error('{} is close'.format(os.path.basename(v)))
                ok, frame = cap.read()
                
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = cv2.resize(frame, (H['image_width'], H['image_height']))
                (np_pred_boxes, np_pred_confidences) = sess.run(
                    [pred_boxes, pred_confidences], feed_dict={X: image})
                pred_anno = al.Annotation()
                new_img, rects = add_rectangles(
                    H, [image], np_pred_confidences, np_pred_boxes,
                    use_stitching=True,
                    rnn_len=H['rnn_len'],
                    min_conf=args.min_conf,
                    tau=args.tau,
                    show_suppressed=args.suppressed
                )

                pred_anno.rects = rects
                pred_anno.imagePath = os.path.abspath(data_dir)
                pred_anno = rescale_boxes(
                    (H["image_height"], H["image_width"]),
                    pred_anno,
                    frame.shape[0],
                    frame.shape[1]
                )

                results = []
                for r in pred_anno.rects:
                    results.append([max(r.y1, 0), max(r.x1, 0), max(r.y2, 0), max(r.x2, 0), r.score])
                data.append(str([frame_idx+1, results]) + '\n')
                pred_annolist.append(pred_anno)
                out.write(new_img)
            
            cap.release()
            out.release()

            with open(txtname, 'w+') as f:
                f.writelines(data)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(filename)12s:L%(lineno)3s [%(levelname)8s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        stream=sys.stdout
    )
    logger = logging.getLogger()
    parser = argparser()
    main(parser.parse_args(), logger)
