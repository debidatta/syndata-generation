import argparse
import glob
import sys
import os
from xml.etree.ElementTree import Element, SubElement, tostring
import xml.dom.minidom
import cv2
import numpy as np
import random
from PIL import Image
import scipy
from multiprocessing import Pool
from functools import partial
import signal
import time

from defaults import *
sys.path.insert(0, POISSON_BLENDING_DIR)
from pb import *
import math
from pyblur import *
from collections import namedtuple

Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')

def randomAngle(kerneldim):
    kernelCenter = int(math.floor(kerneldim/2))
    numDistinctLines = kernelCenter * 4
    validLineAngles = np.linspace(0,180, numDistinctLines, endpoint = False)
    angleIdx = np.random.randint(0, len(validLineAngles))
    return int(validLineAngles[angleIdx])

def LinearMotionBlur3C(img):
    lineLengths = [3,5,7,9]
    lineTypes = ["right", "left", "full"]
    lineLengthIdx = np.random.randint(0, len(lineLengths))
    lineTypeIdx = np.random.randint(0, len(lineTypes)) 
    lineLength = lineLengths[lineLengthIdx]
    lineType = lineTypes[lineTypeIdx]
    lineAngle = randomAngle(lineLength)
    blurred_img = img
    for i in xrange(3):
        blurred_img[:,:,i] = PIL2array1C(LinearMotionBlur(img[:,:,i], lineLength, lineAngle, lineType))
    blurred_img = Image.fromarray(blurred_img, 'RGB')
    return blurred_img

def overlap(a, b):
    dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
    dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
    
    if (dx>=0) and (dy>=0) and float(dx*dy) > MAX_ALLOWED_IOU*(a.xmax-a.xmin)*(a.ymax-a.ymin):
        return True
    else:
        return False
 
def parse_args():
    parser = argparse.ArgumentParser(description="Create dataset with different augmentations")
    parser.add_argument("root",
      help="The root directory which contains the images and annotations.")
    parser.add_argument("exp",
      help="The directory where images and annotation lists will be created.")
    parser.add_argument("--selected",
      help="Keep only selected instances in the test dataset. Default is to keep all instances in the roo directory", action="store_true")
    parser.add_argument("--scale",
      help="Add scale augmentation.Default is to not add scale augmentation.", action="store_true")
    parser.add_argument("--rotation",
      help="Add rotation augmentation.Default is to not add rotation augmentation.", action="store_true")
    parser.add_argument("--num",
      help="Number of times each image will be in dataset", default=1, type=int)
    parser.add_argument("--dontocclude",
      help="Add objects without occlusion. Default is to produce occlusions", action="store_true")
    parser.add_argument("--add_distractors",
      help="Add distractors objects. Default is to not use distractors", action="store_true")
    args = parser.parse_args()
    return args

def get_list_of_images(root_dir, N=1):
    img_list = glob.glob(os.path.join(root_dir, '*/*.jpg'))
    img_list_f = []
    for i in xrange(N):
        img_list_f = img_list_f + random.sample(img_list, len(img_list))
    return img_list_f

def get_mask_file(img_file):
    # Write custom code for getting mask file
    mask_file = img_file.replace('.jpg','.pbm')
    return mask_file

def get_labels(imgs):
    labels = []
    for img_file in imgs:
        label = img_file.split('/')[-2]
        labels.append(label)
    return labels

def get_annotation_from_mask_file(mask_file, scale=1.0):
    if os.path.exists(mask_file):
        mask = cv2.imread(mask_file)
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if len(np.where(rows)[0]) > 0:
            ymin, ymax = np.where(rows)[0][[0, -1]]
            xmin, xmax = np.where(cols)[0][[0, -1]]
            return int(scale*xmin), int(scale*xmax), int(scale*ymin), int(scale*ymax)
        else:
            return -1, -1, -1, -1
    else:
        print "%s not found. Using empty mask instead."%mask_file
        return -1, -1, -1, -1

def get_annotation_from_mask(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if len(np.where(rows)[0]) > 0:
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]
        return xmin, xmax, ymin, ymax
    else:
        return -1, -1, -1, -1

def write_imageset_file(exp_dir, img_files, anno_files):
    with open(os.path.join(exp_dir,'train.txt'),'w') as f:
        for i in xrange(len(img_files)):
            f.write('%s %s\n'%(img_files[i], anno_files[i]))

def write_labels_file(exp_dir, labels):
    unique_labels = ['__background__'] + sorted(set(labels))
    with open(os.path.join(exp_dir,'labels.txt'),'w') as f:
        for i, label in enumerate(unique_labels):
            f.write('%s %s\n'%(i, label))

def keep_selected_labels(img_files, labels):
    with open(SELECTED_LIST_FILE) as f:
        selected_labels = [x.strip() for x in f.readlines()]
    new_img_files = []
    new_labels = []
    for i in xrange(len(img_files)):
        if labels[i] in selected_labels:
            new_img_files.append(img_files[i])
            new_labels.append(labels[i])
    return new_img_files, new_labels

def PIL2array1C(img):
    return np.array(img.getdata(),
                    np.uint8).reshape(img.size[1], img.size[0])

def PIL2array3C(img):
    return np.array(img.getdata(),
                    np.uint8).reshape(img.size[1], img.size[0], 3)

def create_image_anno_wrapper(args, w=WIDTH, h=HEIGHT, scale_augment=False, rotation_augment=False, blending_list=['none'], dontocclude=False):
   return create_image_anno(*args, w=w, h=h, scale_augment=scale_augment, rotation_augment=rotation_augment, blending_list=blending_list, dontocclude=dontocclude)

def create_image_anno(objects, distractor_objects, img_file, anno_file, bg_file,  w=WIDTH, h=HEIGHT, scale_augment=False, rotation_augment=False, blending_list=['none'], dontocclude=False):
    if 'none' not in img_file:
        return 
    
    print "Working on %s" % img_file
    if os.path.exists(anno_file):
        return anno_file
    
    all_objects = objects + distractor_objects
    while True:
        top = Element('annotation')
        background = Image.open(bg_file)
        background = background.resize((w, h), Image.ANTIALIAS)
        backgrounds = []
        for i in xrange(len(blending_list)):
            backgrounds.append(background.copy())
        
        if dontocclude:
            already_syn = []
        for idx, obj in enumerate(all_objects):
           foreground = Image.open(obj[0])
           xmin, xmax, ymin, ymax = get_annotation_from_mask_file(get_mask_file(obj[0]))
           if xmin == -1 or ymin == -1 or xmax-xmin < MIN_WIDTH or ymax-ymin < MIN_HEIGHT :
               continue
           foreground = foreground.crop((xmin, ymin, xmax, ymax))
           orig_w, orig_h = foreground.size
           mask_file =  get_mask_file(obj[0])
           mask = Image.open(mask_file)
           mask = mask.crop((xmin, ymin, xmax, ymax))
           o_w, o_h = orig_w, orig_h
           if scale_augment:
                while True:
                    scale = random.uniform(MIN_SCALE, MAX_SCALE)
                    o_w, o_h = int(scale*orig_w), int(scale*orig_h)
                    if  w-o_w > 0 and h-o_h > 0 and o_w > 0 and o_h > 0:
                        break
                foreground = foreground.resize((o_w, o_h), Image.ANTIALIAS)
                mask = mask.resize((o_w, o_h), Image.ANTIALIAS)
           if rotation_augment:
               max_degrees = MAX_DEGREES  
               while True:
                   rot_degrees = random.randint(-max_degrees, max_degrees)
                   foreground_tmp = foreground.rotate(rot_degrees, expand=True)
                   mask_tmp = mask.rotate(rot_degrees, expand=True)
                   o_w, o_h = foreground_tmp.size
                   if  w-o_w > 0 and h-o_h > 0:
                        break
               mask = mask_tmp
               foreground = foreground_tmp
           
           xmin, xmax, ymin, ymax = get_annotation_from_mask(mask)
           attempt = 0
           while True:
               attempt +=1
               x = random.randint(int(-MAX_TRUNCATION_FRACTION*o_w), int(w-o_w+MAX_TRUNCATION_FRACTION*o_w))
               y = random.randint(int(-MAX_TRUNCATION_FRACTION*o_h), int(h-o_h+MAX_TRUNCATION_FRACTION*o_h))
               if dontocclude:
                   found = True
                   for prev in already_syn:
                       ra = Rectangle(prev[0], prev[2], prev[1], prev[3])
                       rb = Rectangle(x+xmin, y+ymin, x+xmax, y+ymax)
                       if overlap(ra, rb):
                             found = False
                             break
                   if found:
                      break
               else:
                   break
               if attempt == MAX_ATTEMPTS_TO_SYNTHESIZE:
                   break
           if dontocclude:
               already_syn.append([x+xmin, x+xmax, y+ymin, y+ymax])
           for i in xrange(len(blending_list)):
               if blending_list[i] == 'none' or blending_list[i] == 'motion':
                   backgrounds[i].paste(foreground, (x, y), mask)
               elif blending_list[i] == 'poisson':
                  offset = (y, x)
                  img_mask = PIL2array1C(mask)
                  img_src = PIL2array3C(foreground).astype(np.float64)
                  img_target = PIL2array3C(backgrounds[i])
                  img_mask, img_src, offset_adj \
                       = create_mask(img_mask.astype(np.float64),
                          img_target, img_src, offset=offset)
                  background_array = poisson_blend(img_mask, img_src, img_target,
                                    method='normal', offset_adj=offset_adj)
                  backgrounds[i] = Image.fromarray(background_array, 'RGB') 
               elif blending_list[i] == 'gaussian':
                  backgrounds[i].paste(foreground, (x, y), Image.fromarray(cv2.GaussianBlur(PIL2array1C(mask),(5,5),2)))
               elif blending_list[i] == 'box':
                  backgrounds[i].paste(foreground, (x, y), Image.fromarray(cv2.blur(PIL2array1C(mask),(3,3))))
           if idx >= len(objects):
               continue 
           object_root = SubElement(top, 'object')
           object_type = obj[1]
           object_type_entry = SubElement(object_root, 'name')
           object_type_entry.text = str(object_type)
           object_bndbox_entry = SubElement(object_root, 'bndbox')
           x_min_entry = SubElement(object_bndbox_entry, 'xmin')
           x_min_entry.text = '%d'%(max(1,x+xmin))
           x_max_entry = SubElement(object_bndbox_entry, 'xmax')
           x_max_entry.text = '%d'%(min(w,x+xmax))
           y_min_entry = SubElement(object_bndbox_entry, 'ymin')
           y_min_entry.text = '%d'%(max(1,y+ymin))
           y_max_entry = SubElement(object_bndbox_entry, 'ymax')
           y_max_entry.text = '%d'%(min(h,y+ymax))
           difficult_entry = SubElement(object_root, 'difficult')
           difficult_entry.text = '0' # Add heuristic to estimate difficulty later on
        if attempt == MAX_ATTEMPTS_TO_SYNTHESIZE:
           continue
        else:
           break
    for i in xrange(len(blending_list)):
        if blending_list[i] == 'motion':
            backgrounds[i] = LinearMotionBlur3C(PIL2array3C(backgrounds[i]))
        backgrounds[i].save(img_file.replace('none', blending_list[i]))

    xmlstr = xml.dom.minidom.parseString(tostring(top)).toprettyxml(indent="    ")
    with open(anno_file, "w") as f:
        f.write(xmlstr)
   
def gen_syn_data(img_files, labels, img_dir, anno_dir, scale_augment, rotation_augment, dontocclude, add_distractors):
    w = WIDTH
    h = HEIGHT
    background_dir = BACKGROUND_DIR
    background_files = glob.glob(os.path.join(background_dir, BACKGROUND_GLOB_STRING))
   
    print "Number of background images : %s"%len(background_files) 
    img_labels = zip(img_files, labels)
    random.shuffle(img_labels)

    if add_distractors:
        with open(DISTRACTOR_LIST_FILE) as f:
            distractor_labels = [x.strip() for x in f.readlines()]

        distractor_list = []
        for distractor_label in distractor_labels:
            distractor_list += glob.glob(os.path.join(DISTRACTOR_DIR, distractor_label, DISTRACTOR_GLOB_STRING))

        distractor_files = zip(distractor_list, len(distractor_list)*[None])
        random.shuffle(distractor_files)

    idx = 0
    img_files = []
    anno_files = []
    params_list = []
    while len(img_labels) > 0:
        # Get list of objects
        objects = []
        n = min(random.randint(MIN_NO_OF_OBJECTS, MAX_NO_OF_OBJECTS), len(img_labels))
        for i in xrange(n):
            objects.append(img_labels.pop())
        # Get list of distractor objects 
        distractor_objects = []
        if add_distractors:
            n = min(random.randint(MIN_NO_OF_DISTRACTOR_OBJECTS, MAX_NO_OF_DISTRACTOR_OBJECTS), len(distractor_files))
            for i in xrange(n):
                distractor_objects.append(random.choice(distractor_files))
        idx += 1
        bg_file = random.choice(background_files)
        for blur in BLENDING_LIST:
            img_file = os.path.join(img_dir, '%i_%s.jpg'%(idx,blur))
            anno_file = os.path.join(anno_dir, '%i.xml'%idx)
            params = (objects, distractor_objects, img_file, anno_file, bg_file)
            params_list.append(params)
            img_files.append(img_file)
            anno_files.append(anno_file)

    partial_func = partial(create_image_anno_wrapper, w=w, h=h, scale_augment=scale_augment, rotation_augment=rotation_augment, blending_list=BLENDING_LIST, dontocclude=dontocclude) 
    p = Pool(NUMBER_OF_WORKERS, init_worker)
    try:
        p.map(partial_func, params_list)
    except KeyboardInterrupt:
        print "....\nCaught KeyboardInterrupt, terminating workers"
        p.terminate()
    else:
        p.close()
    p.join()
    return img_files, anno_files

def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)
 
def generate_synthetic_dataset(args):
    img_files = get_list_of_images(args.root, args.num) 
    labels = get_labels(img_files)

    if args.selected:
       img_files, labels = keep_selected_labels(img_files, labels)

    if not os.path.exists(args.exp):
        os.makedirs(args.exp)
    
    write_labels_file(args.exp, labels)

    anno_dir = os.path.join(args.exp, 'annotations')
    img_dir = os.path.join(args.exp, 'images')
    if not os.path.exists(os.path.join(anno_dir)):
        os.makedirs(anno_dir)
    if not os.path.exists(os.path.join(img_dir)):
        os.makedirs(img_dir)
    
    syn_img_files, anno_files = gen_syn_data(img_files, labels, img_dir, anno_dir, args.scale, args.rotation, args.dontocclude, args.add_distractors)
    write_imageset_file(args.exp, syn_img_files, anno_files)

if __name__ == '__main__':
    args = parse_args()
    generate_synthetic_dataset(args)
