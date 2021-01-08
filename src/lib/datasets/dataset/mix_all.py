from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
# from pycocotools.cocoeval import COCOeval
from utils.cocoeval import COCOeval
#from utils.voc_metric.pascalvoc import cal_voc_map
import numpy as np
import torch
import json
import os
import glob

import torch.utils.data as data

class MIXALL(data.Dataset):
  num_classes = 8
  default_resolution = [512, 512]
  mean = np.array([0.485, 0.456, 0.406],
                   dtype=np.float32).reshape(1, 1, 3)
  std  = np.array([0.229, 0.224, 0.225],
                   dtype=np.float32).reshape(1, 1, 3)
  
  def __init__(self, opt, split):
    super(MIXALL, self).__init__()
    self.data_dir = os.path.join(opt.data_dir, 'mix_all')
    self.img_dir = os.path.join(self.data_dir, 'image')                 # 这个在载入图片的时候有用
    _ann_name = {'train': 'train', 'val': 'val', 'test': 'test'}
    self.annot_path = os.path.join(
      self.data_dir, 'annotations', 
      '{}_cal_face.json').format(_ann_name[split])

    # self.annot_path = os.path.join(
    #   self.data_dir, 'annotations', 
    #   'cal_face.json')
    self.max_objs = 50
    self.class_name = ['__background__', '1_Body_Adult', '2_Body_Child1-6', '3_Body_Child0-1_Normal', '4_Body_Child0-1_Lie', \
                      '5_Face_Adult', '6_Face_Child1-6', '7_Face_Child0-1_Normal', '8_Face_Child0-1_Covered']
    self._valid_ids = np.arange(1, 21, dtype=np.int32)
    self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}
    # print("self.cat_idsself.cat_idsself.cat_ids    ", self.cat_ids)
    self._data_rng = np.random.RandomState(123)
    self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                             dtype=np.float32)
    self._eig_vec = np.array([
        [-0.58752847, -0.69563484, 0.41340352],
        [-0.5832747, 0.00994535, -0.81221408],
        [-0.56089297, 0.71832671, 0.41158938]
    ], dtype=np.float32)
    self.split = split
    self.opt = opt

    print('==> initializing pascal {} data.'.format(_ann_name[split]))
    self.coco = coco.COCO(self.annot_path)
    self.images = sorted(self.coco.getImgIds())
    self.num_samples = len(self.images)

    print('Loaded {} {} samples'.format(split, self.num_samples))

  def _to_float(self, x):
    return float("{:.2f}".format(x))

  # def convert_eval_format(self, all_bboxes):
  #   detections = [[[] for __ in range(self.num_samples)] \
  #                 for _ in range(self.num_classes + 1)]
  #   for i in range(self.num_samples):
  #     img_id = self.images[i]
  #     for j in range(1, self.num_classes + 1):

  #       if isinstance(all_bboxes[img_id][j], np.ndarray):
  #         detections[j][i] = all_bboxes[img_id][j].tolist()
  #       else:
  #         detections[j][i] = all_bboxes[img_id][j]
  #   return detections
  def convert_eval_format(self, all_bboxes):
    # import pdb; pdb.set_trace()
    detections = []
    for image_id in all_bboxes:
      for cls_ind in all_bboxes[image_id]:
        category_id = self._valid_ids[cls_ind - 1]
        for bbox in all_bboxes[image_id][cls_ind]:
          bbox[2] -= bbox[0]
          bbox[3] -= bbox[1]
          score = bbox[4]
          bbox_out  = list(map(self._to_float, bbox[0:4]))

          detection = {
              "image_id": int(image_id),
              "category_id": int(category_id),
              "bbox": bbox_out,
              "score": float("{:.2f}".format(score))
          }
          if len(bbox) > 5:
              extreme_points = list(map(self._to_float, bbox[5:13]))
              detection["extreme_points"] = extreme_points
          detections.append(detection)
    return detections

  def __len__(self):
    return self.num_samples

  def save_results(self, results, save_dir):
    json.dump(self.convert_eval_format(results), 
              open('{}/results.json'.format(save_dir), 'w'))



  def parser_all_boxes_infos(self, anns, mtype='gt'):
    fix_list = []
    for ann in anns:
      x, y, w, h = ann["bbox"][0], ann["bbox"][1], ann["bbox"][2], ann["bbox"][3]
      if any(a["image_id"] == ann["image_id"] for a in fix_list):
        exist_ann = next((a for a in fix_list if a["image_id"] == ann["image_id"]), None)
        if mtype=='gt':
          exist_ann["all_infos"].append([ann["category_id"], x, y, w, h])
        else:
          exist_ann["all_infos"].append([ann["category_id"], ann["score"], x, y, w, h])
      else:
        if mtype=='gt':
          fix_list.append({"image_id":ann["image_id"],"all_infos":[[ ann["category_id"], x, y, w, h]]})
        else:
          fix_list.append({"image_id":ann["image_id"],"all_infos":[[ ann["category_id"], ann["score"], x, y, w, h ]]})
    return fix_list


  def save_infos_to_txt(self, infos_list, save_folder):
    for a_predict in infos_list:

      image_name = os.path.join(save_folder, str(a_predict["image_id"])+".txt")
      with open(image_name, "w") as f:
        all_infos = a_predict["all_infos"]
        for infos in all_infos:
          for idx, info in enumerate(infos):
            if info < 0:
              info = 0
            info = str(info)
            if idx == 0:
              f.write(info)
            elif idx == (len(infos)-1):
              f.write(' '+info+'\n')
            else:
              f.write(' '+info)
  def remove_folder_files(self, folder_name):

    files = glob.glob(folder_name+'/*')
    for f in files:
      os.remove(f)


  def run_eval(self, results, save_dir):

    self.save_results(results, save_dir)
    coco_dets = self.coco.loadRes('{}/results.json'.format(save_dir))
    coco_eval = COCOeval(self.coco, coco_dets, "bbox")
    # coco_eval.params.catIds = [1]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval.stats[1]


  # def run_eval(self, results, save_dir):

  #   self.save_results(results, save_dir)
  #   pd_json_file = '{}/results.json'.format(save_dir)
  #   gt_json_file = os.path.join(self.data_dir, 'annotations', 'test_cal_face.json')
  #   currentPath = os.path.dirname(os.path.abspath(__file__))
  #   txt_detections_folder = os.path.join(currentPath, '..', '..', '..', 'test_result', 'detections')
  #   txt_groundtruths_folder = os.path.join(currentPath, '..', '..', '..', 'test_result', 'groundtruths')
  #   print("!!!!!!!txt_detections_folder  ", txt_detections_folder)
  #   self.remove_folder_files(txt_detections_folder)
  #   self.remove_folder_files(txt_groundtruths_folder)
  #   with open(pd_json_file) as json_file:
  #     pd_json = json.load(json_file)

  #   with open(gt_json_file) as json_file:
  #     gt_json = json.load(json_file)
  #   predict_list = self.parser_all_boxes_infos(pd_json, 'pd')
  #   gt_json = gt_json['annotations']
  #   gt_list = self.parser_all_boxes_infos(gt_json)
  #   print(len(predict_list))
  #   print(len(gt_list))
  #   self.save_infos_to_txt(predict_list, txt_detections_folder)
  #   self.save_infos_to_txt(gt_list, txt_groundtruths_folder)
  #   cal_voc_map()
  #   self.remove_folder_files(txt_detections_folder)
  #   self.remove_folder_files(txt_groundtruths_folder)



