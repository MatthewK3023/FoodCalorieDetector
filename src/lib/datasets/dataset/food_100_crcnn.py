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

class FOOD100CRCNN(data.Dataset):
  num_classes = 100
  default_resolution = [512, 512]
  mean = np.array([0.485, 0.456, 0.406],
                   dtype=np.float32).reshape(1, 1, 3)
  std  = np.array([0.229, 0.224, 0.225],
                   dtype=np.float32).reshape(1, 1, 3)
  
  def __init__(self, opt, split):
    super(FOOD100CRCNN, self).__init__()
    self.data_dir = os.path.join(opt.data_dir, 'food_100_crcnn')
    self.img_dir = os.path.join(self.data_dir, 'image')                 # 这个在载入图片的时候有用
    _ann_name = {'train': 'train', 'val': 'val', 'test': 'test'}
    self.annot_path = os.path.join(
      self.data_dir, 'annotations', 
      '{}_food.json').format(_ann_name[split])
    # self.annot_path = os.path.join(
    #   self.data_dir, 'annotations', 
    #   'food.json')
    self.max_objs = 120
    self.class_name = ['__background__', '1 rice', '2 eels on rice', '3 pilaf', "4 chicken-n-egg on rice",
                       '5 pork cutlet on rice', '6 beef curry', '7 sushi', '8 chicken rice', 
                       '9 fried rice', '10 tempura bowl', '11 bibimbap', '12 toast',
                       '13 croissant', '14 roll bread', '15 raisin bread', '16 chip butty',
                       '17 hamburger', '18 pizza', '19 sandwiches', '20 udon noodle',
                       '21 tempura udon', '22 soba noodle', '23 ramen noodle', '24 beef noodle',
                       '25 tensin noodle', '26 fried noodle', '27 spaghetti', '28 Japanese-style pancake',
                       '29 takoyaki', '30 gratin', '31 sauteed vegetables', '32 croquette',
                       '33 grilled eggplant', '34 sauteed spinach', '35 vegetable tempura', '36 miso soup',
                       '37 potage', '38 sausage', '39 oden', '40 omelet',
                       '41 ganmodoki', '42 jiaozi', '43_stew', '44 teriyaki grilled fish',
                       '45 fried fish', '46 grilled salmon', '47 salmon meuniere ', '48 sashimi',
                       '49 grilled pacific saury ', '50 sukiyaki', '51 sweet and sour pork', '52 lightly roasted fish',
                       '53 steamed egg hotchpotch', '54 tempura', '55 fried chicken', '56 sirloin cutlet ',
                       '57 nanbanzuke', '58 boiled fish', '59 seasoned beef with potatoes', '60 hambarg steak',
                       '61 beef steak', '62 dried fish', '63 ginger pork saute', '64 spicy chili-flavored tofu',
                       '65 yakitori', '66 cabbage roll', '67 rolled omelet', '68 egg sunny-side up',
                       '69 fermented soybeans', '70 cold tofu', '71 egg roll', '72 chilled noodle',
                       '73 stir-fried beef and peppers', '74 simmered pork', '75 boiled chicken and vegetables', '76 sashimi bowl', 
                       '77 sushi bowl', '78 fish-shaped pancake with bean jam', '79 shrimp with chill source', '80 roast chicken',
                       '81 steamed meat dumpling', '82 omelet with fried rice', '83 cutlet curry', '84 spaghetti meat sauce',
                       '85 fried shrimp', '86 potato salad', '87 green salad', '88 macaroni salad',
                       '89 Japanese tofu and vegetable chowder', '90 pork miso soup', '91 chinese soup', '92 beef bowl',
                       '93 kinpira-style sauteed burdock', '94 rice ball', '95 pizza toast', '96 dipping noodles',
                       '97 hot dog', '98 french fries', '99 mixed rice', '100 goya chanpuru']

    self._valid_ids = np.arange(1, 101, dtype=np.int32)
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
    print('self.annot_path: ', self.annot_path)
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
    save_summarize = self.opt.load_model
    save_summarize = '/'.join([s for s in save_summarize.split('/')][:-1])
    save_summarize = os.path.join(save_summarize, 'result.json')
    print('save_summarize : ', save_summarize )
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize(save_summarize=save_summarize)

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



