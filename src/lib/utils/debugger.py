from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
from .ddd_utils import compute_box_3d, project_to_image, draw_box_3d

class Debugger(object):
  def __init__(self, ipynb=False, theme='black', 
               num_classes=-1, dataset=None, down_ratio=4):
    #self.videoWriter = cv2.VideoWriter('test1.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 480))
    self.ipynb = ipynb
    if not self.ipynb:
      import matplotlib.pyplot as plt
      self.plt = plt
    self.imgs = {}
    self.theme = theme
    colors = [(color_list[_]).astype(np.uint8) \
            for _ in range(len(color_list))]
    self.colors = np.array(colors, dtype=np.uint8).reshape(len(colors), 1, 1, 3)
    if self.theme == 'white':
      self.colors = self.colors.reshape(-1)[::-1].reshape(len(colors), 1, 1, 3)
      self.colors = np.clip(self.colors, 0., 0.6 * 255).astype(np.uint8)
    self.dim_scale = 1
    if dataset in ['food100', 'food100_crcnn', 'food100_frcnn', 'food100_soft2', 'food100_soft3']:
      self.names = food_class_name
    elif dataset == 'mixall':
      self.names = mixall_class_name
      
    num_classes = len(self.names)
    self.down_ratio=down_ratio
    # for bird view
    self.world_size = 64
    self.out_size = 384

  def add_img(self, img, img_id='default', revert_color=False):
    if revert_color:
      img = 255 - img
    self.imgs[img_id] = img.copy()
  
  def add_mask(self, mask, bg, imgId = 'default', trans = 0.8):
    self.imgs[imgId] = (mask.reshape(
      mask.shape[0], mask.shape[1], 1) * 255 * trans + \
      bg * (1 - trans)).astype(np.uint8)
  

  def return_img(self, img_id='multi_pose'):
    return self.imgs[img_id]

  def show_img(self, pause = False, imgId = 'default'):
    cv2.imshow('{}'.format(imgId), self.imgs[imgId])
    if pause:
      cv2.waitKey()
  
  def add_blend_img(self, back, fore, img_id='blend', trans=0.7):
    if self.theme == 'white':
      fore = 255 - fore
    if fore.shape[0] != back.shape[0] or fore.shape[0] != back.shape[1]:
      fore = cv2.resize(fore, (back.shape[1], back.shape[0]))
    if len(fore.shape) == 2:
      fore = fore.reshape(fore.shape[0], fore.shape[1], 1)
    self.imgs[img_id] = (back * (1. - trans) + fore * trans)
    self.imgs[img_id][self.imgs[img_id] > 255] = 255
    self.imgs[img_id][self.imgs[img_id] < 0] = 0
    self.imgs[img_id] = self.imgs[img_id].astype(np.uint8).copy()

  '''
  # slow version
  def gen_colormap(self, img, output_res=None):
    # num_classes = len(self.colors)
    img[img < 0] = 0
    h, w = img.shape[1], img.shape[2]
    if output_res is None:
      output_res = (h * self.down_ratio, w * self.down_ratio)
    color_map = np.zeros((output_res[0], output_res[1], 3), dtype=np.uint8)
    for i in range(img.shape[0]):
      resized = cv2.resize(img[i], (output_res[1], output_res[0]))
      resized = resized.reshape(output_res[0], output_res[1], 1)
      cl = self.colors[i] if not (self.theme == 'white') \
           else 255 - self.colors[i]
      color_map = np.maximum(color_map, (resized * cl).astype(np.uint8))
    return color_map
    '''

  
  def gen_colormap(self, img, output_res=None):
    img = img.copy()
    c, h, w = img.shape[0], img.shape[1], img.shape[2]
    if output_res is None:
      output_res = (h * self.down_ratio, w * self.down_ratio)
    img = img.transpose(1, 2, 0).reshape(h, w, c, 1).astype(np.float32)
    colors = np.array(
      self.colors, dtype=np.float32).reshape(-1, 3)[:c].reshape(1, 1, c, 3)
    if self.theme == 'white':
      colors = 255 - colors
    color_map = (img * colors).max(axis=2).astype(np.uint8)
    color_map = cv2.resize(color_map, (output_res[0], output_res[1]))
    return color_map
    
  '''
  # slow
  def gen_colormap_hp(self, img, output_res=None):
    # num_classes = len(self.colors)
    # img[img < 0] = 0
    h, w = img.shape[1], img.shape[2]
    if output_res is None:
      output_res = (h * self.down_ratio, w * self.down_ratio)
    color_map = np.zeros((output_res[0], output_res[1], 3), dtype=np.uint8)
    for i in range(img.shape[0]):
      resized = cv2.resize(img[i], (output_res[1], output_res[0]))
      resized = resized.reshape(output_res[0], output_res[1], 1)
      cl =  self.colors_hp[i] if not (self.theme == 'white') else \
        (255 - np.array(self.colors_hp[i]))
      color_map = np.maximum(color_map, (resized * cl).astype(np.uint8))
    return color_map
  '''
  def gen_colormap_hp(self, img, output_res=None):
    c, h, w = img.shape[0], img.shape[1], img.shape[2]
    if output_res is None:
      output_res = (h * self.down_ratio, w * self.down_ratio)
    img = img.transpose(1, 2, 0).reshape(h, w, c, 1).astype(np.float32)
    colors = np.array(
      self.colors_hp, dtype=np.float32).reshape(-1, 3)[:c].reshape(1, 1, c, 3)
    if self.theme == 'white':
      colors = 255 - colors
    color_map = (img * colors).max(axis=2).astype(np.uint8)
    color_map = cv2.resize(color_map, (output_res[0], output_res[1]))
    return color_map


  def add_rect(self, rect1, rect2, c, conf=1, img_id='default'): 
    cv2.rectangle(
      self.imgs[img_id], (rect1[0], rect1[1]), (rect2[0], rect2[1]), c, 2)
    if conf < 1:
      cv2.circle(self.imgs[img_id], (rect1[0], rect1[1]), int(10 * conf), c, 1)
      cv2.circle(self.imgs[img_id], (rect2[0], rect2[1]), int(10 * conf), c, 1)
      cv2.circle(self.imgs[img_id], (rect1[0], rect2[1]), int(10 * conf), c, 1)
      cv2.circle(self.imgs[img_id], (rect2[0], rect1[1]), int(10 * conf), c, 1)

  def add_coco_bbox(self, bbox, cat, conf=1, show_txt=True, img_id='default'): 
    bbox = np.array(bbox, dtype=np.int32)
    # cat = (int(cat) + 1) % 80
    cat = int(cat)
    # print('cat', cat, self.names[cat])
    c = self.colors[cat][0][0].tolist()
    if self.theme == 'white':
      c = (255 - np.array(c)).tolist()
    # txt = '{} {:.1f}'.format(self.names[cat], conf)
    txt = self.names[cat]



    font = cv2.FONT_HERSHEY_SIMPLEX
    cat_size = cv2.getTextSize(txt, font, 0.5, 2)[0]
    cv2.rectangle(
      self.imgs[img_id], (bbox[0], bbox[1]), (bbox[2], bbox[3]), c, 4)
    if show_txt:
      cv2.rectangle(self.imgs[img_id],
                    (bbox[0], bbox[1] - cat_size[1] - 2),
                    (bbox[0] + cat_size[0], bbox[1] - 2), c, -1)
      cv2.putText(self.imgs[img_id], txt, (bbox[0], bbox[1] - 2), 
                  font, 0.5, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

  def add_coco_hp(self, points, img_id='default'): 
    points = np.array(points, dtype=np.int32).reshape(self.num_joints, 2)
    for j in range(self.num_joints):
      cv2.circle(self.imgs[img_id],
                 (points[j, 0], points[j, 1]), 3, self.colors_hp[j], -1)
    for j, e in enumerate(self.edges):
      if points[e].min() > 0:
        cv2.line(self.imgs[img_id], (points[e[0], 0], points[e[0], 1]),
                      (points[e[1], 0], points[e[1], 1]), self.ec[j], 2,
                      lineType=cv2.LINE_AA)

  def add_points(self, points, img_id='default'):
    num_classes = len(points)
    # assert num_classes == len(self.colors)
    for i in range(num_classes):
      for j in range(len(points[i])):
        c = self.colors[i, 0, 0]
        cv2.circle(self.imgs[img_id], (points[i][j][0] * self.down_ratio, 
                                       points[i][j][1] * self.down_ratio),
                   5, (255, 255, 255), -1)
        cv2.circle(self.imgs[img_id], (points[i][j][0] * self.down_ratio,
                                       points[i][j][1] * self.down_ratio),
                   3, (int(c[0]), int(c[1]), int(c[2])), -1)

  def show_all_imgs(self, pause=False, time=0):
    if not self.ipynb:
      write = True      
      out = cv2.VideoWriter('test1.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 480))
      for i, v in self.imgs.items():     
        cv2.waitKey(10)
        v2= cv2.resize(v, (1920, 1080) )
        cv2.imshow('{}'.format(i), v2)        
        v1 = v
        #self.videoWriter.write(v)
        #cv2.waitKey(10)        
      ###############################################################
        if (write):
          out.write(v1)
      ###################3
      #self.videoWriter.release()

      if cv2.waitKey(0 if pause else 1) == 27:
        import sys
        sys.exit(0)
    else:
      self.ax = None
      nImgs = len(self.imgs)
      fig=self.plt.figure(figsize=(nImgs * 10,10))
      nCols = nImgs
      nRows = nImgs // nCols
      for i, (k, v) in enumerate(self.imgs.items()):
        fig.add_subplot(1, nImgs, i + 1)
        if len(v.shape) == 3:
          self.plt.imshow(cv2.cvtColor(v, cv2.COLOR_BGR2RGB))
        else:
          self.plt.imshow(v)
      self.plt.show()

  def save_all_imgs_video(self, pause=False):
    out = cv2.VideoWriter('test1.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 480))
    for i, v in self.imgs.items():        
      v1 = v
      out.write(v1)
    
  def save_img(self, imgId='default', path='./cache/debug/'):
    cv2.imwrite(path + '{}.png'.format(imgId), self.imgs[imgId])
    
  def save_all_imgs(self, path='./cache/debug/', prefix='', genID=False):
    if genID:
      try:
        idx = int(np.loadtxt(path + '/id.txt'))
      except:
        idx = 0
      prefix=idx
      np.savetxt(path + '/id.txt', np.ones(1) * (idx + 1), fmt='%d')
    for i, v in self.imgs.items():
      cv2.imwrite(path + '/{}{}.png'.format(prefix, i), v)

  def remove_side(self, img_id, img):
    if not (img_id in self.imgs):
      return
    ws = img.sum(axis=2).sum(axis=0)
    l = 0
    while ws[l] == 0 and l < len(ws):
      l+= 1
    r = ws.shape[0] - 1
    while ws[r] == 0 and r > 0:
      r -= 1
    hs = img.sum(axis=2).sum(axis=1)
    t = 0
    while hs[t] == 0 and t < len(hs):
      t += 1
    b = hs.shape[0] - 1
    while hs[b] == 0 and b > 0:
      b -= 1
    self.imgs[img_id] = self.imgs[img_id][t:b+1, l:r+1].copy()

  def project_3d_to_bird(self, pt):
    pt[0] += self.world_size / 2
    pt[1] = self.world_size - pt[1]
    pt = pt * self.out_size / self.world_size
    return pt.astype(np.int32)

  def add_ct_detection(
    self, img, dets, show_box=False, show_txt=True, 
    center_thresh=0.5, img_id='det'):
    # dets: max_preds x 5
    self.imgs[img_id] = img.copy()
    if type(dets) == type({}):
      for cat in dets:
        for i in range(len(dets[cat])):
          if dets[cat][i, 2] > center_thresh:
            cl = (self.colors[cat, 0, 0]).tolist()
            ct = dets[cat][i, :2].astype(np.int32)
            if show_box:
              w, h = dets[cat][i, -2], dets[cat][i, -1]
              x, y = dets[cat][i, 0], dets[cat][i, 1]
              bbox = np.array([x - w / 2, y - h / 2, x + w / 2, y + h / 2],
                              dtype=np.float32)
              self.add_coco_bbox(
                bbox, cat - 1, dets[cat][i, 2], 
                show_txt=show_txt, img_id=img_id)
    else:
      for i in range(len(dets)):
        if dets[i, 2] > center_thresh:
          # print('dets', dets[i])
          cat = int(dets[i, -1])
          cl = (self.colors[cat, 0, 0] if self.theme == 'black' else \
                                       255 - self.colors[cat, 0, 0]).tolist()
          ct = dets[i, :2].astype(np.int32) * self.down_ratio
          cv2.circle(self.imgs[img_id], (ct[0], ct[1]), 3, cl, -1)
          if show_box:
            w, h = dets[i, -3] * self.down_ratio, dets[i, -2] * self.down_ratio
            x, y = dets[i, 0] * self.down_ratio, dets[i, 1] * self.down_ratio
            bbox = np.array([x - w / 2, y - h / 2, x + w / 2, y + h / 2],
                            dtype=np.float32)
            self.add_coco_bbox(bbox, dets[i, -1], dets[i, 2], img_id=img_id)


  def add_3d_detection(
    self, image_or_path, dets, calib, show_txt=False, 
    center_thresh=0.5, img_id='det'):
    if isinstance(image_or_path, np.ndarray):
      self.imgs[img_id] = image_or_path
    else: 
      self.imgs[img_id] = cv2.imread(image_or_path)
    for cat in dets:
      for i in range(len(dets[cat])):
        cl = (self.colors[cat - 1, 0, 0]).tolist()
        if dets[cat][i, -1] > center_thresh:
          dim = dets[cat][i, 5:8]
          loc  = dets[cat][i, 8:11]
          rot_y = dets[cat][i, 11]
          # loc[1] = loc[1] - dim[0] / 2 + dim[0] / 2 / self.dim_scale
          # dim = dim / self.dim_scale
          if loc[2] > 1:
            box_3d = compute_box_3d(dim, loc, rot_y)
            box_2d = project_to_image(box_3d, calib)
            self.imgs[img_id] = draw_box_3d(self.imgs[img_id], box_2d, cl)

  def compose_vis_add(
    self, img_path, dets, calib,
    center_thresh, pred, bev, img_id='out'):
    self.imgs[img_id] = cv2.imread(img_path)
    # h, w = self.imgs[img_id].shape[:2]
    # pred = cv2.resize(pred, (h, w))
    h, w = pred.shape[:2]
    hs, ws = self.imgs[img_id].shape[0] / h, self.imgs[img_id].shape[1] / w
    self.imgs[img_id] = cv2.resize(self.imgs[img_id], (w, h))
    self.add_blend_img(self.imgs[img_id], pred, img_id)
    for cat in dets:
      for i in range(len(dets[cat])):
        cl = (self.colors[cat - 1, 0, 0]).tolist()
        if dets[cat][i, -1] > center_thresh:
          dim = dets[cat][i, 5:8]
          loc  = dets[cat][i, 8:11]
          rot_y = dets[cat][i, 11]
          # loc[1] = loc[1] - dim[0] / 2 + dim[0] / 2 / self.dim_scale
          # dim = dim / self.dim_scale
          if loc[2] > 1:
            box_3d = compute_box_3d(dim, loc, rot_y)
            box_2d = project_to_image(box_3d, calib)
            box_2d[:, 0] /= hs
            box_2d[:, 1] /= ws
            self.imgs[img_id] = draw_box_3d(self.imgs[img_id], box_2d, cl)
    self.imgs[img_id] = np.concatenate(
      [self.imgs[img_id], self.imgs[bev]], axis=1)

  def add_2d_detection(
    self, img, dets, show_box=False, show_txt=True, 
    center_thresh=0.5, img_id='det'):
    self.imgs[img_id] = img
    for cat in dets:
      for i in range(len(dets[cat])):
        cl = (self.colors[cat - 1, 0, 0]).tolist()
        if dets[cat][i, -1] > center_thresh:
          bbox = dets[cat][i, 1:5]
          self.add_coco_bbox(
            bbox, cat - 1, dets[cat][i, -1], 
            show_txt=show_txt, img_id=img_id)

  def add_bird_view(self, dets, center_thresh=0.3, img_id='bird'):
    bird_view = np.ones((self.out_size, self.out_size, 3), dtype=np.uint8) * 230
    for cat in dets:
      cl = (self.colors[cat - 1, 0, 0]).tolist()
      lc = (250, 152, 12)
      for i in range(len(dets[cat])):
        if dets[cat][i, -1] > center_thresh:
          dim = dets[cat][i, 5:8]
          loc  = dets[cat][i, 8:11]
          rot_y = dets[cat][i, 11]
          rect = compute_box_3d(dim, loc, rot_y)[:4, [0, 2]]
          for k in range(4):
            rect[k] = self.project_3d_to_bird(rect[k])
            # cv2.circle(bird_view, (rect[k][0], rect[k][1]), 2, lc, -1)
          cv2.polylines(
              bird_view,[rect.reshape(-1, 1, 2).astype(np.int32)],
              True,lc,2,lineType=cv2.LINE_AA)
          for e in [[0, 1]]:
            t = 4 if e == [0, 1] else 1
            cv2.line(bird_view, (rect[e[0]][0], rect[e[0]][1]),
                    (rect[e[1]][0], rect[e[1]][1]), lc, t,
                    lineType=cv2.LINE_AA)
    self.imgs[img_id] = bird_view

  def add_bird_views(self, dets_dt, dets_gt, center_thresh=0.3, img_id='bird'):
    alpha = 0.5
    bird_view = np.ones((self.out_size, self.out_size, 3), dtype=np.uint8) * 230
    for ii, (dets, lc, cc) in enumerate(
      [(dets_gt, (12, 49, 250), (0, 0, 255)), 
       (dets_dt, (250, 152, 12), (255, 0, 0))]):
      # cc = np.array(lc, dtype=np.uint8).reshape(1, 1, 3)
      for cat in dets:
        cl = (self.colors[cat - 1, 0, 0]).tolist()
        for i in range(len(dets[cat])):
          if dets[cat][i, -1] > center_thresh:
            dim = dets[cat][i, 5:8]
            loc  = dets[cat][i, 8:11]
            rot_y = dets[cat][i, 11]
            rect = compute_box_3d(dim, loc, rot_y)[:4, [0, 2]]
            for k in range(4):
              rect[k] = self.project_3d_to_bird(rect[k])
            if ii == 0:
              cv2.fillPoly(
                bird_view,[rect.reshape(-1, 1, 2).astype(np.int32)],
                lc,lineType=cv2.LINE_AA)
            else:
              cv2.polylines(
                bird_view,[rect.reshape(-1, 1, 2).astype(np.int32)],
                True,lc,2,lineType=cv2.LINE_AA)
            # for e in [[0, 1], [1, 2], [2, 3], [3, 0]]:
            for e in [[0, 1]]:
              t = 4 if e == [0, 1] else 1
              cv2.line(bird_view, (rect[e[0]][0], rect[e[0]][1]),
                      (rect[e[1]][0], rect[e[1]][1]), lc, t,
                      lineType=cv2.LINE_AA)
    self.imgs[img_id] = bird_view

mixall_class_name = [
  '1_Body_Adult', '2_Body_Child1-6', '3_Body_Child0-1_Normal', '4_Body_Child0-1_Lie', '5_Face_Adult', '6_Face_Child1-6', '7_Face_Child0-1_Normal', '8_Face_Child0-1_Covered'
]

# food_class_name = [
#   '1_rice', '2_eels on rice', '3_pilaf', "4_chicken-n-egg on rice",
#                        '5_pork cutlet on rice', '6_beef curry', '7_sushi', '8_chicken rice', 
#                        '9_fried rice', '10_tempura bowl', '11_bibimbap', '12_toast',
#                        '13_croissant', '14_roll bread', '15_raisin bread', '16_chip butty',
#                        '17_hamburger', '18_pizza', '19_sandwiches', '20_udon noodle',
#                        '21_tempura udon', '22_soba noodle', '23_ramen noodle', '24_beef noodle',
#                        '25_tensin noodle', '26_fried noodle', '27_spaghetti', '28_Japanese-style pancake',
#                        '29_takoyaki', '30_gratin', '31_sauteed vegetables', '32_croquette',
#                        '33_grilled eggplant', '34_sauteed spinach', '35_vegetable tempura', '36_miso soup',
#                        '37_potage', '38_sausage', '39_oden', '40_omelet',
#                        '41_ganmodoki', '42_jiaozi', '43_stew', '44_teriyaki grilled fish',
#                        '45_fried fish', '46_grilled salmon', '47_salmon meuniere ', '48_sashimi',
#                        '49_grilled pacific saury ', '50_sukiyaki', '51_sweet and sour pork', '52_lightly roasted fish',
#                        '53_steamed egg hotchpotch', '54_tempura', '55_fried chicken', '56_sirloin cutlet ',
#                        '57_nanbanzuke', '58_boiled fish', '59_seasoned beef with potatoes', '60_hambarg steak',
#                        '61_beef steak', '62_dried fish', '63_ginger pork saute', '64_spicy chili-flavored tofu',
#                        '65_yakitori', '66_cabbage roll', '67_rolled omelet', '68_egg sunny-side up',
#                        '69_fermented soybeans', '70_cold tofu', '71_egg roll', '72_chilled noodle',
#                        '73_stir-fried beef and peppers', '74_simmered pork', '75_boiled chicken and vegetables', '76_sashimi bowl', 
#                        '77_sushi bowl', '78_fish-shaped pancake with bean jam', '79_shrimp with chill source', '80_roast chicken',
#                        '81_steamed meat dumpling', '82_omelet with fried rice', '83_cutlet curry', '84_spaghetti meat sauce',
#                        '85_fried shrimp', '86_potato salad', '87_green salad', '88_macaroni salad',
#                        '89_Japanese tofu and vegetable chowder', '90_pork miso soup', '91_chinese soup', '92_beef bowl',
#                        '93_kinpira-style sauteed burdock', '94_rice ball', '95_pizza toast', '96_dipping noodles',
#                        '97_hot dog', '98_french fries', '99_mixed rice', '100_goya chanpuru'
# ]

food_class_name = ['Rice: 205 cal [cup (158g)]',
 'Nutritionix - Page not found: None cal [None]',
 'Nutritionix - Page not found: None cal [None]',
 'Nutritionix - Page not found: None cal [None]',
 'Nutritionix - Page not found: None cal [None]',
 'Beef Curry: 266 cal [cup (235g)]',
 'Sushi: 44 cal [piece (26g)]',
 'Chicken Rice: 317 cal [cup (233g)]',
 'Fried Rice: 238 cal [cup (137g)]',
 'Tempura Bowl: 477 cal [bowl (454g)]',
 'Bibimbap: 972 cal [bowl (864g)]',
 'Toast: 64 cal [slice (22g)]',
 'Croissant: 272 cal [croissant, large (67g)]',
 'Nutritionix - Page not found: None cal [None]',
 'Raisin Bread: 71 cal [slice (26g)]',
 'Chip Butty: 402 cal [sandwich (99g)]',
 'Hamburger: 540 cal [sandwich (226g)]',
 'Pizza: 285 cal [slice (107g)]',
 'Sandwiches: 155 cal [slices of bread (58g)]',
 'Nutritionix - Page not found: None cal [None]',
 'Tempura Udon: 681 cal [bowl (about 3 cups) (745g)]',
 'Soba Noodle: 113 cal [cup (114g)]',
 'Ramen Noodle: 384 cal [package (566g)]',
 'Beef Noodle: 662 cal [cups (479g)]',
 'Tensin Noodle: 196 cal [cup spaghetti not packed (124g)]',
 'Fried Noodle: 196 cal [cup spaghetti not packed (124g)]',
 'Spaghetti: 210 cal [cup (140g)]',
 'Japanese Style Pancake: 91 cal [pancake (5") (40g)]',
 'Takoyaki: 58 cal [piece (39g)]',
 'Nutritionix - Page not found: None cal [None]',
 'Sauteed Vegetables: 59 cal [cup (91g)]',
 'Nutritionix - Page not found: None cal [None]',
 'Grilled Eggplant: 42 cal [slice (55g)]',
 'Sauteed Spinach: 89 cal [cup (186g)]',
 'Vegetable Tempura: 53 cal [piece (40g)]',
 'Miso Soup: 59 cal [cup (241g)]',
 'Nutritionix - Page not found: None cal [None]',
 'Sausage: 210 cal [link (4" long x 1-1/8" dia) (68g)]',
 'Oden: 275 cal [cup (231g)]',
 'Omelet: 323 cal [3-egg omelet (178g)]',
 'Ganmodoki: 60 cal [piece (36g)]',
 'Jiaozi: 67 cal [piece (37g)]',
 'Stew: 535 cal [bowl (504g)]',
 'Nutritionix - Page not found: None cal [None]',
 'Fried Fish: 199 cal [fillet (87g)]',
 'Grilled Salmon: 468 cal [fillet (227g)]',
 'Salmon Meuniere: 468 cal [fillet (227g)]',
 'Sashimi: 35 cal [piece (28g)]',
 'Nutritionix - Page not found: None cal [None]',
 'Sukiyaki: 745 cal [serving (704g)]',
 'Sweet And Sour Pork: 1644 cal [order (609g)]',
 'Lightly Roasted Fish: 218 cal [medium fillet (6 oz) (170g)]',
 'Steamed Egg Hotchpotch: 72 cal [large (50g)]',
 'Tempura: 438 cal [serving (416g)]',
 'Fried Chicken: 377 cal [piece (140g)]',
 'Nutritionix - Page not found: None cal [None]',
 'Nutritionix - Page not found: None cal [None]',
 'Boiled Fish: 218 cal [medium fillet (6 oz) (170g)]',
 'Nutritionix - Page not found: None cal [None]',
 'Hambarg Steak: 614 cal [steak (221g)]',
 'Beef Steak: 614 cal [steak (221g)]',
 'Dried Fish: 82 cal [oz (28g)]',
 'Ginger Pork Saute: 459 cal [cup (236g)]',
 'Nutritionix - Page not found: None cal [None]',
 'Yakitori: 140 cal [Skewer (100g)]',
 'Cabbage Roll: 229 cal [roll (259g)]',
 'Rolled Omelet: 323 cal [3-egg omelet (178g)]',
 'Egg Sunny Side Up: 90 cal [large (46g)]',
 'Fermented Soybeans: 296 cal [cup (172g)]',
 'Cold Tofu: 76 cal [block (91g)]',
 'Egg Roll: 223 cal [piece (89g)]',
 'Chilled Noodle: 196 cal [cup spaghetti not packed (124g)]',
 'Nutritionix - Page not found: None cal [None]',
 'Simmered Pork: 202 cal [oz (85g)]',
 'Nutritionix - Page not found: None cal [None]',
 'Sashimi Bowl: 561 cal [bowl (454g)]',
 'Sushi Bowl: 749 cal [bowl (454g)]',
 'Nutritionix - Page not found: None cal [None]',
 'Shrimp With Chill Source: 6 cal [shrimp (5g)]',
 'Roast Chicken: 190 cal [oz (85g)]',
 'Nutritionix - Page not found: None cal [None]',
 'Nutritionix - Page not found: None cal [None]',
 'Nutritionix - Page not found: None cal [None]',
 'Spaghetti Meat Sauce: 667 cal [serving (about 2 cups) (660g)]',
 'Fried Shrimp: 38 cal [piece (12g)]',
 'Potato Salad: 358 cal [cup (250g)]',
 'Green Salad: 20 cal [serving (85g)]',
 'Macaroni Salad: 207 cal [cup (102g)]',
 'Nutritionix - Page not found: None cal [None]',
 'Nutritionix - Page not found: None cal [None]',
 'Chinese Soup: 62 cal [serving 1 cup (248g)]',
 'Beef Bowl: 1319 cal [bowl (453g)]',
 'Nutritionix - Page not found: None cal [None]',
 'Rice Ball: 47 cal [piece (21g)]',
 'Pizza Toast: 232 cal [slice (86g)]',
 'Dipping Noodles: 206 cal [cup (239g)]',
 'Hot Dog: 155 cal [frankfurter (48g)]',
 'French Fries: 365 cal [serving medium (117g)]',
 'Mixed Rice: 205 cal [cup (158g)]',
 'Goya Chanpuru: 258 cal [cup (214g)]']


color_list = np.array(
        [
            1.000, 1.000, 1.000,
            1,0,0, 
            0,1,0, 
            0,0,1, 
            1,1,0, 
            0,1,1, 
            1,0,1, 
            0,0,0,
            0.850, 0.325, 0.098,
            0.929, 0.694, 0.125,
            0.494, 0.184, 0.556,
            0.466, 0.674, 0.188,
            0.301, 0.745, 0.933,
            0.635, 0.078, 0.184,
            0.300, 0.300, 0.300,
            0.600, 0.600, 0.600,
            1.000, 0.000, 0.000,
            1.000, 0.500, 0.000,
            0.749, 0.749, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 1.000,
            0.667, 0.000, 1.000,
            0.333, 0.333, 0.000,
            0.333, 0.667, 0.000,
            0.333, 1.000, 0.000,
            0.667, 0.333, 0.000,
            0.667, 0.667, 0.000,
            0.667, 1.000, 0.000,
            1.000, 0.333, 0.000,
            1.000, 0.667, 0.000,
            1.000, 1.000, 0.000,
            0.000, 0.333, 0.500,
            0.000, 0.667, 0.500,
            0.000, 1.000, 0.500,
            0.333, 0.000, 0.500,
            0.333, 0.333, 0.500,
            0.333, 0.667, 0.500,
            0.333, 1.000, 0.500,
            0.667, 0.000, 0.500,
            0.667, 0.333, 0.500,
            0.667, 0.667, 0.500,
            0.667, 1.000, 0.500,
            1.000, 0.000, 0.500,
            1.000, 0.333, 0.500,
            1.000, 0.667, 0.500,
            1.000, 1.000, 0.500,
            0.000, 0.333, 1.000,
            0.000, 0.667, 1.000,
            0.000, 1.000, 1.000,
            0.333, 0.000, 1.000,
            0.333, 0.333, 1.000,
            0.333, 0.667, 1.000,
            0.333, 1.000, 1.000,
            0.667, 0.000, 1.000,
            0.667, 0.333, 1.000,
            0.667, 0.667, 1.000,
            0.667, 1.000, 1.000,
            1.000, 0.000, 1.000,
            1.000, 0.333, 1.000,
            1.000, 0.667, 1.000,
            0.167, 0.000, 0.000,
            0.333, 0.000, 0.000,
            0.500, 0.000, 0.000,
            0.667, 0.000, 0.000,
            0.833, 0.000, 0.000,
            1.000, 0.000, 0.000,
            0.000, 0.167, 0.000,
            0.000, 0.333, 0.000,
            0.000, 0.500, 0.000,
            0.000, 0.667, 0.000,
            0.000, 0.833, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 0.167,
            0.000, 0.000, 0.333,
            0.000, 0.000, 0.500,
            0.000, 0.000, 0.667,
            0.000, 0.000, 0.833,
            0.000, 0.000, 1.000,
            0.000, 0.000, 0.000,
            0.143, 0.143, 0.143,
            0.286, 0.286, 0.286,
            0.429, 0.429, 0.429,
            0.571, 0.571, 0.571,
            0.714, 0.714, 0.714,
            0.857, 0.857, 0.857,
            0.000, 0.447, 0.741,
            0.50, 0.5, 0,
            0.000, 0.000, 0.500,
            0.000, 0.000, 0.667,
            0.000, 0.000, 0.833,
            0.000, 0.000, 1.000,
            0.000, 0.000, 0.000,
            0.143, 0.143, 0.143,
            0.286, 0.286, 0.286,
            0.429, 0.429, 0.429,
            0.571, 0.571, 0.571,
            0.714, 0.714, 0.714,
            0.857, 0.857, 0.857,
            0.000, 0.447, 0.741,
            0.50, 0.5, 0
        ]
    ).astype(np.float32)
color_list = color_list.reshape((-1, 3)) * 255
