import sys
import os
import scipy.io as sio
import cv2
import json
path = os.path.dirname(__file__)
CENTERNET_PATH = os.path.join(path,'../src/lib')
sys.path.insert(0, CENTERNET_PATH)
from detectors.detector_factory import detector_factory
from opts import opts
from datasets.dataset_factory import get_dataset

import random
import numpy as np


calfb_class_name = [
  '1_rice', '2_eels on rice', '3_pilaf', "4_chicken-n-egg on rice",
                       '5_pork cutlet on rice', '6_beef curry', '7_sushi', '8_chicken rice', 
                       '9_fried rice', '10_tempura bowl', '11_bibimbap', '12_toast',
                       '13_croissant', '14_roll bread', '15_raisin bread', '16_chip butty',
                       '17_hamburger', '18_pizza', '19_sandwiches', '20_udon noodle',
                       '21_tempura udon', '22_soba noodle', '23_ramen noodle', '24_beef noodle',
                       '25_tensin noodle', '26_fried noodle', '27_spaghetti', '28_Japanese-style pancake',
                       '29_takoyaki', '30_gratin', '31_sauteed vegetables', '32_croquette',
                       '33_grilled eggplant', '34_sauteed spinach', '35_vegetable tempura', '36_miso soup',
                       '37_potage', '38_sausage', '39_oden', '40_omelet',
                       '41_ganmodoki', '42_jiaozi', '43_stew', '44_teriyaki grilled fish',
                       '45_fried fish', '46_grilled salmon', '47_salmon meuniere ', '48_sashimi',
                       '49_grilled pacific saury ', '50_sukiyaki', '51_sweet and sour pork', '52_lightly roasted fish',
                       '53_steamed egg hotchpotch', '54_tempura', '55_fried chicken', '56_sirloin cutlet ',
                       '57_nanbanzuke', '58_boiled fish', '59_seasoned beef with potatoes', '60_hambarg steak',
                       '61_beef steak', '62_dried fish', '63_ginger pork saute', '64_spicy chili-flavored tofu',
                       '65_yakitori', '66_cabbage roll', '67_rolled omelet', '68_egg sunny-side up',
                       '69_fermented soybeans', '70_cold tofu', '71_egg roll', '72_chilled noodle',
                       '73_stir-fried beef and peppers', '74_simmered pork', '75_boiled chicken and vegetables', '76_sashimi bowl', 
                       '77_sushi bowl', '78_fish-shaped pancake with bean jam', '79_shrimp with chill source', '80_roast chicken',
                       '81_steamed meat dumpling', '82_omelet with fried rice', '83_cutlet curry', '84_spaghetti meat sauce',
                       '85_fried shrimp', '86_potato salad', '87_green salad', '88_macaroni salad',
                       '89_Japanese tofu and vegetable chowder', '90_pork miso soup', '91_chinese soup', '92_beef bowl']

calfb_class_name = ['Rice: 205 cal [cup (158g)]',
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

calfb_class_name = [i for i in range(1, 101)]

def get_bbox(bbox, cat, conf=1): 
    bbox = np.array(bbox, dtype=np.int32)
    # cat = (int(cat) + 1) % 80
    cat = int(cat)
    txt = '{} {:.2f}'.format(calfb_class_name[cat], conf)
    print(txt)
    return bbox[0], bbox[1], bbox[2], bbox[3], calfb_class_name[cat]

# from torchsummary import summary
def test_img(MODEL_PATH):
    debug = -1            # draw and show the result image
    thresh = 0.3
    TASK = 'ctdet'
    dataset = 'food100' # ['calface', 'calbody', 'callall', 'mixall']
    arch = 'mobilenetv2fpn_10' # ''
    input_h, intput_w = 512, 512
    opt = opts().init('--task {} --load_model {} --debug {} --input_h {} --input_w {} --dataset {} --vis_thresh {} --arch {} --nms'.format(
        TASK, MODEL_PATH, debug, intput_w, input_h, dataset, thresh, arch).split(' '), target=dataset)
    detector = detector_factory[opt.task](opt)
    print(opt)

    path = os.getcwd()
    img_folder = os.path.join(path, '../data/food_100/image')
    test_folder = os.path.join(path, '../data/food_100/annotations/test_food.json')
    
    with open(test_folder, 'r') as file:
        json_dict = json.load(open(test_folder, 'r'))
    images = json_dict['images']
    annots = json_dict['annotations']
    test_images = [image['file_name'] for image in images]
    random.shuffle(test_images)
    
    # files = listdir(mypath)
    idx = 0
    file_names = os.listdir(img_folder)
    random.shuffle(file_names)

    save_folder = os.path.join(path, 'demo_images/food100_fpn')
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    print('total numbers of images to detect: ', len(test_images))
    for file_name in test_images:
        img = os.path.join(img_folder, file_name)
        print('idx: ', idx, img)
        res = detector.run(img)
        
        for j in range(1, detector.num_classes + 1):
            for bbox in res['results'][j]:
                if bbox[4] > thresh:## show threshold
                    x_min, y_min, x_max, y_max, cat_name = get_bbox(bbox[:4], j - 1, bbox[4])

                    print("informartion  ", x_min, y_min, x_max, y_max, cat_name)
        # cv2.imshow('face detect', res['plot_img'])
        cv2.imwrite(save_folder+'/result_img_{}.jpg'.format(idx), res['plot_img']) 
        idx = idx + 1
        if idx == 100:
            break
        

if __name__ == '__main__':
    # if GOAL is face:
    # MODEL_PATH = '../exp/ctdet/calface_mobile_l1/model_last.pth'
    # MODEL_PATH = '../exp/ctdet/calbody_mobile_l1/model_last.pth'

    # if GOAL is all:
    # MODEL_PATH = '../exp/ctdet/calall_mobile_l1/model_120.pth'
    # MODEL_PATH = '../exp/ctdet/mixall_lossWA_ptBody_1105/model_120.pth'

    # if GOAL is all & is FPN:
    # MODEL_PATH = '../exp/ctdet/mixall_fpn1234_widerface/model_last.pth'
    MODEL_PATH = '../exp/ctdet/food100_fpn/model_last.pth'

    test_img(MODEL_PATH)
    # test_vedio(MODEL_PATH)
    # test_wider_Face(MODEL_PATH)
