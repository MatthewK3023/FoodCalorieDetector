import xml.etree.ElementTree as ET
import os
import json

coco = dict()
coco['images'] = []
coco['type'] = 'instances'
coco['annotations'] = []
coco['categories'] = []

category_set = dict()
image_set = set()

category_item_id = 0
image_id = 20180000000
annotation_id = 0

# cat_mapping_list = ['5_Face_Adult', '6_Face_Child1-6', '7_Face_Child0-1_Normal', '8_Face_Child0-1_Covered']
cat_mapping_list = ['1_Body_Adult', '2_Body_Child1-6', '3_Body_Child0-1_Normal', '4_Body_Child0-1_Lie', '5_Face_Adult', '6_Face_Child1-6', '7_Face_Child0-1_Normal', '8_Face_Child0-1_Covered']
# cat_mapping_list = ['Body', 'Face']
# cat_mapping_list = ['Body', 'Face', 'Face_Cover']

def addCatItem(name):
    global category_item_id
    category_item = dict()
    category_item['supercategory'] = 'none'
    category_item_id += 1
    category_item['id'] = category_item_id
    category_item['name'] = name
    coco['categories'].append(category_item)
    category_set[name] = category_item_id
    return category_item_id

def addImgItem(file_name, size):
    global image_id
    if file_name is None:
        raise Exception('Could not find filename tag in xml file.')
    if size['width'] is None:
        raise Exception('Could not find width tag in xml file.')
    if size['height'] is None:
        raise Exception('Could not find height tag in xml file.')
    image_id += 1
    image_item = dict()
    image_item['id'] = image_id
    image_item['file_name'] = file_name
    image_item['width'] = size['width']
    image_item['height'] = size['height']
    coco['images'].append(image_item)
    image_set.add(file_name)
    return image_id

def addAnnoItem(object_name, image_id, category_id, bbox):
    global annotation_id
    annotation_item = dict()
    annotation_item['segmentation'] = []
    seg = []
    #bbox[] is x,y,w,h
    #left_top
    seg.append(bbox[0])
    seg.append(bbox[1])
    #left_bottom
    seg.append(bbox[0])
    seg.append(bbox[1] + bbox[3])
    #right_bottom
    seg.append(bbox[0] + bbox[2])
    seg.append(bbox[1] + bbox[3])
    #right_top
    seg.append(bbox[0] + bbox[2])
    seg.append(bbox[1])

    annotation_item['segmentation'].append(seg)

    annotation_item['area'] = bbox[2] * bbox[3]
    annotation_item['iscrowd'] = 0
    annotation_item['ignore'] = 0
    annotation_item['image_id'] = image_id
    annotation_item['bbox'] = bbox
    annotation_item['category_id'] = category_id
    annotation_id += 1
    annotation_item['id'] = annotation_id
    coco['annotations'].append(annotation_item)


def parseXmlFiles(fs):

    for cat_map in cat_mapping_list:
        addCatItem(cat_map)
    for f in fs:
        if not f.endswith('.xml'):
            continue
        bndbox = dict()
        size = dict()
        current_image_id = None
        current_category_id = None
        file_name = None
        size['width'] = None
        size['height'] = None
        size['depth'] = None

        xml_file = os.path.join(xml_path, f)
        print(xml_file)

        tree = ET.parse(xml_file)
        root = tree.getroot()
        if root.tag != 'annotation':
            raise Exception('pascal voc xml root element should be annotation, rather than {}'.format(root.tag))


        #elem is <folder>, <filename>, <size>, <object>
        for elem in root:
            current_parent = elem.tag
            current_sub = None
            object_name = None
            
            if elem.tag == 'folder':
                continue
            
            if elem.tag == 'filename':
                file_name = elem.text
                if file_name in category_set:
                    raise Exception('file_name duplicated')
                
            #add img item only after parse <size> tag
            elif current_image_id is None and file_name is not None and size['width'] is not None:
                if file_name not in image_set:
                    current_image_id = addImgItem(file_name, size)
                    print('add image with {} and {}'.format(file_name, size))
                else:
                    raise Exception('duplicated image: {}'.format(file_name)) 
            #subelem is <width>, <height>, <depth>, <name>, <bndbox>

            for subelem in elem:
                bndbox ['xmin'] = None
                bndbox ['xmax'] = None
                bndbox ['ymin'] = None
                bndbox ['ymax'] = None
                
                current_sub = subelem.tag
                if current_parent == 'object' and subelem.tag == 'name':
                    object_name = subelem.text
                    #print('object_name:', object_name)
                    # ##### cal_face
                    # if 'Body' in object_name:
                    #     break
                    #### cal_fb
                    # if '1_Body_Adult' in object_name or '5_Face_Adult' in object_name:
                    #     break
                    # if 'Body' in object_name:
                    #     object_name = 'Body'
                    # if 'Face' in object_name:
                    #     object_name = 'Face' 

                    #### cal_fbc
                    # if '1_Body_Adult' in object_name or '5_Face_Adult' in object_name:
                    #     break
                    # if 'Body' in object_name:
                    #     object_name = 'Body'
                    # if 'Face' in object_name and 'Cover' not in object_name:
                    #     object_name = 'Face'
                    # if 'Face' in object_name and 'Cover' in object_name:
                    #     object_name = 'Face_Cover'

                    ### kick the pet
                    if ('9' in object_name) or ('10' in object_name) or \
                        ('11' in object_name) or ('12' in object_name):
                        break

                    if object_name not in category_set:
                        current_category_id = addCatItem(object_name)
                    else:
                        current_category_id = category_set[object_name]

                elif current_parent == 'size':
                    if size[subelem.tag] is not None:
                        raise Exception('xml structure broken at size tag.')
                    size[subelem.tag] = int(subelem.text)

                #option is <xmin>, <ymin>, <xmax>, <ymax>, when subelem is <bndbox>
                for option in subelem:
                    if current_sub == 'bndbox':
                        if bndbox[option.tag] is not None:
                            raise Exception('xml structure corrupted at bndbox tag.')
                        bndbox[option.tag] = int(option.text)

                #only after parse the <object> tag
                if bndbox['xmin'] is not None:
                    # print('object_name ', object_name)
                    # print('current_image_id ', current_image_id)
                    # print('current_category_id ', current_category_id)

                    if object_name is None:
                        raise Exception('xml structure broken at bndbox tag [object_name is None]')
                    if current_image_id is None:
                        raise Exception('xml structure broken at bndbox tag [current_image_id is None]')
                    if current_category_id is None:
                        raise Exception('xml structure broken at bndbox tag [current_category_id is None]')
                    bbox = []
                    #x
                    bbox.append(bndbox['xmin'])
                    #y
                    bbox.append(bndbox['ymin'])
                    #w
                    bbox.append(bndbox['xmax'] - bndbox['xmin'])
                    #h
                    bbox.append(bndbox['ymax'] - bndbox['ymin'])
                    print('add annotation with {},{},{},{}'.format(object_name, current_image_id, current_category_id, bbox))
                    addAnnoItem(object_name, current_image_id, current_category_id, bbox )
    # print("iiiiii ",  ii) 
    print(coco['categories'])

def reset_coco():
    global coco
    coco = dict()
    coco['images'] = []
    coco['type'] = 'instances'
    coco['annotations'] = []
    coco['categories'] = []
    global category_set
    category_set = dict()
    global category_item_id
    category_item_id=0

def seperate_data(xml_path, train_json, val_json, test_json, train_split_rate=0.7, val_split_rate=0.8):
    
    # train_nums = int(train_split_rate * len(os.listdir(xml_path)))
    # val_nums = int(val_split_rate * len(os.listdir(xml_path)))
    # # nums = 0
    total_files = os.listdir(xml_path)
    # # ii  = 0
    # train_files = total_files[:train_nums]
    # val_files = total_files[train_nums:val_nums]
    # test_files = total_files[val_nums:]

    test_files = total_files

    # parseXmlFiles(train_files)
    # json.dump(coco, open(train_json, 'w'))
    # reset_coco()
    # # print(coco)
    # parseXmlFiles(val_files)
    # json.dump(coco, open(val_json, 'w'))
    # reset_coco()
    # #print(coco)
    parseXmlFiles(test_files)
    json.dump(coco, open(test_json, 'w'))

    # print(len(train_files))
    # print(len(val_files))
    print('length:', len(test_files))

    


if __name__ == '__main__':
    # xml_path = '/home/yangna/deepblue/32_face_detect/centerface/data/wider_face/voc_xml'
    # json_file = './data/wider_face/annotations/val_wider_face.json'

    # xml_path = '/home/jamesyen/my_project/face_detection/CenterNet/data/cal_face/Annotations/'
    
    # train_json_file = '/home/jamesyen/my_project/face_detection/CenterNet/data/cal_fbc/annotations/train_cal_face.json'
    # val_json_file = '/home/jamesyen/my_project/face_detection/CenterNet/data/cal_fbc/annotations/val_cal_face.json'
    # test_json_file = '/home/jamesyen/my_project/face_detection/CenterNet/data/cal_fbc/annotations/test_cal_face.json'

    path = '/home/kenyo/compal-facecovered/CenterNetBase_2/data/pixsee_phase/Recipe_v3/'
    xml_path = path + 'Annotations/'
    train_json_file = path + 'annotations/train_cal_face.json'
    val_json_file = path + 'annotations/val_cal_face.json'
    test_json_file = path + 'annotations/test_cal_face.json'
    print(len(os.listdir(xml_path)))    
    seperate_data(xml_path, train_json_file, val_json_file, test_json_file)

    # print(coco)
    # json.dump(coco, open(json_file, 'w'))

    # add annotation with face,20180012875,1,[848, 445, 20, 29]
    # /home/yangna/deepblue/32_face_detect/centerface/data/wider_face/voc/35_Basketball_basketballgame_ball_35_79.xml
