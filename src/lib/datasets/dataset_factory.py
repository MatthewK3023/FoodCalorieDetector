from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .sample.ddd import DddDataset
from .sample.exdet import EXDetDataset
from .sample.ctdet import CTDetDataset
from .sample.multi_pose import MultiPoseDataset
from .sample.ctdet_multi import CTDetDataset_multi

# from .dataset.coco import COCO
# from .dataset.pascal import PascalVOC
# from .dataset.kitti import KITTI
# from .dataset.coco_hp import COCOHP
# from .dataset.cal_fb import CALFB
# from .dataset.cal_fbc import CALFBC
# from .dataset.wider_face import WIDERFACE
# from .dataset.cal_face import CALFACE
# from .dataset.open_face import OPENFACE
# from .dataset.mix_face import MIXFACE
# #from .dataset.cal_body import CALBODY
# from .dataset.open_body import OPENBODY
# from .dataset.mix_body import MIXBODY
# from .dataset.cal_body2 import CALBODY2
# from .dataset.open_body2 import OPENBODY2
# from .dataset.mix_body2 import MIXBODY2
# from .dataset.cal_all import CALALL
# from .dataset.open_all import OPENALL
from .dataset.mix_all import MIXALL
from .dataset.food_100 import FOOD100
from .dataset.food_100_crcnn import FOOD100CRCNN
from .dataset.food_100_frcnn import FOOD100FRCNN
from .dataset.foodsoft2_100 import FOOD100soft2
from .dataset.foodsoft3_100 import FOOD100soft3
# from .dataset.cal_multi import CALMULTI
# from .dataset.cal_body import CALBODY
# from .dataset.open_dataset import OPENDATASET
# from .dataset.pixsee_recipe1 import PIXSEE_r1
# from .dataset.pixsee_recipe2 import PIXSEE_r2
# from .dataset.pixsee_recipe3 import PIXSEE_r3


dataset_factory = {
  'mixall': MIXALL,
  'food100': FOOD100,
  'food100_crcnn': FOOD100CRCNN,
  'food100_frcnn': FOOD100FRCNN,
  'food100_soft2': FOOD100soft2,
  'food100_soft3': FOOD100soft3
}

_sample_factory = {
  'exdet': EXDetDataset,
  'ctdet': CTDetDataset,
  'ddd': DddDataset,
  'multi_pose': MultiPoseDataset,
  'ctdet_multi': CTDetDataset_multi
}


def get_dataset(dataset, task):
  class Dataset(dataset_factory[dataset], _sample_factory[task]):
    pass
  return Dataset
  
  
  
 