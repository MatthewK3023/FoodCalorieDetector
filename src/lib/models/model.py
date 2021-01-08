from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torchvision.models as models
import torch
import torch.nn as nn
import os
import re

from .networks.msra_resnet import get_pose_net
from .networks.dlav0 import get_pose_net as get_dlav0
# from .networks.pose_dla_dcn import get_pose_net as get_dla_dcn
# from .networks.resnet_dcn import get_pose_net as get_pose_net_dcn
from .networks.mobilenev2 import get_mobilenetv2
from .networks.large_hourglass import get_large_hourglass_net
from .networks.mobilenev2_multi import get_mobilenetv2_multi
from .networks.mobilenev2_TaskRouting import get_mobilenetv2_taskrouting
from .networks.mobilenev2_OnMerge import get_mobilenetv2_onmerge
from .networks.mobilenev2_mmoe import get_mobilenetv2_MMoE
from .networks.mobilenev2_withFPN import get_mobilenetv2_withFPN
from .networks.mobilenev2_withFPNzeropad import get_mobilenetv2_withFPNzeropad
from .networks.mobilenev2_withFPNbicubic import get_mobilenetv2_withFPNbicubic


_model_factory = {
  'res': get_pose_net, # default Resnet with deconv
  'dlav0': get_dlav0, # default DLAup
  # 'dla': get_dla_dcn,
  # 'resdcn': get_pose_net_dcn,
  'hourglass': get_large_hourglass_net,
  'mobilenetv2': get_mobilenetv2,
  'mobilenetv2multi': get_mobilenetv2_multi,
  'mobilenetv2taskrouting': get_mobilenetv2_taskrouting, 
  'mobilenetv2onmerge': get_mobilenetv2_onmerge,
  'mobilenetv2mmoe': get_mobilenetv2_MMoE,
  'mobilenetv2fpn': get_mobilenetv2_withFPN,
  'mobilenetv2fpnzeropad': get_mobilenetv2_withFPNzeropad,
  'mobilenetv2fpnbicubic': get_mobilenetv2_withFPNbicubic,
}

def create_model(arch, heads, head_conv, device):
  num_layers = int(arch[arch.find('_') + 1:]) if '_' in arch else 0
  arch = arch[:arch.find('_')] if '_' in arch else arch
  get_model = _model_factory[arch]
  model = get_model(num_layers=num_layers, heads=heads, head_conv=head_conv, device=device)
  return model

def load_model(model, model_path, arch='mobilenetv2_10', optimizer=None, resume=False, 
               lr=None, lr_step=None):
  start_epoch = 0
  
  checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
  print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
  state_dict_ = checkpoint['state_dict']
  state_dict = {}
  
  arch = arch[:arch.find('_')] if '_' in arch else arch

  # convert data_parallal to model
  for k in state_dict_:
    if k.startswith('module') and not k.startswith('module_list'):
      state_dict[k[7:]] = state_dict_[k]
    else:
      state_dict[k] = state_dict_[k]
  model_state_dict = model.state_dict()

  # warnings message
  msg = 'If you see this, your model does not fully load the ' + \
        'pre-trained weight. Please make sure ' + \
        'you have correctly specified --arch xxx ' + \
        'or set the correct --num_classes for your own dataset.'

  # check loaded parameters and created model parameters

  # if arch == 'mobilenetv2mmoe'
  if arch == 'mobilenetv2mmoe':
      model_path_1 = '/Users/matthewk/Desktop/lab_workspace/compal/compal_face_2/exp/\
                      ctdet/wider_mobile/model_last.pth'
      model_path_2 = '/Users/matthewk/Desktop/lab_workspace/compal/compal_face_2/exp/\
                      ctdet/calbody_mobile_l1/model_last.pth'
      model_path_3 = None
      checkpoint_1 = torch.load(model_path_1, map_location=lambda storage, loc: storage)
      state_dict_1_ = checkpoint_1['state_dict']
      state_dict_2 = {}
      checkpoint_2 = torch.load(model_path_2, map_location=lambda storage, loc: storage)
      state_dict_2_ = checkpoint_2['state_dict']
      state_dict_1 = {}
      # checkpoint_3 = torch.load(model_path_3, map_location=lambda storage, loc: storage)
      # state_dict_3_ = checkpoint_3['state_dict']
      state_dict_3 = {}

      for k in state_dict_1_:
          if k.startswith('module') and not k.startswith('module_list'):
              state_dict_1[k[7:]] = state_dict_1_[k]
          else:
              state_dict_1[k] = state_dict_1_[k]
              
      for k in state_dict_2_:
          if k.startswith('module') and not k.startswith('module_list'):
              state_dict_2[k[7:]] = state_dict_2_[k]
          else:
              state_dict_2[k] = state_dict_2_[k]

      # for k in state_dict_3_:
      #     if k.startswith('module') and not k.startswith('module_list'):
      #         state_dict_3[k[7:]] = state_dict_3_[k]
      #     else:
      #         state_dict_3[k] = state_dict_3_[k]
        
      model_state_dict = model.state_dict()

      for i, state_dict in enumerate([state_dict_1, state_dict_2, state_dict_3]):
          for k in model_state_dict:   
              if k.startswith('base_%d'%(i+1)):
                  k_base = k[:4]+k[6:]
              elif k.startswith('dla_up_%d'%(i+1)):
                  k_base = k[:6]+k[8:]
              elif not k.startswith('base') and not k.startswith('dla_up'):
                  k_base = k
              else:
                  k_base = ''
              #print(k_base)
              if k_base in state_dict:
                  if state_dict[k_base].shape != model_state_dict[k].shape:
                      print('k_base: ', state_dict[k_base].shape, k_base)
                      print('k     :', model_state_dict[k].shape, k)
                      print('Skip loading parameter {}, required shape{}, '\
                                  'loaded shape{}. {}'.format(
                          k, model_state_dict[k].shape, state_dict[k_base].shape, msg))

          for k in model_state_dict:
              if k.startswith('base_%d'%(i+1)):
                  k_base = k[:4]+k[6:]
              elif k.startswith('dla_up_%d'%(i+1)):
                  k_base = k[:6]+k[8:]
              if k.startswith('base_%d'%(i+1)) or k.startswith('dla_up_%d'%(i+1)):
                  model_state_dict[k] = state_dict[k_base]


  # if arch == 'mobilenetv2taskrouting'
  elif arch == 'mobilenetv2taskrouting':
      model_state_dict_renamed = {}

      for k in model_state_dict:
          #print(k)
          if k.startswith('base'):
              model_state_dict_renamed['base.feature_'+k[4:]] = \
                                      model_state_dict[k]
          elif k.startswith('dla_up1'):
              model_state_dict_renamed['dla_up.conv'+k[7:]] = \
                                      model_state_dict[k]
          elif k.startswith('dla_up2'):
              model_state_dict_renamed['dla_up.conv_last'+k[7:]] = \
                                      model_state_dict[k]
          elif k.startswith('dla_up3'):
              model_state_dict_renamed['dla_up.up_0.up'+k[7:]] = \
                                      model_state_dict[k]
          elif k.startswith('dla_up4'):
              model_state_dict_renamed['dla_up.up_1.up'+k[7:]] = \
                                      model_state_dict[k]
          elif k.startswith('dla_up5'):
              model_state_dict_renamed['dla_up.up_2.up'+k[7:]] = \
                                      model_state_dict[k]
          else:
              model_state_dict_renamed[k] = model_state_dict[k]

      del model_state_dict
      model_state_dict = model_state_dict_renamed
      del model_state_dict_renamed

      for k in state_dict:
          if k in model_state_dict:
              if state_dict[k].shape != model_state_dict[k].shape:
                  print('Skip loading parameter {}, required shape{}, '\
                      'loaded shape{}. {}'.format(
                      k, model_state_dict[k].shape, state_dict[k].shape, msg))
                  state_dict[k] = model_state_dict[k]

      for k in model_state_dict:
              if not (k in state_dict):
                  if not k.endswith('_unit_mapping'):
                      print('No param {}.'.format(k) + msg)
                  state_dict[k] = model_state_dict[k]
      model.load_state_dict(state_dict, strict=False)



  # if arch == 'mobilenetv2onmerge'
  elif arch == 'mobilenetv2onmerge':
    model_path_1 = '/home/kenyo/compal-facecovered/CenterNetBase_2/exp/ctdet/wider_mobile/model_last.pth'
    model_path_2 = '/home/kenyo/compal-facecovered/CenterNetBase_2/exp/ctdet/calbody_mobile_l1/model_last.pth'
    checkpoint_1 = torch.load(model_path_1, map_location=lambda storage, loc: storage)
    checkpoint_2 = torch.load(model_path_2, map_location=lambda storage, loc: storage)
    state_dict_1_ = checkpoint_1['state_dict']
    state_dict_2_ = checkpoint_2['state_dict']
    state_dict_1 = {}
    state_dict_2 = {}

    for k in state_dict_1_:
        if k.startswith('module') and not k.startswith('module_list'):
            state_dict_1[k[7:]] = state_dict_1_[k]
        else:
            state_dict_1[k] = state_dict_1_[k]
            
    for k in state_dict_2_:
        if k.startswith('module') and not k.startswith('module_list'):
            state_dict_2[k[7:]] = state_dict_2_[k]
        else:
            state_dict_2[k] = state_dict_2_[k]
            
    model_state_dict = model.state_dict()
    for i, k in enumerate(model_state_dict):
      if '_t1' in k:
          if k.startswith('base'):
              k_base = k[:4]+'.feature_'+k[4]+k[8:]
          elif k.startswith('dla_up1'):
              k_base = 'dla_up.conv'+k[10:]
          elif k.startswith('dla_up2'):
              k_base = 'dla_up.conv_last'+k[10:]
          elif k.startswith('dla_up3'):
              k_base = 'dla_up.up_0.up'+k[10:]
          elif k.startswith('dla_up4'):
              k_base = 'dla_up.up_1.up'+k[10:]
          elif k.startswith('dla_up5'):
              k_base = 'dla_up.up_2.up'+k[10:]
          else:
              k_base = re.sub('_t1', '', k)
          if k_base in state_dict_1:
              model_state_dict[k] = state_dict_1[k_base]
      
      elif '_t2' in k:
          if k.startswith('base'):
              k_base = k[:4]+'.feature_'+k[4]+k[8:]
          elif k.startswith('dla_up1'):
              k_base = 'dla_up.conv'+k[10:]
          elif k.startswith('dla_up2'):
              k_base = 'dla_up.conv_last'+k[10:]
          elif k.startswith('dla_up3'):
              k_base = 'dla_up.up_0.up'+k[10:]
          elif k.startswith('dla_up4'):
              k_base = 'dla_up.up_1.up'+k[10:]
          elif k.startswith('dla_up5'):
              k_base = 'dla_up.up_2.up'+k[10:]
          else:
              k_base = re.sub('_t2', '', k)
          if k_base in state_dict_2:
              model_state_dict[k] = state_dict_2[k_base]



  # if arch == 'mobilenetv2'
  else:
    msg = 'If you see this, your model does not fully load the ' + \
          'pre-trained weight. Please make sure ' + \
          'you have correctly specified --arch xxx ' + \
          'or set the correct --num_classes for your own dataset.'
    for k in state_dict:
      if k in model_state_dict:
        if state_dict[k].shape != model_state_dict[k].shape:
          print('Skip loading parameter {}, required shape{}, '\
                'loaded shape{}. {}'.format(
            k, model_state_dict[k].shape, state_dict[k].shape, msg))
          state_dict[k] = model_state_dict[k]
      else:
        print('Drop parameter {}.'.format(k) + msg)
    for k in model_state_dict:
      if not (k in state_dict):
        print('No param {}.'.format(k) + msg)
        state_dict[k] = model_state_dict[k]
    model.load_state_dict(state_dict, strict=False)


  # resume optimizer parameters
  if optimizer is not None and resume:
    if 'optimizer' in checkpoint:
      optimizer.load_state_dict(checkpoint['optimizer'])
      start_epoch = checkpoint['epoch']
      start_lr = lr
      for step in lr_step:
        if start_epoch >= step:
          start_lr *= 0.1
      for param_group in optimizer.param_groups:
        param_group['lr'] = start_lr
      print('Resumed optimizer with start lr', start_lr)
    else:
      print('No optimizer parameters in checkpoint.')
  if optimizer is not None:
    return model, optimizer, start_epoch
  else:
    return model

def save_model(path, epoch, model, optimizer=None):
  if isinstance(model, torch.nn.DataParallel):
    state_dict = model.module.state_dict()
  else:
    state_dict = model.state_dict()
  data = {'epoch': epoch,
          'state_dict': state_dict}
  if not (optimizer is None):
    data['optimizer'] = optimizer.state_dict()
  torch.save(data, path)

