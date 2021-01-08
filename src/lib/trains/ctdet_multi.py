from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np

from models.losses import FocalLoss
from models.losses import RegL1Loss, RegLoss, NormRegL1Loss, RegWeightedL1Loss
from models.losses import DynamicLoss
from models.losses import RegL1Loss_acc, RegLoss, NormRegL1Loss, RegWeightedL1Loss
from models.decode import ctdet_decode
from models.utils import _sigmoid
from utils.debugger import Debugger
from utils.post_process import ctdet_post_process
from utils.oracle_utils import gen_oracle_map
from .base_trainer_multi import BaseTrainer

task = 2
loss_type = 'normal'
'''
  loss_type
    'normal':
    'weighted':
    'geometric':
    'dynamic':
'''

# if loss_type == 'weighted'
task_weight = [0.7, 0.3]


class CtdetLoss_multi(torch.nn.Module):
  def __init__(self, opt):
    super(CtdetLoss_multi, self).__init__()
    if loss_type == 'dynamic':
      self.crit = torch.nn.MSELoss() if opt.mse_loss else DynamicLoss()
      self.crit_reg = RegL1Loss_acc() if opt.reg_loss == 'l1' else \
                RegLoss() if opt.reg_loss == 'sl1' else None
      self.crit_wh = torch.nn.L1Loss(reduction='sum') if opt.dense_wh else \
                NormRegL1Loss() if opt.norm_wh else \
                RegWeightedL1Loss() if opt.cat_spec_wh else self.crit_reg
    else:
      self.crit = torch.nn.MSELoss() if opt.mse_loss else DynamicLoss()
      self.crit_reg = RegL1Loss_acc() if opt.reg_loss == 'l1' else \
                RegLoss() if opt.reg_loss == 'sl1' else None
      self.crit_wh = torch.nn.L1Loss(reduction='sum') if opt.dense_wh else \
                NormRegL1Loss() if opt.norm_wh else \
                RegWeightedL1Loss() if opt.cat_spec_wh else self.crit_reg
    self.opt = opt

  # def forward(self, outputs, batch):
  #   opt = self.opt
  #   hm_loss, wh_loss, off_loss = 0, 0, 0
  #   for s in range(opt.num_stacks):
  #     output = outputs[s]
  #     if not opt.mse_loss:
  #       output['hm'] = _sigmoid(output['hm'])

  #     if opt.eval_oracle_hm:
  #       output['hm'] = batch['hm']
  #     if opt.eval_oracle_wh:
  #       output['wh'] = torch.from_numpy(gen_oracle_map(
  #         batch['wh'].detach().cpu().numpy(), 
  #         batch['ind'].detach().cpu().numpy(), 
  #         output['wh'].shape[3], output['wh'].shape[2])).to(opt.device)
  #     if opt.eval_oracle_offset:
  #       output['reg'] = torch.from_numpy(gen_oracle_map(
  #         batch['reg'].detach().cpu().numpy(), 
  #         batch['ind'].detach().cpu().numpy(), 
  #         output['reg'].shape[3], output['reg'].shape[2])).to(opt.device)

  #     hm_loss += self.crit(output['hm'], batch['hm']) / opt.num_stacks
  #     if opt.wh_weight > 0:
  #       if opt.dense_wh:
  #         mask_weight = batch['dense_wh_mask'].sum() + 1e-4
  #         wh_loss += (
  #           self.crit_wh(output['wh'] * batch['dense_wh_mask'],
  #           batch['dense_wh'] * batch['dense_wh_mask']) / 
  #           mask_weight) / opt.num_stacks
  #       elif opt.cat_spec_wh:
  #         wh_loss += self.crit_wh(
  #           output['wh'], batch['cat_spec_mask'],
  #           batch['ind'], batch['cat_spec_wh']) / opt.num_stacks
  #       else:
  #         wh_loss += self.crit_reg(
  #           output['wh'], batch['reg_mask'],
  #           batch['ind'], batch['wh']) / opt.num_stacks
      
  #     if opt.reg_offset and opt.off_weight > 0:
  #       off_loss += self.crit_reg(output['reg'], batch['reg_mask'],
  #                            batch['ind'], batch['reg']) / opt.num_stacks
        
  #   loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + \
  #          opt.off_weight * off_loss
  #   loss_stats = {'loss': loss, 'hm_loss': hm_loss,
  #                 'wh_loss': wh_loss, 'off_loss': off_loss}
  #   return loss, loss_stats

  def forward(self, outputs_1, outputs_2, batch):
    opt = self.opt
    hm_loss_1, wh_loss_1, off_loss_1 = 0, 0, 0
    hm_loss_2, wh_loss_2, off_loss_2 = 0, 0, 0
    for s in range(opt.num_stacks):
      output_1 = outputs_1[s]
      output_2 = outputs_2[s]
      if not opt.mse_loss:
        output_1['hm_1'] = _sigmoid(output_1['hm_1'])
        output_2['hm_2'] = _sigmoid(output_2['hm_2'])

      if opt.eval_oracle_hm:
        output_1['hm_1'] = batch['hm_1']
        output_2['hm_2'] = batch['hm_2']

      if opt.eval_oracle_wh:
        output_1['wh_1'] = torch.from_numpy(gen_oracle_map(
          batch['wh_1'].detach().cpu().numpy(), 
          batch['ind_1'].detach().cpu().numpy(), 
          output_1['wh_1'].shape[3], output_1['wh_1'].shape[2])).to(opt.device)
        output_2['wh_2'] = torch.from_numpy(gen_oracle_map(
          batch['wh_2'].detach().cpu().numpy(), 
          batch['ind_2'].detach().cpu().numpy(), 
          output_2['wh_2'].shape[3], output_2['wh_2'].shape[2])).to(opt.device)

      if opt.eval_oracle_offset:
        output_1['reg_1'] = torch.from_numpy(gen_oracle_map(
          batch['reg_1'].detach().cpu().numpy(), 
          batch['ind_1'].detach().cpu().numpy(), 
          output_1['reg_1'].shape[3], output_1['reg_1'].shape[2])).to(opt.device)
        output_2['reg_2'] = torch.from_numpy(gen_oracle_map(
          batch['reg_2'].detach().cpu().numpy(), 
          batch['ind_2'].detach().cpu().numpy(), 
          output_2['reg_2'].shape[3], output_2['reg_2'].shape[2])).to(opt.device)

      hm_loss_1 += self.crit(output_1['hm_1'], batch['hm_1']) / opt.num_stacks
      hm_loss_2 += self.crit(output_2['hm_2'], batch['hm_2']) / opt.num_stacks

      if opt.wh_weight > 0:
        if opt.dense_wh:
          mask_weight = batch['dense_wh_mask'].sum() + 1e-4      # '''
          wh_loss += (
            self.crit_wh(output['wh'] * batch['dense_wh_mask'],  # 不可用，沒更新成multitask版本
            batch['dense_wh'] * batch['dense_wh_mask']) / 
            mask_weight) / opt.num_stacks                        # '''

        elif opt.cat_spec_wh:
          wh_loss_1 += self.crit_wh(
            output_1['wh_1'], batch['cat_spec_mask_1'],
            batch['ind_1'], batch['cat_spec_wh_1']) / opt.num_stacks
          wh_loss_2 += self.crit_wh(
            output_2['wh_2'], batch['cat_spec_mask_2'],
            batch['ind_2'], batch['cat_spec_wh_2']) / opt.num_stacks

        else:
          wh_loss_1 += self.crit_reg(
            output_1['wh_1'], batch['reg_mask_1'],
            batch['ind_1'], batch['wh_1'], output_1['hm_1'], batch['hm_1']) / opt.num_stacks
          wh_loss_2 += self.crit_reg(
            output_2['wh_2'], batch['reg_mask_2'],
            batch['ind_2'], batch['wh_2'], output_1['hm_2'], batch['hm_2']) / opt.num_stacks
      
      if opt.reg_offset and opt.off_weight > 0:
        off_loss_1 += self.crit_reg(output_1['reg_1'], batch['reg_mask_1'],
                             batch['ind_1'], batch['reg_1'],
                             output_1['hm_1'], batch['hm_1']) / opt.num_stacks
        off_loss_2 += self.crit_reg(output_2['reg_2'], batch['reg_mask_2'],
                             batch['ind_2'], batch['reg_2'],
                             output_1['hm_2'], batch['hm_2']) / opt.num_stacks
        
    loss_1 = opt.hm_weight * hm_loss_1 + opt.wh_weight * wh_loss_1 + \
           opt.off_weight * off_loss_1
    loss_2 = opt.hm_weight * hm_loss_2 + opt.wh_weight * wh_loss_2 + \
           opt.off_weight * off_loss_2

    loss_type = 'normal'
    if loss_type == 'weighted':
      loss_weight_1, loss_weight_2 = task_weight[0], task_weight[1]
      loss = (loss_weight_1 * loss_1) + (loss_weight_2 * loss_2)
    elif loss_type == 'geometric':
      n = task
      loss = (loss_1 * loss_2)**(1/n)
    else:
      loss_weight_1, loss_weight_2 = 1.0, 1.0
      loss = (loss_weight_1 * loss_1) + (loss_weight_2 * loss_2)
      
    loss_stats = {'loss': loss, 'loss_1': loss_1, 'hm_loss_1': hm_loss_1,
                  'wh_loss_1': wh_loss_1, 'off_loss_1': off_loss_1, 
                  'loss_2': loss_2, 'hm_loss_2': hm_loss_2,
                  'wh_loss_2': wh_loss_2, 'off_loss_2': off_loss_2}
    return loss, loss_stats

class CtdetTrainer_multi(BaseTrainer):
  def __init__(self, opt, model, optimizer=None):
    super(CtdetTrainer_multi, self).__init__(opt, model, optimizer=optimizer)
  
  def _get_losses(self, opt):
    # loss_states = ['loss', 'hm_loss', 'wh_loss', 'off_loss']
    loss_states = ['loss', 'loss_1', 'hm_loss_1', 'wh_loss_1', 'off_loss_1',
                  'loss_2', 'hm_loss_2', 'wh_loss_2', 'off_loss_2']
    loss = CtdetLoss_multi(opt)
    return loss_states, loss

  def debug(self, batch, output, iter_id):
    opt = self.opt
    reg = output['reg'] if opt.reg_offset else None
    dets = ctdet_decode(
      output['hm'], output['wh'], reg=reg,
      cat_spec_wh=opt.cat_spec_wh, K=opt.K)
    dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
    dets[:, :, :4] *= opt.down_ratio
    dets_gt = batch['meta']['gt_det'].numpy().reshape(1, -1, dets.shape[2])
    dets_gt[:, :, :4] *= opt.down_ratio
    for i in range(1):
      debugger = Debugger(
        dataset=opt.dataset, ipynb=(opt.debug==3), theme=opt.debugger_theme)
      img = batch['input'][i].detach().cpu().numpy().transpose(1, 2, 0)
      img = np.clip(((
        img * opt.std + opt.mean) * 255.), 0, 255).astype(np.uint8)
      pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
      gt = debugger.gen_colormap(batch['hm'][i].detach().cpu().numpy())
      debugger.add_blend_img(img, pred, 'pred_hm')
      debugger.add_blend_img(img, gt, 'gt_hm')
      debugger.add_img(img, img_id='out_pred')
      for k in range(len(dets[i])):
        if dets[i, k, 4] > opt.center_thresh:
          debugger.add_coco_bbox(dets[i, k, :4], dets[i, k, -1],
                                 dets[i, k, 4], img_id='out_pred')

      debugger.add_img(img, img_id='out_gt')
      for k in range(len(dets_gt[i])):
        if dets_gt[i, k, 4] > opt.center_thresh:
          debugger.add_coco_bbox(dets_gt[i, k, :4], dets_gt[i, k, -1],
                                 dets_gt[i, k, 4], img_id='out_gt')

      if opt.debug == 4:
        debugger.save_all_imgs(opt.debug_dir, prefix='{}'.format(iter_id))
      else:
        debugger.show_all_imgs(pause=True)

  def save_result(self, output, batch, results):
    reg = output['reg'] if self.opt.reg_offset else None
    dets = ctdet_decode(
      output['hm'], output['wh'], reg=reg,
      cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)
    dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
    dets_out = ctdet_post_process(
      dets.copy(), batch['meta']['c'].cpu().numpy(),
      batch['meta']['s'].cpu().numpy(),
      output['hm'].shape[2], output['hm'].shape[3], output['hm'].shape[1])
    results[batch['meta']['img_id'].cpu().numpy()[0]] = dets_out[0]