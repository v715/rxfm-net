
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#import pytorch3d.transforms as pt3d_xfms

# pool_obj = nn.AvgPool3d( 5, stride=1, padding=2)
def boundary_weighted_loss(true, pred, true_mask, pool_obj, boundary_weight=5):
  err = true - pred 
  pooled = pool_obj.forward( true_mask )
  boundaries = boundary_weight * (pooled > 0).float() * (pooled < 1).float()
  SE = (err * err)
  return (boundaries * SE + SE).mean()

def xfm_loss_MSE( true, pred, weight_R = 1.0, weight_T = 1.0 ):
  true_R = true[:,0:3,0:3]
  pred_R = pred[:,0:3,0:3]

  err_R = true_R - pred_R
  err_R = (err_R * err_R).mean()

  true_T = true[:,0:3,:]
  pred_T = pred[:,0:3,:]

  err_T = true_T - pred_T
  err_T = (err_T * err_T).mean()

  return weight_R * err_R + weight_T * err_T

#def xfm_loss_6D( true, pred, weight_R = 1.0, weight_T = 1.0 ):
#  true_R = pt3d_xfms.matrix_to_rotation_6d(true[:,0:3,0:3])
#  pred_R = pt3d_xfms.matrix_to_rotation_6d(pred[:,0:3,0:3])
#
#  err_R = true_R - pred_R
#  err_R = (err_R * err_R).mean()
#
#  true_T = true[:,0:3,:]
#  pred_T = pred[:,0:3,:]
#
#  err_T = true_T - pred_T
#  err_T = (err_T * err_T).mean()
#
#  return weight_R * err_R + weight_T * err_T

def compute_geodesic_distance_from_two_matrices(m1, m2):
    batch=m1.shape[0]
    m = torch.bmm(m1, m2.transpose(1,2)) #batch*3*3
    
    cos = (  m[:,0,0] + m[:,1,1] + m[:,2,2] - 1 )/2
    cos = torch.min(cos, torch.autograd.Variable(torch.ones(batch).cuda()) )
    cos = torch.max(cos, torch.autograd.Variable(torch.ones(batch).cuda())*-1 )
    
    theta = torch.acos(cos)
    
    #theta = torch.min(theta, 2*np.pi - theta)
    
    return theta

def xfm_loss_6D( true, pred, weight_R = 1.0, weight_T = 1.0 ):
    true_R = true[:,0:3,0:3]
    pred_R = pred[:,0:3,0:3]

    err_R = compute_geodesic_distance_from_two_matrices(true_R, pred_R)
    err_R = (err_R * err_R).mean()

    true_T = true[:,0:3,:]
    pred_T = pred[:,0:3,:]

    err_T = true_T - pred_T
    err_T = (err_T * err_T).mean()

    return weight_R * err_R + weight_T * err_T


def xfm_loss_geodesic( true, pred, weight_R = 1.0, weight_T = 1.0 ):
  # eq 10 from 
  # https://arxiv.org/abs/1803.05982

  true_R = true[:,0:3,0:3]
  pred_R = pred[:,0:3,0:3]

  true_R = true_R.float()
  pred_R = pred_R.float()

  err_R = (torch.einsum('bii->b',torch.matmul(pred_R,true_R)) - 1) / 2
  err_R = (torch.arccos(err_R)).mean()

  true_T = true[:,0:3,:]
  pred_T = pred[:,0:3,:]

  err_T = true_T - pred_T
  err_T = (err_T * err_T).mean()

  return weight_R * err_R + weight_T * err_T















