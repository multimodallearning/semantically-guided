import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import cm
from matplotlib import colors as mcolors
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt as edt
from scipy.ndimage import *

# simple B-spline transformation model using spatial transformer
def warpImage(img_input,def_x2):
    ''' img_input: image to warp (tensor)
        def_x2: deformation field (tensor)
    '''
    B, C, H, W = img_input.size() 
    # adding identity transform
    theta = (torch.zeros((B,2,3))).to(img_input.device) 
    theta[:,0,0] = 1.0
    theta[:,1,1] = 1.0
    x_grid = F.affine_grid(theta, img_input.size()) + def_x2.permute(0,2,3,1)
    output_im = F.grid_sample(img_input,x_grid)
    return output_im

def BsplineTrafo(img_input,def_x,kernelsize=31):
    ''' img_input: image to warp (tensor)
        def_x: deformation field (tensor)
        kernelsize: kernel size of smoothing kernel (5% and 3% of the largest image dimension)
    '''
    B, C, H, W = img_input.size()
    padsize = int(kernelsize//2)
    #estimated transformation for each sequence in batch
    def_x2 = F.upsample(def_x, size=(H,W), mode='bilinear')
    avg5 = nn.AvgPool2d((kernelsize,kernelsize),stride=(1,1),padding=(padsize,padsize)).to(img_input.device)
    def_x2 = avg5(avg5(avg5(def_x2)))
    img_warp = warpImage(img_input,def_x2)
    return img_warp, def_x2

def showFlow(def_x):
    x = def_x.squeeze().numpy()[0,:,:]
    y = def_x.squeeze().numpy()[1,:,:]
    #show flow map for numpy
    H, W = x.shape
    rho = np.sqrt(x*x+y*y)
    theta = np.arctan2(x,-y)
    theta2 = (-theta+np.pi)/(2.0*np.pi);
    rho = np.clip(rho/np.percentile(rho, 99),0,1)
    hsv = np.stack((theta2,rho,np.ones((H,W))),axis=2)
    rgb = mcolors.hsv_to_rgb(hsv)
    return rgb

# signed distance map
def sdm(matrix_seg):
    ''' matrix_seg: onehot label matrix (tensor)
    '''
    L, H, W = matrix_seg.size()
    for i in range(L-1):
        matrix_seg[i,:,:] = torch.from_numpy(edt(1-matrix_seg[i+1,:,:].numpy())+edt(matrix_seg[i+1,:,:].numpy()))
    return matrix_seg[:-1,:,:].unsqueeze(0)

def meancontourdist(x,y):
    ''' x: onehot label matrix of image 1 (tensor)
        y: onehot label matrix of image 2 (tensor)
    '''
    x_sdm = sdm(x.clone())
    y_sdm = sdm(y.clone())
    x_mask = x_sdm<=1
    y_mask = y_sdm<=1
    result = np.zeros((x_sdm.shape[0],x_sdm.shape[1]))
    for i in range(x_sdm.shape[0]):
        for j in range(x_sdm.shape[1]):
            a = y_sdm[i,j,:,:]
            b = x_sdm[i,j,:,:]
            result[i,j] = 0.5*a[x_mask[i,j,:,:]].mean() + 0.5*b[y_mask[i,j,:,:]].mean()
    return result

def overlaySegment(gray1,seg1):
    H, W = seg1.squeeze().size()
    colors=torch.FloatTensor([0,0,0,199,67,66,225,140,154,78,129,170,45,170,170,240,110,38,111,163,91,235,175,86,202,255,52,162,0,183]).view(-1,3)/255.0
    segs1 = labelMatrixOneHot(seg1.unsqueeze(0),8)

    seg_color = torch.mm(segs1.view(8,-1).t(),colors[:8,:]).view(H,W,3)
    alpha = torch.clamp(1.0 - 0.5*(seg1>0).float(),0,1.0)

    overlay = (gray1*alpha).unsqueeze(2) + seg_color*(1.0-alpha).unsqueeze(2)

    return overlay

def labelMatrixOneHot(segmentation, label_num):
    ''' segmentation: label image (tensor)
        label_num: max number of labels incl. background
    '''
    B, H, W = segmentation.size()
    values = segmentation.view(B,1,H,W).expand(B,label_num,H,W).to(segmentation.device)
    linspace = torch.linspace(0, label_num-1, label_num).long().view(1,label_num,1,1).expand(B,label_num,H,W).to(segmentation.device)
    matrix = (values.float()==linspace.float()).float().to(segmentation.device)
    for j in range(2,matrix.shape[1]):
        matrix[0,j,:,:] = matrix[0,j,:,:] #torch.from_numpy(grey_dilation(grey_erosion(,size=(3,3)),size=(3,3)))
    return matrix

def dice_coeff(outputs, labels, max_label):
    ''' outputs: onehot label matrix (tensor)
        labels: segmentation image (tensor)
        max_labels: max number of labels incl. background
    '''
    predicted = np.argmax(outputs.numpy(), axis = 1)
    dice = np.zeros((np.size(predicted,0),max_label-1))
    for label_num in range(1, max_label):
        intersect = np.sum(np.sum(np.logical_and(predicted==label_num, labels.numpy()==label_num), axis = 2), axis = 1)
        sum_label = np.sum(np.sum(predicted==label_num,axis=2),axis=1).astype('float32')
        sum_label += np.sum(np.sum(labels.numpy()==label_num,axis=2),axis=1).astype('float32')
        dice[:,label_num-1] = (2.0*intersect/(sum_label+1e-10))
    return dice


def augmentAffine(imgs,label_one,strength = 0.05):
    ''' img: image to augment (tensor)
        label_one: corresponding segmentation to augment (tensor)
        strength: random augmentation strength
    '''
    H_in = 320
    W_in = 260
    B = imgs.size(0)

    sample_grid = torch.cat((torch.linspace(-1,1,W_in).view(1,-1,1).repeat(H_in,1,1),\
          torch.linspace(-1,1,H_in).view(-1,1,1).repeat(1,W_in,1)),dim=2).view(1,H_in*W_in,2).cuda()
    affine = torch.randn(B,2,2).cuda()*strength+torch.eye(2,2).cuda().unsqueeze(0).repeat(B,1,1)
    translate = torch.randn(B,1,2).cuda()*strength
    sample_affine = (torch.bmm(sample_grid.repeat(B,1,1),affine)+translate).view(-1,H_in,W_in,2)

    label_affine = F.grid_sample(label_one,sample_affine).detach()
    img_affine = F.grid_sample(imgs,sample_affine).detach()
    return img_affine, label_affine

def jacobian_det(est_x2):
    ''' est_x2: deformation field (tensor)
    '''
    B,C,H,W = est_x2.size()
    est_pix = torch.zeros_like(est_x2)
    est_pix[:,0,:,:] = est_x2[:,0,:,:]*(H-1)/2.0
    est_pix[:,1,:,:] = est_x2[:,1,:,:]*(W-1)/2.0
    gradx = nn.Conv2d(2,2,(3,1),padding=(1,0),bias=False,groups=2)
    gradx.weight.data[:,0,:,0] = torch.tensor([-0.5,0,0.5]).view(1,3).repeat(2,1)
    grady = nn.Conv2d(2,2,(1,3),padding=(0,1),bias=False,groups=2)
    grady.weight.data[:,0,0,:] = torch.tensor([-0.5,0,0.5]).view(1,3).repeat(2,1)
    with torch.no_grad():
        J1 = gradx(est_pix)#+torch.tensor([1.0,0.0]).view(1,2,1,1)
        J2 = grady(est_pix)#+torch.tensor([0.0,1.0]).view(1,2,1,1)
    J = (J1[:,0,:,:]+1)*(J2[:,1,:,:]+1)-(J1[:,1,:,:])*(J2[:,0,:,:])
    return J