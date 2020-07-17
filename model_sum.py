import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import build_targets
from groupConv import InvertedResidual, conv_bn, conv_1x1_bn, _extendedLayers, _outputLayers


class yoloOutputAndLoss(nn.Module):
    def __init__(self, anchors, numOfClass, imgSiz, opt):
        super(yoloOutputAndLoss, self).__init__()
        self._opt = opt
        self.anchors = anchors
        self.numAnchors = len(anchors)
        self.numOfClass = numOfClass
        self.obj_scale = 1
        self.noobj_scale = 0.5
        self.coord_scale = 5
        self.gridSiz = 0
        self.imgSiz = imgSiz
        self.ignore_thres = 0.5
        self.mse_loss = nn.MSELoss(reduction="sum")
        self.bce_loss = nn.BCELoss(reduction="sum")
        self.device = torch.device(self._opt.device if torch.cuda.is_available() else 'cpu')
        self.gpu = opt.device
        
    def compute_grid_offsets(self, grid_size, cuda=True):

        self.gridSiz = grid_size
        g = self.gridSiz
        FloatTensor=torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.stride = self.imgSiz/self.gridSiz
        self.gridX=torch.arange(g).repeat(g,1).view([1,1,g,g]).type(FloatTensor).to(self.device)
        self.gridY=torch.arange(g).repeat(g,1).t().view([1,1,g,g]).type(FloatTensor).to(self.device)
        self.scaledAnchors=FloatTensor([(a_w/self.stride,a_h/self.stride) for a_w,a_h in self.anchors]).to(self.device)
        self.anchor_w = self.scaledAnchors[:, 0:1].view((1, self.numAnchors, 1, 1)).to(self.device)
        self.anchor_h = self.scaledAnchors[:, 1:2].view((1, self.numAnchors, 1, 1)).to(self.device)
    
    def forward(self, x, targets=None):
        FloatTensor=torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        
        numBatch=x.shape[0]
        gridSiz=x.shape[2]
        
        # Create grid offset if not created/different
        if gridSiz != self.gridSiz:
            self.compute_grid_offsets(gridSiz, cuda=x.is_cuda)
        
        # Separate output to managable blocks
        prediction = (x.view(numBatch, self.numAnchors, self.numOfClass+5, gridSiz, -1)
                    .permute(0,1,3,4,2).contiguous()) 
        # Note:prediction is (b,anc,grid,grid,numclasses+5)
        # Note: output sequence is (x,y,w,h,conf,classes:)
        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid(prediction[..., 1])
        w = prediction[..., 2]
        h = prediction[..., 3]
        predConf = torch.sigmoid(prediction[..., 4])
        # Note: x,y,w,h,predConf -> (b,box,grid,grid)
        predCls=torch.sigmoid(prediction[...,5:])   # Question: use sigmoid?
        # Note: predCls -> (b,box,grid,grid,numClass)
        
        # Convert to grid space
        pred_boxes = FloatTensor(prediction[..., :4].shape, device=self.gpu)    # Note:becomes (b,box,grid,grid,4)
        pred_boxes[..., 0] = x.data+self.gridX  # Note:convert x to grid space
        pred_boxes[..., 1] = y.data+self.gridY
        pred_boxes[..., 2] = torch.exp(w.data)*self.anchor_w
        pred_boxes[..., 3] = torch.exp(h.data)*self.anchor_h
        
        output = torch.cat(
            (
                pred_boxes.view(numBatch, -1, 4) * self.stride,     # Note:becomes (b,box*grid*grid,4)
                predConf.view(numBatch, -1, 1),     # Note: (b,box*grid*grid,1)
                predCls.view(numBatch, -1, self.numOfClass)), -1)    # Note: (b,box*grid*grid,numClass
            
        if targets is None:
            return output, 0
        else:
            iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf = build_targets(
                pred_boxes=pred_boxes, 
                pred_cls=predCls,
                target=targets,
                anchors=self.scaledAnchors,
                ignore_thres=self.ignore_thres,
                cuda=self._opt.device
            )
            
            # Loss : Mask outputs to ignore non-existing objects (except with conf. loss)
            loss_x = self.mse_loss(x[obj_mask], tx[obj_mask])
            loss_y = self.mse_loss(y[obj_mask], ty[obj_mask])
            loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])
            loss_h = self.mse_loss(h[obj_mask], th[obj_mask])
            loss_conf_obj = self.bce_loss(predConf[obj_mask], tconf[obj_mask])
            loss_conf_noobj = self.bce_loss(predConf[noobj_mask], tconf[noobj_mask])
            loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj
            loss_cls = self.bce_loss(predCls[obj_mask], tcls[obj_mask])
            total_loss = ((loss_y + loss_x)*self.coord_scale + 
                          (loss_w + loss_h)*self.coord_scale + 
                          loss_conf + loss_cls)
            
            return output, total_loss
                                                                               

class objDetNet(nn.Module):
    def __init__(self, opt):
        super(objDetNet, self).__init__()
        self._opt = opt
        
        self.conv0=conv_bn(1, 32, 2)     # 416->208
        self.conv1=InvertedResidual(32,16,1,1)
        self.conv2=nn.Sequential(InvertedResidual(16,24,2,6),InvertedResidual(24,24,1,6))   # 208->104
        self.conv3=nn.Sequential(InvertedResidual(24,32,2,6),InvertedResidual(32,32,1,6)
                                ,InvertedResidual(32,32,1,6))#104->52
        self.conv4=nn.Sequential(InvertedResidual(32,64,2,6),InvertedResidual(64,64,1,6)
                                ,InvertedResidual(64,64,1,6),InvertedResidual(64,64,1,6))#52->26
        self.conv5=nn.Sequential(InvertedResidual(64,96,1,6),InvertedResidual(96,96,1,6)
                                ,InvertedResidual(96,96,1,6))
        self.conv6=nn.Sequential(InvertedResidual(96,160,2,6),InvertedResidual(160,160,1,6)
                                ,InvertedResidual(160,160,1,6))     # 26->13
        
        self.convEx1=_extendedLayers(160,512)
        self.convOt1=_outputLayers(512,6*3)
        
        self.convUp1=nn.Sequential(conv_1x1_bn(512, 256),nn.ConvTranspose2d(256,256,3,2,1,1,256))
        
        self.convEx2=_extendedLayers(320,256)
        self.convOt2=_outputLayers(256,6*3)
        self.convUp2=nn.Sequential(conv_1x1_bn(256, 128),nn.ConvTranspose2d(128,128,3,2,1,1,128))
        
        self.convEx3=_extendedLayers(160,256)
        self.convOt3=_outputLayers(256,6*3)

        self.yolo13 = yoloOutputAndLoss([(116, 90), (156, 198), (373, 326)], 1, self._opt.imgSquareSize, self._opt)
        self.yolo26=yoloOutputAndLoss([(30, 61), (62, 45), (59, 119)], 1, self._opt.imgSquareSize, self._opt)
        self.yolo52=yoloOutputAndLoss([(10, 13), (16, 30), (33, 23)], 1, self._opt.imgSquareSize, self._opt)
        
    def loadPretrainedParams(self):
        deviceBool = next(self.parameters()).is_cuda
        device = torch.device(self._opt.device if deviceBool else "cpu")
        try:
            pretrainedDict = torch.load(self._opt.pretrainedParamFile, map_location=device.type)
            modelDict=self.state_dict()
            pretrainedDict = {k: v for k, v in pretrainedDict.items() if k in modelDict}
            modelDict.update(pretrainedDict)
            self.load_state_dict(modelDict)
        except:
            print("Can't load pre-trained parameter files")

    def forward(self,x,target=None):    # input x: [48, 3, 416, 416]        target: [96, 6]
        x = self.conv0(x)   # [48, 32, 208, 208]
        x = self.conv1(x)   # [48, 16, 208, 208]
        x = self.conv2(x)   # [48, 24, 104, 104]
        x = self.conv3(x)   # [48, 32, 52, 52]
        xR2 = x
        x = self.conv4(x)   # [48, 64 26, 26]
        xR1 = x
        x = self.conv5(x)   # [48, 96,26, 26]
        x = self.conv6(x)   # [48, 160, 13, 13]
        x = self.convEx1(x)     # [48, 512, 13, 13]
        xOt1 = self.convOt1(x)  # [48, 255, 13, 13]
        x = self.convUp1(x)     # [48, 256, 26, 26]
        x = torch.cat([x, xR1], dim=1)      # [48, 320, 26, 26]
        
        x = self.convEx2(x)     # [48, 256, 26, 26]
        xOt2 = self.convOt2(x)      # [48, 255, 26, 26]
        x = self.convUp2(x)         # [48, 128, 52, 52]
        x = torch.cat([x, xR2], dim=1)      # [48, 160, 52, 52]
        
        x = self.convEx3(x)     # [48, 256, 52, 52]
        xOt3 = self.convOt3(x)      # [48, 255, 52, 52]
        
        out13, loss13 = self.yolo13(xOt1, target)
        out26, loss26 = self.yolo26(xOt2, target)
        out52, loss52 = self.yolo52(xOt3, target)
        
        return [out13, out26, out52], loss13+loss26+loss52
