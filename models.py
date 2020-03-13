import torch.nn.functional as F
from utils.parse_config import *
from utils.utils import *

def create_modules(module_defs, img_size, arc):
    # Constructs module list of layer blocks from module configuration in module_defs
    hyperparams = module_defs.pop(0)
    output_filters = [int(hyperparams['channels'])]
    module_list = nn.ModuleList()
    routs = []  # list of index values, using which feature maps can be added or concatenated.
    yolo_index = -1

    for i, mdef in enumerate(module_defs):
        modules = nn.Sequential()

        if mdef['type'] == 'convolutional':
            bn = int(mdef['batch_normalize'])
            filters = int(mdef['filters'])
            size = int(mdef['size'])
            stride = int(mdef['stride'])
            pad = (size - 1) // 2 if int(mdef['pad']) else 0
            modules.add_module('Conv2d', nn.Conv2d(in_channels=output_filters[-1],out_channels=filters,kernel_size=size,stride=stride,padding=pad,bias=not bn))  #in_channels taken from output_filters history.
            if bn:
                modules.add_module('BatchNorm2d', nn.BatchNorm2d(filters, momentum=0.1))
            if mdef['activation'] == 'leaky':
                modules.add_module('activation', nn.LeakyReLU(0.1, inplace=True))

        elif mdef['type'] == 'upsample':
            modules = nn.Upsample(scale_factor=int(mdef['stride']), mode='nearest')    #in upsample case filter remain same as previos convolution filter shape.

        elif mdef['type'] == 'shortcut':
            filters = output_filters[int(mdef['from'])]  #filter size from previous layers
            layer = int(mdef['from'])
            routs.extend([i + layer if layer < 0 else layer])  #default index is backward direction but we convert it to forward index [0,1,2,3,4] herre -1 is 3

        elif mdef['type'] == 'route':
            layers = [int(x) for x in mdef['layers'].split(',')]
            filters = sum([output_filters[i + 1 if i > 0 else i] for i in layers])   #sum filters sizes from both places, since index starts from 0 add 1 to it
            routs.extend([l if l > 0 else l + i for l in layers])                    #default index is backward direction but we convert it to forward index [0,1,2,3,4] herre -1 is 3

        elif mdef['type'] == 'yolo':
            yolo_index += 1
            mask = [int(x) for x in mdef['mask'].split(',')]  # anchor mask

            modules = YOLOLayer(anchors=mdef['anchors'][mask],  # anchor list using indexing
                                nc=int(mdef['classes']),  # number of classes
                                arc=arc)  # yolo architecture

        else:
            print('Warning: Unrecognized Layer Type: ' + mdef['type'])

        module_list.append(modules)           #appending individual module to module list
        output_filters.append(filters)        #appending filters of every module

    return module_list, routs

class YOLOLayer(nn.Module):
    def __init__(self, anchors, nc, arc):
        super(YOLOLayer, self).__init__()

        self.anchors = torch.Tensor(anchors)
        self.na = len(anchors)  # number of anchors (3)
        self.nc = nc  # number of classes (80)
        self.no = nc + 5  # number of outputs
        self.nx = 0  # initialize number of x gridpoints
        self.ny = 0  # initialize number of y gridpoints
        self.arc = arc

    def forward(self, p, img_size, var=None):
        bs, _, ny, nx = p.shape  # bs, 255, 13, 13

        if (self.nx, self.ny) != (nx, ny):
            create_grids(self, img_size, (nx, ny), p.device, p.dtype)

        # p.view(bs, 255, 13, 13) -- > (bs, 3, 13, 13, 85)  # (bs, anchors, grid, grid, classes + xywh)
        p = p.view(bs, self.na, self.no, self.ny, self.nx).permute(0, 1, 3, 4, 2).contiguous()  # prediction

        if self.training:
            return p

        else:  # inference
            io = p.clone()  # inference output
            io[..., :2] = torch.sigmoid(io[..., :2]) + self.grid_xy  # xy
            io[..., 2:4] = torch.exp(io[..., 2:4]) * self.anchor_wh  # wh yolo method
            io[..., :4] *= self.stride

            if 'default' in self.arc:  # seperate obj and cls
                torch.sigmoid_(io[..., 4:])

            # reshape from [1, 3, 13, 13, 85] to [1, 507, 85]
            return io.view(bs, -1, self.no), p

class Darknet(nn.Module):
    # YOLOv3 object detection model

    def __init__(self, cfg, img_size=(416, 416), arc='default'):
        super(Darknet, self).__init__()
        self.module_defs = parse_model_cfg(cfg)
        self.module_list, self.routs = create_modules(self.module_defs, img_size, arc)
        self.yolo_layers = get_yolo_layers(self)

    def forward(self, x, var=None):
        img_size = x.shape[-2:]
        output, layer_outputs = [], []

        for i, (mdef, module) in enumerate(zip(self.module_defs, self.module_list)):
            mtype = mdef['type']
            if mtype in ['convolutional', 'upsample', 'maxpool']:
                x = module(x)
            elif mtype == 'route':
                layers = [int(x) for x in mdef['layers'].split(',')]
                if len(layers) == 1:
                    x = layer_outputs[layers[0]]
                else:
                    try:
                        x = torch.cat([layer_outputs[i] for i in layers], 1)
                    except:  # apply stride 2 for darknet reorg layer
                        layer_outputs[layers[1]] = F.interpolate(layer_outputs[layers[1]], scale_factor=[0.5, 0.5])
                        x = torch.cat([layer_outputs[i] for i in layers], 1)
            elif mtype == 'shortcut':
                j = int(mdef['from'])
                x = x + layer_outputs[j]
            elif mtype == 'yolo':
                output.append(module(x, img_size))
            layer_outputs.append(x if i in self.routs else [])

        if self.training:
            return output
        else:
            io, p = zip(*output)  # inference output, training output
            return torch.cat(io, 1), p

def get_yolo_layers(model):
    return [i for i, x in enumerate(model.module_defs) if x['type'] == 'yolo']  # [82, 94, 106] for yolov3


def create_grids(self, img_size=416, ng=(13, 13), device='cpu', type=torch.float32):
    nx, ny = ng  # x and y grid size
    self.img_size = max(img_size)
    self.stride = self.img_size / max(ng)                             #kind of scale factor which is used to scale down objects in oginal image to small images.
    yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])

    self.grid_xy = torch.stack((xv, yv), 2).to(device).type(type).view((1, 1, ny, nx, 2))   #y,x grid values kept in arrays of 2 channel depth.
    self.anchor_vec = self.anchors.to(device) / self.stride                                 #scale anchor boxes to new grid dimension.
    self.anchor_wh = self.anchor_vec.view(1, self.na, 1, 1, 2).to(device).type(type)

    self.ng = torch.Tensor(ng).to(device)
    self.nx = nx
    self.ny = ny