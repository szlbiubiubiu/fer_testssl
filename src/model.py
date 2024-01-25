import pickle
from resnet import *
from torch.autograd import Variable

from vit_face_withlandmark import ViT_face_landmark_patch8
import pdb
def load_part_checkpoint_landmark(path,model,pretrain_name=['stn','output'],freeze=True):
    # pdb.set_trace()
    pretrained_dict =  torch.load(path, map_location='cpu')
    model_dict = model.state_dict()

    # 1. filter out unnecessary keys
    # pretrained_dict=list(pretrained_dict.keys())
    back_remove=list(pretrained_dict.keys())
    for keys in back_remove:
        if 'dummy_orthogonal_classifier' in keys:
            # pdb.set_trace()
            continue
        pretrained_dict[keys.replace('module.','')]=pretrained_dict.pop(keys)

    # pdb.set_trace()
    # for name_space in pretrain_name:
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if pretrain_name[0] in k or pretrain_name[1] in k}
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if pretrain_name[0] in k or pretrain_name[1] in k}
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict) 
    # 3. load the new state dict
    model.load_state_dict(model_dict,strict=True)
    # model.encoder.output_layer.load_state_dict(pretrained_dict,strict=True)
    model_dict = model.state_dict()
    #freeze stn and output layer
    # pdb.set_trace()
    if freeze:
        for name, param in model.named_parameters():
            # if not param.requires_grad:
            if pretrain_name[0] in name or pretrain_name[1] in name:
                # pdb.set_trace()
                param.requires_grad = False


class Model_withvit(nn.Module):
    
    def __init__(self, args, pretrained=True, num_classes=7):
        super(Model_withvit, self).__init__()
        # resnet50 = ResNet(Bottleneck, [3, 4, 6, 3])
        HEAD_NAME='CosFace'
        num_patches=196  
        patch_size=8
        with_land=True
        # pdb.set_trace()
        BACKBONE=ViT_face_landmark_patch8(
                                loss_type = HEAD_NAME,
                                GPU_ID = None,
                                num_class = 10000,
                                num_patches=num_patches,
                                image_size=112,
                                patch_size=patch_size,#8
                                dim=768,#512
                                depth=12,#20
                                heads=11,#8
                                mlp_dim=2048,
                                dropout=0.1,
                                emb_dropout=0.1,
                                with_land=with_land
                            )
        # BACKBONE=ViT_face_landmark_patch8()
        # with open(args.resnet50_path, 'rb') as f:
        #     obj = f.read()
        # weights = {key: torch.from_numpy(arr) for key, arr in pickle.loads(obj, encoding='latin1').items()}
        # resnet50.load_state_dict(weights)
        model_dir=args.model_dir
        mobi_pretrain=args.mobi_pretrain
        best_model_dict = torch.load(model_dir,map_location=torch.device('cpu'))['teacher']#['teacher'],['model']
        #remove 'backbone' from dino
        back_remove=list(best_model_dict.keys())
        for keys in back_remove:
            if 'dummy_orthogonal_classifier' in keys:
                # pdb.set_trace()
                continue
            best_model_dict[keys.replace('encoder.','').replace('backbone.','').replace('module.','')]=best_model_dict.pop(keys)
            # best_model_dict[keys.replace('backbone.','')]=best_model_dict.pop(keys)
            # best_model_dict[keys.replace('module.','')]=best_model_dict.pop(keys)
        # for keys in back_remove:
        #     if 'dummy_orthogonal_classifier' in keys:
        #         # pdb.set_trace()
        #         continue
        #     # best_model_dict[keys.replace('encoder.','')]=best_model_dict.pop(keys)
        #     best_model_dict[keys.replace('backbone.','')]=best_model_dict.pop(keys)
        #     # best_model_dict[keys.replace('module.','')]=best_model_dict.pop(keys)
        # pdb.set_trace()
        BACKBONE.load_state_dict(best_model_dict,strict=False)
        #load landmark part
        if with_land:
            # load_part_checkpoint_landmark_fromdino(path=mobi_pretrain,model=BACKBONE,pretrain_name=['stn','random'],freeze=False)    
            load_part_checkpoint_landmark(path=mobi_pretrain,model=BACKBONE,pretrain_name=['stn','output'],freeze=False)
        
        self.features = nn.Sequential(*list(BACKBONE.children())[:-2])  
        self.features2 = nn.Sequential(*list(BACKBONE.children())[-2:-1])  
        self.fc = nn.Linear(768, 7)  
        
        
    def forward(self, x):        
        x = self.features(x)
        #### 1, 2048, 7, 7
        feature = self.features2(x)
        #### 1, 2048, 1, 1
        
        feature = feature.view(feature.size(0), -1)
        output = self.fc(feature)
        
        params = list(self.parameters())
        fc_weights = params[-2].data
        fc_weights = fc_weights.view(1, 7, 2048, 1, 1)
        fc_weights = Variable(fc_weights, requires_grad = False)

        # attention
        feat = x.unsqueeze(1) # N * 1 * C * H * W
        hm = feat * fc_weights
        hm = hm.sum(2) # N * self.num_labels * H * W

        return output, hm

class Model(nn.Module):
    
    def __init__(self, args, pretrained=True, num_classes=7):
        super(Model, self).__init__()
        resnet50 = ResNet(Bottleneck, [3, 4, 6, 3])
        with open(args.resnet50_path, 'rb') as f:
            obj = f.read()
        weights = {key: torch.from_numpy(arr) for key, arr in pickle.loads(obj, encoding='latin1').items()}
        resnet50.load_state_dict(weights)
        
        self.features = nn.Sequential(*list(resnet50.children())[:-2])  
        self.features2 = nn.Sequential(*list(resnet50.children())[-2:-1])  
        self.fc = nn.Linear(2048, 7)  
        
        
    def forward(self, x):        
        x = self.features(x)
        #### 1, 2048, 7, 7
        feature = self.features2(x)
        #### 1, 2048, 1, 1
        
        feature = feature.view(feature.size(0), -1)
        output = self.fc(feature)
        
        params = list(self.parameters())
        fc_weights = params[-2].data
        fc_weights = fc_weights.view(1, 7, 2048, 1, 1)
        fc_weights = Variable(fc_weights, requires_grad = False)

        # attention
        feat = x.unsqueeze(1) # N * 1 * C * H * W
        hm = feat * fc_weights
        hm = hm.sum(2) # N * self.num_labels * H * W

        return output, hm