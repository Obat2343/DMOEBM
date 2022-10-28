import torch
from einops import rearrange, repeat
from .model.base_module import LinearBlock
from .model.tools import compute_rotation_matrix_from_ortho6d as compute_rm

class Pose_Transformer(torch.nn.Module):
    
    def __init__(self, input_dim, output_dict, emb_dim, attn_block=4, grasp_activation="linear"):
        super(Pose_Transformer, self).__init__()
        
        self.emb_dim = emb_dim
        self.num_attn_block = attn_block
        
        self.input_emb_block = torch.nn.Sequential(
                            LinearBlock(input_dim, emb_dim),
                            LinearBlock(emb_dim, emb_dim*2),
                            LinearBlock(emb_dim*2, emb_dim*2),
                            LinearBlock(emb_dim*2, emb_dim))
        
        self.output_emb_block = torch.nn.Sequential(
                            LinearBlock(1, emb_dim),
                            LinearBlock(emb_dim, emb_dim*2),
                            LinearBlock(emb_dim*2, emb_dim*2),
                            LinearBlock(emb_dim*2, emb_dim))
        
        input_qkv_list = []
        output_qkv_list = []
        attn_list = []
        for _ in range(self.num_attn_block):
            input_qkv_list.append(LinearBlock(emb_dim, emb_dim*3))
            output_qkv_list.append(LinearBlock(emb_dim, emb_dim*3))
            attn_list.append(torch.nn.MultiheadAttention(emb_dim, 1, batch_first=True))
        
        self.input_qkv_modules = torch.nn.ModuleList(input_qkv_list)
        self.output_qkv_modules = torch.nn.ModuleList(output_qkv_list)
        self.attn_modules = torch.nn.ModuleList(attn_list)
        
        module_dict = {}
        for key in output_dict:
            module_dict[key] = torch.nn.Sequential(
                            LinearBlock(emb_dim, emb_dim),
                            LinearBlock(emb_dim, output_dict[key], activation='none'))
        
        self.output_module_dict = torch.nn.ModuleDict(module_dict)
        
        if grasp_activation == "sigmoid":
            self.grasp_act = torch.nn.Sigmoid()
        elif grasp_activation == "linear":
            self.grasp_act = lambda input, min=0., max=1.: torch.clamp(input, min=min, max=max)

        
    def forward(self, input_dict, output_dict):
        
        input_pos = input_dict["pos"]
        input_rot = input_dict["rotation"][:,:,:2,:]
        input_grasp = input_dict["grasp_state"]
        input_time = input_dict["time"]
        input_mask = input_dict["mask"]
        B, S = input_mask.shape
        
        output_time = output_dict["time"]
        output_mask = output_dict["mask"]
        B, S_out = output_mask.shape
        
        mask = torch.cat([input_mask, output_mask], 1)
        attn_mask = torch.einsum('bi,bj->bij',mask,mask)
        attn_mask = torch.where(attn_mask == 0, True, False)
        
        input_vec = torch.cat([input_pos.view(B,S,-1),input_rot.view(B,S,-1),input_grasp.view(B,S,-1),input_time.view(B,S,-1)], 2)
        input_emb = self.input_emb_block(input_vec)
        output_emb = self.output_emb_block(output_time)
        
        for input_qkv_module, output_qkv_module, attn_module in zip(self.input_qkv_modules, self.output_qkv_modules, self.attn_modules):
            
            input_qkv = input_qkv_module(input_emb)
            input_q, input_k, input_v = input_qkv[:,:,:self.emb_dim], input_qkv[:,:,self.emb_dim:2*self.emb_dim], input_qkv[:,:,self.emb_dim*2:]

            output_qkv = output_qkv_module(output_emb)
            output_q, output_k, output_v = output_qkv[:,:,:self.emb_dim], output_qkv[:,:,self.emb_dim:2*self.emb_dim], output_qkv[:,:,self.emb_dim*2:]

            q,k,v = torch.cat([input_q, output_q], 1), torch.cat([input_k, output_k], 1), torch.cat([input_v, output_v], 1)
            attn_emb, attn_weights = attn_module(q,k,v,attn_mask=attn_mask)
            attn_emb = torch.nan_to_num(attn_emb, nan=0.0)
        
            input_emb, output_emb = attn_emb[:,:S], attn_emb[:,S:]
            
        pred_dict = {}
        for key in self.output_module_dict.keys():
            pred_dict[key] = self.output_module_dict[key](output_emb)
            if key == "grasp_state":
                pred_dict["grasp_state"] = self.grasp_act(pred_dict["grasp_state"])
        
        return pred_dict

class Denoise_Transformer(torch.nn.Module):
    
    def __init__(self, input_dim, output_dict, emb_dim=128, attn_block=4, head=4, grasp_activation="linear"):
        super(Denoise_Transformer, self).__init__()
        
        self.emb_dim = emb_dim
        self.head = head
        self.num_attn_block = attn_block
        
        self.input_emb_block = torch.nn.Sequential(
                            LinearBlock(input_dim, emb_dim),
                            LinearBlock(emb_dim, emb_dim*2),
                            LinearBlock(emb_dim*2, emb_dim*2),
                            LinearBlock(emb_dim*2, emb_dim))
        
        qkv_list = []
        attn_list = []
        ff_list = []
        for _ in range(self.num_attn_block):
            qkv_list.append(LinearBlock(emb_dim, emb_dim*3))
            attn_list.append(torch.nn.MultiheadAttention(emb_dim, head, batch_first=True))
            ff_list.append(torch.nn.Sequential(
                            LinearBlock(emb_dim, emb_dim),
                            LinearBlock(emb_dim, emb_dim)))
        
        self.qkv_modules = torch.nn.ModuleList(qkv_list)
        self.attn_modules = torch.nn.ModuleList(attn_list)
        self.ff_modules = torch.nn.ModuleList(ff_list)
        
        module_dict = {}
        for key in output_dict:
            module_dict[key] = torch.nn.Sequential(
                            LinearBlock(emb_dim, emb_dim),
                            LinearBlock(emb_dim, output_dict[key]))
        
        self.output_module_dict = torch.nn.ModuleDict(module_dict)
        if grasp_activation == "sigmoid":
            self.grasp_act = torch.nn.Sigmoid()
        elif grasp_activation == "linear":
            self.grasp_act = lambda input, min=0., max=1.: torch.clamp(input, min=min, max=max)

        
    def forward(self, input_dict):
        
        input_pos = input_dict["pos"]
        input_rot = input_dict["rotation"]
        input_grasp = input_dict["grasp_state"]
        input_time = input_dict["time"]
        mask = input_dict["mask"]
        B, S = mask.shape
        
        attn_mask = torch.where(mask == 0, float('-inf'), 0.)
        
        input_vec = torch.cat([input_pos.view(B,S,-1),input_rot.view(B,S,-1),input_grasp.view(B,S,-1),input_time.view(B,S,-1)], 2)
        emb_vec = self.input_emb_block(input_vec)
        
        for qkv_module, attn_module, ff_module in zip(self.qkv_modules, self.attn_modules, self.ff_modules):
            
            qkv = qkv_module(emb_vec)
            q, k, v = qkv[:,:,:self.emb_dim], qkv[:,:,self.emb_dim:2*self.emb_dim], qkv[:,:,self.emb_dim*2:]

            attn_emb, attn_weights = attn_module(q,k,v,key_padding_mask=attn_mask)
            attn_emb = torch.nan_to_num(attn_emb, nan=0.0)
        
            emb_vec = emb_vec + attn_emb
            emb_vec = emb_vec + ff_module(emb_vec)
            
        pred_dict = {}
        for key in self.output_module_dict.keys():
            pred_dict[key] = input_dict[key] + self.output_module_dict[key](emb_vec)
            if key == "grasp_state":
                pred_dict["grasp_state"] = self.grasp_act(pred_dict["grasp_state"])
        
        return pred_dict

class Rotation_Loss(torch.nn.Module):
    
    def __init__(self, device='cuda'):
        super(Rotation_Loss, self).__init__()
        self.MSE = torch.nn.MSELoss(reduction='none')
        self.device = device
        
    def forward(self, pred_dict, gt_dict):
        pred_rot = pred_dict["rotation"].to(self.device)
        B, N, D = pred_rot.shape
        pred_rot = rearrange(pred_rot, 'B D N -> (B D) N')
        pred_rot = compute_rm(pred_rot)
        pred_rot = rearrange(pred_rot, '(B D) N M -> B D N M', B=B)
        gt_rot = gt_dict["rotation"].to(self.device)
        mask = gt_dict["mask"].to(self.device)
        
        loss = self.MSE(pred_rot, gt_rot)
        loss = torch.mean(loss.view(B,N,-1), 2) * mask
        loss = torch.sum(loss, 1)
        length = torch.sum(mask, 1)
        loss = torch.mean(loss / length)
        return loss 

class Position_Loss(torch.nn.Module):
    
    def __init__(self, device='cuda'):
        super(Position_Loss, self).__init__()
        self.MSE = torch.nn.MSELoss(reduction='none')
        self.device = device
        
    def forward(self, pred_dict, gt_dict):
        pred_pos = pred_dict["pos"].to(self.device)
        B, N, D = pred_pos.shape
        gt_pos = gt_dict["pos"].to(self.device)
        mask = gt_dict["mask"].to(self.device)
        
        loss = self.MSE(pred_pos, gt_pos)
        loss = torch.mean(loss, 2) * mask
        loss = torch.sum(loss, 1)
        length = torch.sum(mask, 1)
        loss = torch.mean(loss / length)
        return loss 

class Grasp_Loss(torch.nn.Module):
    
    def __init__(self, device='cuda'):
        super(Grasp_Loss, self).__init__()
        self.BCE = torch.nn.MSELoss(reduction="none")
        self.device = device
        
    def forward(self, pred_dict, gt_dict):
        pred_grasp = pred_dict["grasp_state"].to(self.device)
        B, N, D = pred_grasp.shape
        gt_grasp = gt_dict["grasp_state"].to(self.device)
        mask = gt_dict["mask"].to(self.device)
        
        loss = self.BCE(pred_grasp, gt_grasp)
        loss = torch.sum(loss, 2) * mask
        loss = torch.sum(loss, 1)
        length = torch.sum(mask, 1)
        loss = torch.mean(loss / length)
        return loss 

class Motion_Loss(torch.nn.Module):
    
    def __init__(self, device='cuda'):
        super(Motion_Loss, self).__init__()
        self.device = device
        self.rot_loss = Rotation_Loss(device)
        self.pos_loss = Position_Loss(device)
        self.grasp_loss = Grasp_Loss(device)
    
    def forward(self, pred_dict, gt_dict, mode="train"):
        loss_dict = {}
        
        rot_loss = self.rot_loss(pred_dict, gt_dict)
        loss_dict[f"{mode}/rot_loss"] = rot_loss.item()
        
        pos_loss = self.pos_loss(pred_dict, gt_dict)
        loss_dict[f"{mode}/pos_loss"] = pos_loss.item()
        
        grasp_loss = self.grasp_loss(pred_dict, gt_dict)
        loss_dict[f"{mode}/grasp_loss"] = grasp_loss.item()
        
        loss = rot_loss + pos_loss + grasp_loss
        loss_dict[f"{mode}/loss"] = loss.item()
        
        return loss, loss_dict

class RMSE_Pose_Eval(torch.nn.Module):

    def __init__(self, device='cuda'):
        super(RMSE_Pose_Eval, self).__init__()
        self.MSE = torch.nn.MSELoss(reduction='none')
        self.device = device
    
    def forward(self, pred_dict, gt_dict):
        pred_pos = pred_dict["pos"].to(self.device)
        B, N, D = pred_pos.shape
        gt_pos = gt_dict["pos"].to(self.device)
        mask = gt_dict["mask"].to(self.device)
        
        pred_pos = pred_pos * 1000
        gt_pos = gt_pos * 1000

        loss = self.MSE(pred_pos, gt_pos)
        loss = torch.sqrt(torch.sum(loss, 2)) * mask
        loss = torch.sum(loss, 1)
        length = torch.sum(mask, 1)
        loss = torch.mean(loss / length)
        return loss

class RMSE_Rotation_Eval(torch.nn.Module):
    
    def __init__(self, device='cuda'):
        super(RMSE_Rotation_Eval, self).__init__()
        self.MSE = torch.nn.MSELoss(reduction='none')
        self.device = device
        
    def forward(self, pred_dict, gt_dict):
        pred_rot = pred_dict["rotation"].to(self.device)
        B, N, D = pred_rot.shape
        pred_rot = rearrange(pred_rot, 'B N D -> (B N) D')
        pred_rot = compute_rm(pred_rot)
        pred_rot = rearrange(pred_rot, '(B D) N M -> B D N M', B=B)
        gt_rot = gt_dict["rotation"].to(self.device)
        mask = gt_dict["mask"].to(self.device)
        
        loss = self.MSE(pred_rot, gt_rot)
        loss = torch.sqrt(torch.sum(loss.view(B,N,-1), 2)) * mask
        loss = torch.sum(loss, 1)
        length = torch.sum(mask, 1)
        loss = torch.mean(loss / length)
        return loss 

class Accuracy_Grasp_Eval(torch.nn.Module):
    
    def __init__(self, device='cuda'):
        super(Accuracy_Grasp_Eval, self).__init__()
        self.L1 = torch.nn.L1Loss(reduction='none')
        self.device = device
        
    def forward(self, pred_dict, gt_dict):
        pred_grasp = pred_dict["grasp_state"].to(self.device)
        B, N, D = pred_grasp.shape
        pred_grasp = (pred_grasp>0.5).float()

        gt_grasp = gt_dict["grasp_state"].to(self.device)
        mask = gt_dict["mask"].to(self.device)
        
        loss = self.L1(pred_grasp, gt_grasp)
        loss = torch.sum(loss, 2) * mask
        loss = torch.sum(loss, 1)
        length = torch.sum(mask, 1)
        loss = torch.mean(loss / length)
        return 1 - loss 

class Evaluation(torch.nn.Module):
    
    def __init__(self, device='cuda'):
        super(Evaluation, self).__init__()
        self.device = device
        self.rot_loss = RMSE_Rotation_Eval(device)
        self.pos_loss = RMSE_Pose_Eval(device)
        self.grasp_loss = Accuracy_Grasp_Eval(device)
    
    def forward(self, pred_dict, gt_dict, mode="train"):
        loss_dict = {}
        
        rot_loss = self.rot_loss(pred_dict, gt_dict)
        loss_dict[f"{mode}/rot_error"] = rot_loss.item()
        
        pos_loss = self.pos_loss(pred_dict, gt_dict)
        loss_dict[f"{mode}/pos_error"] = pos_loss.item()
        
        grasp_loss = self.grasp_loss(pred_dict, gt_dict)
        loss_dict[f"{mode}/grasp_accuracy"] = grasp_loss.item()
        
        return loss_dict