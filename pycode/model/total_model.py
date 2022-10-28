
import torch

from .resnet_module import Resnet_Like_Decoder, Resnet_Like_Encoder
from .predictor import Predictor
from .obs_encoder import obs_emb_model

class DMOEBM(torch.nn.Module):
    def __init__(self, query_list, query_dims, img_size=256, input_dim=4, dims=[96, 192, 384, 768], enc_depths=[3,3,9,3], enc_layers=['conv','conv','conv','conv'],
                 dec_depths=[3,3,3], dec_layers=['conv','conv','conv'], enc_act="gelu", enc_norm="layer", dec_act="gelu", dec_norm="layer", drop_path_rate=0.1,
                 extractor_name="query_uv_feature", predictor_name="HIBC_Transformer_with_cat_feature", num_attn_block=4,
                 mlp_drop=0.1, query_emb_dim=128):
        """
        Args:
        img_size (int): Size of image. We assume image is square.
        input_dim (int): Size of channel. 3 for RGB image.
        enc_depths (list[int]): Number of blocks at each stage for encoder.
        dec_depths (list[int]): Number of blocks at each stage for decoder.
        predictor_depth (int): Number of blocks at predicotr. This value is for IABC predictor.
        dims (list[int]): The channel size of each feature map.
        enc_layers (list[str]): Name of layer at each stage of encoder.
        dec_layers (list[str]): Name of layer at each stage of decoder.
        predictor (str): Name of predictor
        predictor_prob_func (str): Function to change the vector to probability. This function is for IABC.
        act (str): Activation function.
        norm (str): Normalization function.
        atten (str): Name of attention. If layer name is atten, indicated atten layer is used.
        drop_path (float): Stochastic depth rate for encoder and decoder. Default: 0.1
        """
        super().__init__()
        self.img_size = img_size
        if query_emb_dim == 0:
            emb_dim = dims[0] // 2
        else:
            emb_dim = query_emb_dim

        self.enc = Resnet_Like_Encoder(img_size, in_chans=input_dim, depths=enc_depths, dims=dims, layers=enc_layers, drop_path_rate=drop_path_rate, activation=enc_act, norm=enc_norm)
        self.dec = Resnet_Like_Decoder(img_size, depths=dec_depths, enc_dims=dims, layers=dec_layers, drop_path_rate=drop_path_rate, emb_dim=emb_dim, activation=dec_act, norm=dec_norm)
        self.predictor = Predictor(extractor_name, predictor_name, emb_dim, query_list, query_dims, drop=mlp_drop, img_emb_dim=emb_dim, num_attn_block=num_attn_block)
    
    def forward(self, img, query, with_feature=False):
        debug_info = {}
        if with_feature == False:
            img_feature = self.enc(img)
            img_feature = self.dec(img_feature)
            self.img_feature = img_feature
        else:
            img_feature = self.img_feature

        output_dict, pred_info = self.predictor(img_feature, query)
        for key in pred_info.keys():
            debug_info[key] = pred_info[key]
        return output_dict, debug_info

# class Resnet_EncDec_Predictor(torch.nn.Module):
#     def __init__(self, img_size, input_dim, query_list, query_dims, enc_depths=[3,3,9,3], dims=[96, 192, 384, 768], enc_layers=['conv','conv','conv','conv'],
#                  dec_depths=[3,3,3], dec_layers=['conv','conv','conv'],
#                  extractor_list=["query_uv_feature"], predictor_name="feature_IBC_sep",
#                  act='gelu', norm='layer', atten='axial', pos_emb='axial', feedforward='mlp', drop_path_rate=0.1, heads=4, obs_dim=48, mlp_drop=0.1,
#                  img_emb_dim=0, query_emb_dim=0, obs_emb_dim=0):
#         """
#         Args:
#         img_size (int): Size of image. We assume image is square.
#         input_dim (int): Size of channel. 3 for RGB image.
#         enc_depths (list[int]): Number of blocks at each stage for encoder.
#         dec_depths (list[int]): Number of blocks at each stage for decoder.
#         predictor_depth (int): Number of blocks at predicotr. This value is for IABC predictor.
#         dims (list[int]): The channel size of each feature map.
#         enc_layers (list[str]): Name of layer at each stage of encoder.
#         dec_layers (list[str]): Name of layer at each stage of decoder.
#         predictor (str): Name of predictor
#         predictor_prob_func (str): Function to change the vector to probability. This function is for IABC.
#         act (str): Activation function.
#         norm (str): Normalization function.
#         atten (str): Name of attention. If layer name is atten, indicated atten layer is used.
#         drop_path (float): Stochastic depth rate for encoder and decoder. Default: 0.1
#         """
#         super().__init__()
#         self.img_size = img_size
#         down_scale = 1
#         if img_emb_dim == 0:
#             img_emb_dim = dims[0] // 2
#         if query_emb_dim == 0:
#             query_emb_dim = dims[0] // 2
#         if obs_emb_dim == 0:
#             obs_emb_dim = dims[0] // 2

#         self.enc = Resnet_Like_Encoder(img_size, in_chans=input_dim, depths=enc_depths, dims=dims, layers=enc_layers, drop_path_rate=drop_path_rate, atten=atten, heads=heads)
#         self.dec = Resnet_Like_Decoder(img_size, depths=dec_depths, enc_dims=dims, layers=dec_layers, drop_path_rate=drop_path_rate, atten=atten, heads=heads, emb_dim=img_emb_dim)
#         self.predictor = predictor(extractor_list, predictor_name, (img_size, img_size), query_emb_dim, down_scale, query_list, query_dims, drop=mlp_drop, img_emb_dim=img_emb_dim, obs_emb_dim=obs_emb_dim)
        
#         self.obs_dim = obs_dim
#         if obs_dim != 0:
#             self.obs_encoder = obs_emb_model(obs_dim, obs_emb_dim)
    
#     def forward(self, img, query, obs, with_feature=False):
#         debug_info = {}
#         if with_feature == False:
#             img_feature = self.enc(img)
#             img_feature = self.dec(img_feature)
#             self.img_feature = img_feature
#         else:
#             img_feature = self.img_feature
            
#         if with_feature == False:
#             obs_emb = self.obs_encoder(obs)
#             self.obs_feature = obs_emb
#         else:
#             obs_emb = self.obs_feature

#         output_dict, pred_info = self.predictor(img_feature, query, obs_emb)
#         for key in pred_info.keys():
#             debug_info[key] = pred_info[key]
#         return output_dict, debug_info
    

# class Resnet_Enc_Predictor(torch.nn.Module):
#     def __init__(self, img_size, input_dim, query_list, query_dims, enc_depths=[3,3,9,3], dims=[96, 192, 384, 768], enc_layers=['conv','conv','conv','conv'],
#                  extractor_list=["query_uv_feature"], predictor_name="feature_IBC_sep", num_vec=0,
#                  act='gelu', norm='layer', atten='axial', pos_emb='axial', feedforward='mlp', drop_path_rate=0.1, heads=4):
#         """
#         Args:
#         img_size (int): Size of image. We assume image is square.
#         input_dim (int): Size of channel. 3 for RGB image.
#         enc_depths (list[int]): Number of blocks at each stage for encoder.
#         predictor_depth (int): Number of blocks at predicotr. This value is for IABC predictor.
#         dims (list[int]): The channel size of each feature map.
#         enc_layers (list[str]): Name of layer at each stage of encoder.
#         predictor (str): Name of predictor
#         predictor_prob_func (str): Function to change the vector to probability. This function is for IABC.
#         act (str): Activation function.
#         norm (str): Normalization function.
#         atten (str): Name of attention. If layer name is atten, indicated atten layer is used.
#         drop_path (float): Stochastic depth rate. Default: 0.0
#         """
#         super().__init__()
#         self.img_size = img_size        
#         self.enc = Resnet_Like_Encoder(img_size, in_chans=input_dim, depths=enc_depths, dims=dims, layers=enc_layers, drop_path_rate=drop_path_rate, atten=atten, heads=heads)
        
#         down_scale = 2 ** (len(enc_depths) + 1)
#         self.predictor = predictor(extractor_list, predictor_name, (img_size, img_size), dims[-1], down_scale, query_list, query_dims, num_vec=num_vec)
        
#     def forward(self, x, y=None, with_feature=False):
#         debug_info = {}
#         if with_feature == False:
#             x = self.enc(x)
#             x = x[-1]
#             self.feature = x
#         else:
#             x = self.feature
            
#         output_dict, pred_info = self.predictor(x, y)
#         for key in pred_info.keys():
#             debug_info[f"pred_{key}"] = pred_info[key]
#         return output_dict, debug_info
