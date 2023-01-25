import argparse
import sys
import os
import yaml
import time
import shutil
import datetime

import torch

sys.path.append("../")
from pycode.config import _C as cfg
from pycode.dataset import RLBench_DMOEBM
from pycode.misc import str2bool, save_checkpoint, gaussian_noise, get_pos, visualize_multi_query_pos, Timer, Time_memo, convert_rotation_6d_to_matrix, save_args
from pycode.model.total_model import DMOEBM
from pycode.model.Motion_Gen import VAE_add_noise

from pycode.loss.Regression_loss import Iterative_Motion_Loss, Evaluation
##### parser #####
parser = argparse.ArgumentParser(description='parser for image generator')
parser.add_argument('--config_file', type=str, default='../config/RLBench_DMO.yaml', metavar='FILE', help='path to config file')
parser.add_argument('--name', type=str, default="")
parser.add_argument('--add_name', type=str, default="")
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--reset_dataset', type=str2bool, default=False)
parser.add_argument('--frame', type=int, default=100)
parser.add_argument('--rot_mode', type=str, default="6d")
parser.add_argument('--num_refine_per_iter', type=int, default=5)
parser.add_argument('--tasks', nargs="*", type=str, default=["none"]) # PutRubbishInBin StackWine CloseBox PushButton ReachTarget TakePlateOffColoredDishRack PutKnifeOnChoppingBoard StackBlocks
args = parser.parse_args()

##### config #####
# get cfg data
if len(args.config_file) > 0:
    print('Loaded configration file {}'.format(args.config_file))
    cfg.merge_from_file(args.config_file)

    with open(args.config_file) as file:
        obj = yaml.safe_load(file)

if args.tasks[0] != "none":
    task_list = args.tasks
else:
    task_list = [cfg.DATASET.RLBENCH.TASK_NAME]

base_yamlname = os.path.basename(args.config_file)
head, ext = os.path.splitext(args.config_file)
dt_now = datetime.datetime.now()
temp_yaml_path = f"{head}_{dt_now.year}{dt_now.month}{dt_now.day}_{dt_now.hour}:{dt_now.minute}:{dt_now.second}{ext}"

for task_name in task_list:
    cfg.DATASET.RLBENCH.TASK_NAME = task_name
    
    if cfg.NOISE.METHOD == "vae":
        if cfg.VAE.NAME == "Transformer_VAE":
            noise_name = f"Transformer_vae_{cfg.VAE.LATENT_DIM}"
        else:
            noise_name = f"vae_{cfg.VAE.LATENT_DIM}"
    else:
        noise_name = cfg.NOISE.METHOD

    if args.name == "":
        dir_name = f"DMO_iterative_{args.num_refine_per_iter}_frame_{args.frame}_mode_{args.rot_mode}_noise_{noise_name}"
    else:
        dir_name = args.name

    if args.add_name != "":
        dir_name = f"{dir_name}_{args.add_name}"

    device = args.device
    rot_mode = args.rot_mode
    max_iter = cfg.OUTPUT.MAX_ITER
    save_iter = cfg.OUTPUT.SAVE_ITER
    eval_iter = cfg.OUTPUT.EVAL_ITER
    log_iter = cfg.OUTPUT.LOG_ITER
    batch_size = cfg.DATASET.BATCH_SIZE

    save_dir = os.path.join(cfg.OUTPUT.BASE_DIR, cfg.DATASET.NAME, cfg.DATASET.RLBENCH.TASK_NAME)

    save_path = os.path.join(save_dir, dir_name)
    print(f"save path:{save_path}")
    if os.path.exists(save_path):
        while 1:
            ans = input('The specified output dir is already exists. Overwrite? y or n: ')
            if ans == 'y':
                break
            elif ans == 'n':
                raise ValueError("Please specify correct output dir")
            else:
                print('please type y or n')
    else:
        os.makedirs(save_path)

    if rot_mode == "quat":
        rot_dim = 4
    elif rot_mode == "6d":
        rot_dim = 6
    else:
        raise ValueError("TODO")

    model_save_dir = os.path.join(save_path, "model")
    log_dir = os.path.join(save_path, 'log')
    vis_dir = os.path.join(save_path, 'vis')
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    # copy source code
    shutil.copy(sys.argv[0], save_path)
    if (len(args.config_file) > 0):
        shutil.copy(temp_yaml_path, os.path.join(save_path, base_yamlname))

    # save args
    argsfile_path = os.path.join(save_path, "args.json")
    save_args(args,argsfile_path)

    # set dataset
    train_dataset  = RLBench_DMOEBM("train", cfg, save_dataset=False, num_frame=args.frame, rot_mode=rot_mode)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    val_dataset = RLBench_DMOEBM("val", cfg, save_dataset=False, num_frame=args.frame, rot_mode=rot_mode)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=100, shuffle=False, num_workers=8)

    # set model
    conv_dims = cfg.MODEL.CONV_DIMS
    enc_depths = cfg.MODEL.ENC_DEPTHS
    enc_layers = cfg.MODEL.ENC_LAYERS

    dec_depths = cfg.MODEL.DEC_DEPTHS
    dec_layers = cfg.MODEL.DEC_LAYERS

    extractor_name = cfg.MODEL.EXTRACTOR_NAME
    predictor_name = cfg.MODEL.PREDICTOR_NAME

    conv_droppath_rate = cfg.MODEL.CONV_DROP_PATH_RATE
    atten_dropout_rate = cfg.MODEL.ATTEN_DROPOUT_RATE
    query_emb_dim = cfg.MODEL.QUERY_EMB_DIM
    num_atten_block = cfg.MODEL.NUM_ATTEN_BLOCK

    model = DMOEBM(["uv","z","rotation","grasp_state","time"], [2,1,rot_dim,1,1], 
                        dims=conv_dims, enc_depths=enc_depths, enc_layers=enc_layers, dec_depths=dec_depths, dec_layers=dec_layers, 
                        extractor_name=extractor_name, predictor_name=predictor_name, num_attn_block=num_atten_block, query_emb_dim=query_emb_dim,
                        drop_path_rate=conv_droppath_rate, mlp_drop=atten_dropout_rate)
    # model = HIBC_Model(["uv","z","rotation","grasp_state","time"], [2,1,rot_dim,1,1], extractor_name="gap", predictor_name="Regressor_Transformer_with_img_and_pose_feature")
    model = model.to(device)

    # set optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.OPTIM.LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.OPTIM.SCHEDULER.STEP, gamma=cfg.OPTIM.SCHEDULER.GAMMA)
    Train_loss = Iterative_Motion_Loss('cuda',mode=rot_mode)
    Eval_loss = Evaluation('cuda',mode=rot_mode)

    # set noise sampler
    if cfg.NOISE.METHOD == "gaussian":
        pos_noise = cfg.NOISE.GAUSSIAN.POSE_RANGE
        rot_noise = cfg.NOISE.GAUSSIAN.ROT_RANGE
        grasp_noise = cfg.NOISE.GAUSSIAN.GRASP_PROB
        add_noise = lambda query: gaussian_noise(query, pos_noise, rot_noise, grasp_noise, train_dataset.info_dict["data_list"][0]["camera_intrinsic"], rot_mode=rot_mode)
    elif cfg.NOISE.METHOD == "vae":
        vae_name = cfg.VAE.NAME
        vae_pretrained_path = f"../global_result/RLBench/{cfg.DATASET.RLBENCH.TASK_NAME}/{vae_name}_frame_{args.frame}_latentdim_{cfg.VAE.LATENT_DIM}_mode_{rot_mode}/model/model_iter100000.pth"
        add_noise = VAE_add_noise(vae_pretrained_path, vae_name, train_dataset.info_dict["data_list"][0]["camera_intrinsic"], latent_dim=cfg.VAE.LATENT_DIM, rot_mode=rot_mode, frame=args.frame, device=device)

    ##### start training #####
    iteration = 0
    start = time.time()
    time_memo = Time_memo()
    flag = False
    for epoch in range(10000000):
        for data in train_dataloader:
            image, h_query = data
            noise_query = add_noise(h_query) 

            optimizer.zero_grad()
            for key in noise_query.keys():
                noise_query[key] = noise_query[key].to(device)
            image = image.to(device)
            time_query = noise_query["time"]

            pred_dict_list = []
            for k in range(args.num_refine_per_iter):
                with Timer() as t:
                    if k == 0:
                        pred_dict, info = model(image, noise_query)
                    else:
                        pred_dict, info = model(image, pred_dict, with_feature=True)
                time_memo.add("forward", t.secs)

                pred_dict["time"] = time_query
                pred_dict_list.append(pred_dict)

            for key in noise_query.keys():
                noise_query[key] = torch.squeeze(noise_query[key], 1)
                for pred_dict in pred_dict_list:
                    try:
                        pred_dict[key] = torch.squeeze(pred_dict[key], 1)
                    except:
                        pass

            loss, loss_dict = Train_loss(pred_dict_list, h_query)
            loss.backward()
            optimizer.step()
            scheduler.step()

            # log
            if iteration % log_iter == 0:
                end = time.time()
                cost = (end - start) / (iteration+1)
                print(f'Train Iter: {iteration} Cost: {cost:.4g} Loss: {loss_dict["train/loss"]:.4g} uv:{loss_dict["train/uv_loss"]:.4g}, z:{loss_dict["train/z_loss"]:.4g}, rot:{loss_dict["train/rot_loss"]:.4g}, grasp:{loss_dict["train/grasp_loss"]:.4g}')

            # evaluate model
            if iteration % eval_iter == 0:
                with torch.no_grad():
                    pred_dict = get_pos(pred_dict, train_dataset.info_dict["data_list"][0]["camera_intrinsic"])
                    if rot_mode == "6d":
                        h_query, noise_query, pred_dict = convert_rotation_6d_to_matrix([h_query, noise_query, pred_dict])
                    loss_dict = Eval_loss(pred_dict, h_query)
                    print(f'Train: {iteration} Pos:{loss_dict["train/pos_error"]:.4g}, z:{loss_dict["train/z_error"]:.4g}, rot:{loss_dict["train/rot_error"]:.4g}, grasp:{loss_dict["train/grasp_accuracy"]:.4g}')
                    
                    for key in h_query.keys():
                        h_query[key] = h_query[key].cpu()
                        noise_query[key] = noise_query[key].cpu()
                        try:
                            pred_dict[key] = pred_dict[key].cpu()
                        except:
                            pass

                    vis_img = visualize_multi_query_pos(image, [h_query, noise_query, pred_dict], train_dataset.info_dict["data_list"][0]["camera_intrinsic"], rot_mode=rot_mode)
                    vis_img.save(os.path.join(vis_dir, f"pos_img_train_{iteration}.png"))

                    for data in val_dataloader:
                        model.eval()
                        image, h_query = data
                        noise_query = add_noise(h_query) 

                        optimizer.zero_grad()
                        for key in noise_query.keys():
                            noise_query[key] = noise_query[key].to(device)
                        image = image.to(device)

                        pred_dict, info = model(image, noise_query)

                        for key in noise_query.keys():
                            noise_query[key] = torch.squeeze(noise_query[key], 1)
                            try:
                                pred_dict[key] = torch.squeeze(pred_dict[key], 1)
                            except:
                                pass

                        pred_dict = get_pos(pred_dict, train_dataset.info_dict["data_list"][0]["camera_intrinsic"])
                        if rot_mode == "6d":
                            h_query, noise_query, pred_dict = convert_rotation_6d_to_matrix([h_query, noise_query, pred_dict])
                        loss_dict = Eval_loss(pred_dict, h_query, mode="val")
                        print(f'Val: {iteration} Pos:{loss_dict["val/pos_error"]:.4g}, z:{loss_dict["val/z_error"]:.4g}, rot:{loss_dict["val/rot_error"]:.4g}, grasp:{loss_dict["val/grasp_accuracy"]:.4g}')
                        
                        for key in pred_dict.keys():
                            h_query[key] = h_query[key].cpu()
                            noise_query[key] = noise_query[key].cpu()
                            pred_dict[key] = pred_dict[key].cpu()

                        vis_img = visualize_multi_query_pos(image, [h_query, noise_query, pred_dict], train_dataset.info_dict["data_list"][0]["camera_intrinsic"], rot_mode=rot_mode)
                        vis_img.save(os.path.join(vis_dir, f"pos_img_val_{iteration}.png"))

                        model.train()

            # save model
            if iteration % save_iter == 0:
                model_save_path = os.path.join(model_save_dir, f"model_iter{str(iteration).zfill(5)}.pth")
                save_checkpoint(model, optimizer, epoch, iteration, model_save_path)

            if iteration == max_iter + 1:
                flag = True
                break
            
            iteration += 1
        
        if flag == True:
            break

os.remove(temp_yaml_path)