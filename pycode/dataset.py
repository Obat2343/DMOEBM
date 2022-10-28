import os
import random
import pickle
import json
import time
import math

import torch
import torchvision
import pytorch3d
import pytorch3d.transforms

import numpy as np
import imgaug.augmenters as iaa
from tqdm import tqdm
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import RotationSpline
from scipy import interpolate

from .sampling import random_negative_sample_RLBench, random_relative_negative_sample_RLBench

class RLBench_DMOEBM(torch.utils.data.Dataset):
    """
    RLBench dataset for train IBC model
    
    Attributes
    ----------
    index_list: list[int]
        List of valid index of dictionaly.
    """
    
    def __init__(self, mode, cfg, save_dataset=False, debug=False, num_frame=100, rot_mode="quat"):

        # set dataset root
        if cfg.DATASET.NAME == "RLBench":
            data_root_dir = os.path.join(cfg.DATASET.RLBENCH.PATH, mode)
        else:
            raise ValueError("Invalid dataset name")

        self.cfg = cfg
        self.num_frame = num_frame
        self.rot_mode = rot_mode
        self.seed = 0
        self.info_dict = {}
        self.mode = mode
        random.seed(self.seed)

        task_names = cfg.DATASET.RLBENCH.TASK_NAME
        print(f"TASK: {task_names}")
        self._pickle_file_name = '{}_{}_{}_HIBC.pickle'.format(cfg.DATASET.NAME,mode,task_names)
        self._pickle_path = os.path.join(data_root_dir, 'pickle', self._pickle_file_name)
        if not os.path.exists(self._pickle_path) or save_dataset:
            # create dataset
            print('There is no pickle data')
            print('create pickle data')
            self.add_data(data_root_dir, cfg)
            self.preprocess()
            print('done')
            
            # save json data
            print('save pickle data')
            os.makedirs(os.path.join(data_root_dir, 'pickle'), exist_ok=True)
            with open(self._pickle_path, mode='wb') as f:
                pickle.dump(self.info_dict,f)
            print('done')
        else:
            # load json data
            print('load pickle data')
            with open(self._pickle_path, mode='rb') as f:
                self.info_dict = pickle.load(f)
            print('done')

        self.ToTensor = torchvision.transforms.ToTensor()

        self.debug = debug
        self.without_img = False
        self.img_aug = iaa.OneOf([
                        iaa.AdditiveGaussianNoise(scale=0.05*255),
                        iaa.JpegCompression(compression=(30, 70)),
                        iaa.WithBrightnessChannels(iaa.Add((-40, 40))),
                        iaa.AverageBlur(k=(2, 5)),
                        iaa.CoarseDropout(0.02, size_percent=0.5),
                        iaa.Identity()
                    ])

    def __len__(self):
        return len(self.info_dict["sequence_index_list"])

    def __getitem__(self, data_index):
        # get image
        # print('i:{}'.format(i))

        start_index, end_index = self.info_dict["sequence_index_list"][data_index]
        
        if self.without_img == False:
            rgb_path = os.path.join(self.info_dict["data_list"][start_index]['image_dir'], "front_rgb_00000000.png")
            rgb_image = Image.open(rgb_path)
            if self.mode == "train":
                rgb_image = np.array(rgb_image)
                rgb_image = Image.fromarray(self.img_aug(image=rgb_image))
            rgb_image = self.ToTensor(rgb_image)
        
            depth_path = os.path.join(self.info_dict["data_list"][start_index]['image_dir'], "front_depth_00000000.pickle")
            with open(depth_path, 'rb') as f:
                depth_image = pickle.load(f)
            depth_image = torch.unsqueeze(torch.tensor(np.array(depth_image), dtype=torch.float), 0)
            
            image = torch.cat([rgb_image, depth_image], 0)
        else:
            image = torch.zeros(1)
        
        sequence_index_list = [i / self.num_frame for i in range(self.num_frame + 1)]
        
        pos = self.info_dict["pos_curve_list"][data_index](sequence_index_list).transpose((1,0))
        
        if self.rot_mode == "quat":
            rot = self.info_dict["rot_curve_list"][data_index](sequence_index_list).as_quat()
            rot = torch.tensor(rot, dtype=torch.float)
        elif self.rot_mode == "euler":
            rot = self.info_dict["rot_curve_list"][data_index](sequence_index_list).as_euler('zxy', degrees=True)
            rot = torch.tensor(rot, dtype=torch.float)
        elif self.rot_mode == "matrix":
            rot = self.info_dict["rot_curve_list"][data_index](sequence_index_list).as_matrix()
            rot = torch.tensor(rot, dtype=torch.float)
        elif self.rot_mode == "6d":
            rot = self.info_dict["rot_curve_list"][data_index](sequence_index_list).as_matrix()
            rot = pytorch3d.transforms.matrix_to_rotation_6d(torch.tensor(rot, dtype=torch.float))
        else:
            raise ValueError("invalid mode for get_gripper")
        
        grasp = self.info_dict["grasp_state_curve_list"][data_index](sequence_index_list).transpose((1,0))
        uv = self.info_dict["uv_curve_list"][data_index](sequence_index_list).transpose((1,0))
        z = self.info_dict["z_curve_list"][data_index](sequence_index_list).transpose((1,0))
        
        action_dict = {}
        action_dict["pos"] = torch.tensor(pos, dtype=torch.float)
        action_dict["rotation"] = rot
        action_dict["grasp_state"] = torch.tensor(grasp, dtype=torch.float)
        action_dict["uv"] = torch.tensor(uv, dtype=torch.float)
        action_dict["z"] = torch.tensor(z, dtype=torch.float)
        action_dict["time"] = torch.unsqueeze(torch.tensor(sequence_index_list, dtype=torch.float), 1)

        return image, action_dict
    
    def add_data(self, folder_path, cfg):
        """
        output:
        data_list: list of data_dict
        data_dict = {
        'filename': str -> name of each data except file name extension. e.g. 00000
        'image_dir': str -> path to image dir which includes rgb and depth images
        'pickle_dir': str -> path to pickle dir.
        'end_index': index of data when task will finish
        'start_index': index of date when task started
        'gripper_state_change': index of gripper state is changed. The value is 0 when the gripper state is not changed
        }
        next_index_list: If we set self.next_len > 1, next frame is set to current_index + self.next_len. However, if the next frame index skip the grassping frame, it is modified to the grassping frame.
        index_list: this is used for index of __getitem__()
        sequence_index_list: this list contains lists of [start_index of sequennce, end_index of sequence]. so sequence_index_list[0] returns start and end frame index of 0-th sequence.
        """
        # for data preparation
        self.info_dict["data_list"] = []
        self.info_dict["index_list"] = []
        self.info_dict["sequence_index_list"] = []
        index = 0
        
        task_list = os.listdir(folder_path) # get task list
        task_list.sort() # sort task
        task_name = cfg.DATASET.RLBENCH.TASK_NAME

        print(f"taskname: {task_name}")
        task_path = os.path.join(folder_path, task_name)
        
        sequence_list = os.listdir(task_path)
        sequence_list.sort()
        
        for sequence_index in tqdm(sequence_list):
            start_index = index
            image_folder_path = os.path.join(task_path, sequence_index, 'image')
            pickle_folder_path = os.path.join(task_path, sequence_index, 'base_data')
            pickle_name_list = os.listdir(pickle_folder_path)
            pickle_name_list.sort()
            end_index = start_index + len(pickle_name_list) - 1

            past_gripper_open = 1.0 # default gripper state is open. If not, please change
            pickle_data_list = []
            for pickle_name in pickle_name_list:
                # gripper state check to keep grasping frame
                with open(os.path.join(pickle_folder_path, pickle_name), 'rb') as f:
                    pickle_data = pickle.load(f)
                    pickle_data_list.append(pickle_data)
                
                head, ext = os.path.splitext(pickle_name)
                data_dict = {}
                data_dict['image_dir'] = image_folder_path
                data_dict['filename'] = os.path.join(head)
                data_dict['pickle_path'] = pickle_folder_path
                data_dict['start_index'] = start_index
                data_dict['end_index'] = end_index
                self.info_dict["data_list"].append(data_dict)
                # get camera info
                camera_intrinsic = pickle_data['front_intrinsic_matrix']
                data_dict["camera_intrinsic"] = camera_intrinsic
                gripper_open = pickle_data['gripper_open']
                data_dict['gripper_state'] = gripper_open
                
            # image size
            if index == 0:
                rgb_path = os.path.join(data_dict['image_dir'], "front_rgb_{}.png".format(head))
                rgb_image = Image.open(rgb_path)
                image_size = rgb_image.size
            self.info_dict["image_size"] = image_size

            # get gripper info
            pose, rotation = self.get_gripper(pickle_data_list)

            # get uv cordinate and pose image
            uv = self.get_uv(pose, camera_intrinsic)
            uv = self.preprocess_uv(uv, image_size)
            
            for j in range(len(pose)):
                self.info_dict["data_list"][index]["pose"] = pose[j]
                self.info_dict["data_list"][index]["rotation"] = rotation[j]
                self.info_dict["data_list"][index]["uv"] = uv[j]
                self.info_dict["data_list"][index]['current_index'] = index
                index += 1
        
            self.info_dict["sequence_index_list"].append([start_index, end_index])
            
    def preprocess(self):
        print("start preprocess")
        self.info_dict["max_len"] = 0
        self.info_dict["pos_curve_list"] = []
        self.info_dict["rot_curve_list"] = []
        self.info_dict["grasp_state_curve_list"] = []
        self.info_dict["uv_curve_list"] = []
        self.info_dict["z_curve_list"] = []


        for i, (start_index, end_index) in enumerate(self.info_dict["sequence_index_list"]):
            print(f"{i}/{len(self.info_dict['sequence_index_list'])}")

            if self.info_dict["max_len"] < end_index - start_index + 1:
                self.info_dict["max_len"] = end_index - start_index + 1

            index_list = [index for index in range(start_index, end_index+1)]
            time_batch, pose_batch, rotation_batch, grasp_state_batch, uv_batch, z_batch = self.get_list(index_list, start_index, end_index)
            pos_curve, rot_curve, grasp_curve, uv_curve, z_curve = self.get_spline_curve(time_batch, pose_batch, rotation_batch, grasp_state_batch, uv_batch, z_batch)
            
            self.info_dict["pos_curve_list"].append(pos_curve)
            self.info_dict["rot_curve_list"].append(rot_curve)
            self.info_dict["grasp_state_curve_list"].append(grasp_curve)
            self.info_dict["uv_curve_list"].append(uv_curve)
            self.info_dict["z_curve_list"].append(z_curve)


    def preprocess_uv(self, uv, image_size):
        """
        Preprocess includes
        1. convert to torch.tensor
        2. convert none to 0.
        3. normalize uv from [0, image_size] to [-1, 1]
        """
        u, v = uv[:, 0], uv[:, 1]
        h, w = image_size
        u = (u / (w - 1) * 2) - 1
        v = (v / (h - 1) * 2) - 1
        uv = np.stack([u, v], 1)
        return uv
    
    def postprocess_uv(self, uv, image_size):
        """
        Preprocess includes
        1. denormalize uv from [-1, 1] to [0, image_size]
        """
        if uv.dim() == 2:
            u, v = uv[:, 0], uv[:, 1]
        elif uv.dim() == 1:
            u, v = uv[0], uv[1]
        
        h, w = image_size
        
        denorm_u = (u + 1) / 2 * (w - 1)
        denorm_v = (v + 1) / 2 * (h - 1)
        
        denorm_uv = torch.stack([denorm_u, denorm_v], dim=(uv.dim()-1))
        return denorm_uv
        
    def get_gripper(self, pickle_list):
        gripper_pos_WorldCor = np.array([np.append(pickle_data['gripper_pose'][:3], 1) for pickle_data in pickle_list])
        gripper_matrix_WorldCor = np.array([pickle_data['gripper_matrix'] for pickle_data in pickle_list])
        # gripper_open = torch.unsqueeze(gripper_open, 1)

        world2camera_matrix = np.array([pickle_data['front_extrinsic_matrix'] for pickle_data in pickle_list])
        camera2world_matrix = np.linalg.inv(world2camera_matrix)
        
        gripper_pose_CamCor = np.einsum('bij,bj->bi', camera2world_matrix, gripper_pos_WorldCor)
        gripper_matrix_CamCor = np.einsum('bij,bjk->bik', camera2world_matrix, gripper_matrix_WorldCor)
        gripper_rot_CamCor = R.from_matrix(gripper_matrix_CamCor[:,:3,:3])
            
        # return torch.tensor(gripper_pose_CamCor[:, :3], dtype=torch.float), torch.tensor(gripper_rot_CamCor, dtype=torch.float)
        return gripper_pose_CamCor[:, :3], gripper_rot_CamCor

    def update_seed(self):
        # change seed. augumentation will be changed.
        self.seed += 1
        random.seed(self.seed)
        
    @staticmethod
    def get_task_names(task_list):
        for i, task in enumerate(task_list):
            if i == 0:
                task_name = task
            else:
                task_name = task_name + "_" + task
        return task_name

    def get_uv(self, pos_data, intrinsic_matrix):
        # transfer position data(based on motive coordinate) to camera coordinate
        B, _ = pos_data.shape
        z = np.repeat(pos_data[:, 2], 3).reshape((B,3))
        pos_data = pos_data / z # u,v,1
        uv_result = np.einsum('ij,bj->bi', intrinsic_matrix, pos_data)
        return uv_result[:, :2]
    
    def get_list(self, index_list, start_index, end_index):
        pose_batch = []
        rotation_batch = []
        grasp_state_batch = []
        uv_batch = []
        z_batch = []
        time_batch = []

        # get pickle data
        start = time.time()
        for i,index in enumerate(index_list):
            data_dict = self.info_dict["data_list"][index]
            pose_batch.append(data_dict["pose"])
            uv_batch.append(data_dict["uv"])
            z_batch.append(data_dict["pose"][2:])
            grasp_state_batch.append([data_dict['gripper_state']])

            gripper_rot_CamCor = data_dict["rotation"]
            gripper_rot_CamCor = gripper_rot_CamCor.as_matrix()

            rotation_batch.append(gripper_rot_CamCor)

            normalized_time = (index - start_index) / (end_index - start_index)
            time_batch.append(normalized_time)
        
        time_batch = np.array(time_batch)
        pose_batch = np.array(pose_batch)
        rotation_batch = np.array(rotation_batch)
        grasp_state_batch = np.array(grasp_state_batch)
        uv_batch = np.array(uv_batch)
        z_batch = np.array(z_batch)
        return time_batch, pose_batch, rotation_batch, grasp_state_batch, uv_batch, z_batch
        
    def get_spline_curve(self, time_batch, pose_batch, rotation_batch, grasp_state_batch, uv_batch, z_batch):
        pose_batch = pose_batch.transpose((1,0))
        pos_curve = interpolate.interp1d(time_batch, pose_batch, kind="cubic", fill_value="extrapolate")
        # interpolated_pos = pos_curve(output_time).transpose((1,0))

        
        query_rot = R.from_matrix(rotation_batch)
        rot_curve = RotationSpline(time_batch, query_rot)
        # interpolated_rot = spline(output_time).as_matrix()

        grasp_state_batch = grasp_state_batch.transpose((1,0))
        grasp_curve = interpolate.interp1d(time_batch, grasp_state_batch, fill_value="extrapolate")

        
        uv_batch = uv_batch.transpose((1,0))
        uv_curve = interpolate.interp1d(time_batch, uv_batch, kind="cubic", fill_value="extrapolate")

        z_batch = z_batch.transpose((1,0))
        z_curve = interpolate.interp1d(time_batch, z_batch, kind="cubic", fill_value="extrapolate")
        
        return pos_curve, rot_curve, grasp_curve, uv_curve, z_curve
    
class RLBench_HIBC(torch.utils.data.Dataset):
    """
    RLBench dataset for train IBC model
    
    Attributes
    ----------
    index_list: list[int]
        List of valid index of dictionaly.
    """
    
    def __init__(self, mode, cfg, save_dataset=False, debug=False, num_frame_high=10, interpolate_frame=5, rot_mode="quat"):

        # set dataset root
        if cfg.DATASET.NAME == "RLBench":
            data_root_dir = os.path.join(cfg.DATASET.RLBENCH.PATH, mode)
        else:
            raise ValueError("Invalid dataset name")

        self.cfg = cfg
        self.num_frame_h = num_frame_high
        self.num_frame_l = num_frame_high * interpolate_frame
        self.rot_mode = rot_mode
        self.seed = 0
        self.info_dict = {}
        random.seed(self.seed)

        task_names = cfg.DATASET.RLBENCH.TASK_NAME
        print(f"TASK: {task_names}")
        self._pickle_file_name = '{}_{}_{}_HIBC.pickle'.format(cfg.DATASET.NAME,mode,task_names)
        self._pickle_path = os.path.join(data_root_dir, 'pickle', self._pickle_file_name)
        if not os.path.exists(self._pickle_path) or save_dataset:
            # create dataset
            print('There is no pickle data')
            print('create pickle data')
            self.add_data(data_root_dir, cfg)
            self.preprocess()
            print('done')
            
            # save json data
            print('save pickle data')
            os.makedirs(os.path.join(data_root_dir, 'pickle'), exist_ok=True)
            with open(self._pickle_path, mode='wb') as f:
                pickle.dump(self.info_dict,f)
            print('done')
        else:
            # load json data
            print('load pickle data')
            with open(self._pickle_path, mode='rb') as f:
                self.info_dict = pickle.load(f)
            print('done')

        self.ToTensor = torchvision.transforms.ToTensor()

        self.debug = debug
        self.without_img = False
    
    def __len__(self):
        return len(self.info_dict["sequence_index_list"])

    def __getitem__(self, data_index):
        # get image
        # print('i:{}'.format(i))

        start_index, end_index = self.info_dict["sequence_index_list"][data_index]
        
        if self.without_img == False:
            rgb_path = os.path.join(self.info_dict["data_list"][start_index]['image_dir'], "front_rgb_00000000.png")
            rgb_image = Image.open(rgb_path)
            rgb_image = self.ToTensor(rgb_image)
        
            depth_path = os.path.join(self.info_dict["data_list"][start_index]['image_dir'], "front_depth_00000000.pickle")
            with open(depth_path, 'rb') as f:
                depth_image = pickle.load(f)
            depth_image = torch.unsqueeze(torch.tensor(np.array(depth_image), dtype=torch.float), 0)
            
            image = torch.cat([rgb_image, depth_image], 0)
        else:
            image = torch.zeros(1)
        
        high_sequence_index_list = [i / self.num_frame_h for i in range(self.num_frame_h + 1)]
        low_sequence_index_list = [i / self.num_frame_l for i in range(self.num_frame_l + 1)]
        
        pos = self.info_dict["pos_curve_list"][data_index](high_sequence_index_list).transpose((1,0))
        
        if self.rot_mode == "quat":
            rot = self.info_dict["rot_curve_list"][data_index](high_sequence_index_list).as_quat()
            rot = torch.tensor(rot, dtype=torch.float)
        elif self.rot_mode == "euler":
            rot = self.info_dict["rot_curve_list"][data_index](high_sequence_index_list).as_euler('zxy', degrees=True)
            rot = torch.tensor(rot, dtype=torch.float)
        elif self.rot_mode == "matrix":
            rot = self.info_dict["rot_curve_list"][data_index](high_sequence_index_list).as_matrix()
            rot = torch.tensor(rot, dtype=torch.float)
        elif self.rot_mode == "6d":
            rot = self.info_dict["rot_curve_list"][data_index](high_sequence_index_list).as_matrix()
            rot = pytorch3d.transforms.matrix_to_rotation_6d(torch.tensor(rot, dtype=torch.float))
        else:
            raise ValueError("invalid mode for get_gripper")
        
        grasp = self.info_dict["grasp_state_curve_list"][data_index](high_sequence_index_list).transpose((1,0))
        uv = self.info_dict["uv_curve_list"][data_index](high_sequence_index_list).transpose((1,0))
        z = self.info_dict["z_curve_list"][data_index](high_sequence_index_list).transpose((1,0))
        
        high_action_dict = {}
        high_action_dict["pos"] = torch.tensor(pos, dtype=torch.float)
        high_action_dict["rotation"] = rot
        high_action_dict["grasp_state"] = torch.tensor(grasp, dtype=torch.float)
        high_action_dict["uv"] = torch.tensor(uv, dtype=torch.float)
        high_action_dict["z"] = torch.tensor(z, dtype=torch.float)
        high_action_dict["time"] = torch.unsqueeze(torch.tensor(high_sequence_index_list, dtype=torch.float), 1)
        
        low_action_dict = {}
        if self.num_frame_l != 0:
            pos_l = self.info_dict["pos_curve_list"][data_index](low_sequence_index_list).transpose((1,0))
        
            if self.rot_mode == "quat":
                rot_l = self.info_dict["rot_curve_list"][data_index](low_sequence_index_list).as_quat()
                rot_l = torch.tensor(rot_l, dtype=torch.float)
            elif self.rot_mode == "euler":
                rot_l = self.info_dict["rot_curve_list"][data_index](low_sequence_index_list).as_euler('zxy', degrees=True)
                rot_l = torch.tensor(rot_l, dtype=torch.float)
            elif self.rot_mode == "matrix":
                rot_l = self.info_dict["rot_curve_list"][data_index](low_sequence_index_list).as_matrix()
                rot_l = torch.tensor(rot_l, dtype=torch.float)
            elif self.rot_mode == "6d":
                rot_l = self.info_dict["rot_curve_list"][data_index](low_sequence_index_list).as_matrix()
                rot_l = pytorch3d.transforms.matrix_to_rotation_6d(torch.tensor(rot_l, dtype=torch.float))
            else:
                raise ValueError("invalid mode for get_gripper")

            grasp_l = self.info_dict["grasp_state_curve_list"][data_index](low_sequence_index_list).transpose((1,0))
            uv_l = self.info_dict["uv_curve_list"][data_index](low_sequence_index_list).transpose((1,0))
            z_l = self.info_dict["z_curve_list"][data_index](low_sequence_index_list).transpose((1,0))
            
            low_action_dict["pos"] = torch.tensor(pos_l, dtype=torch.float)
            low_action_dict["rotation"] = rot_l
            low_action_dict["grasp_state"] = torch.tensor(grasp_l, dtype=torch.float)
            low_action_dict["uv"] = torch.tensor(uv_l, dtype=torch.float)
            low_action_dict["z"] = torch.tensor(z_l, dtype=torch.float)
            low_action_dict["time"] = torch.unsqueeze(torch.tensor(low_sequence_index_list, dtype=torch.float), 1)
        
        return image, high_action_dict, low_action_dict
    
    def add_data(self, folder_path, cfg):
        """
        output:
        data_list: list of data_dict
        data_dict = {
        'filename': str -> name of each data except file name extension. e.g. 00000
        'image_dir': str -> path to image dir which includes rgb and depth images
        'pickle_dir': str -> path to pickle dir.
        'end_index': index of data when task will finish
        'start_index': index of date when task started
        'gripper_state_change': index of gripper state is changed. The value is 0 when the gripper state is not changed
        }
        next_index_list: If we set self.next_len > 1, next frame is set to current_index + self.next_len. However, if the next frame index skip the grassping frame, it is modified to the grassping frame.
        index_list: this is used for index of __getitem__()
        sequence_index_list: this list contains lists of [start_index of sequennce, end_index of sequence]. so sequence_index_list[0] returns start and end frame index of 0-th sequence.
        """
        # for data preparation
        self.info_dict["data_list"] = []
        self.info_dict["index_list"] = []
        self.info_dict["sequence_index_list"] = []
        index = 0
        
        task_list = os.listdir(folder_path) # get task list
        task_list.sort() # sort task
        task_name = cfg.DATASET.RLBENCH.TASK_NAME

        print(f"taskname: {task_name}")
        task_path = os.path.join(folder_path, task_name)
        
        sequence_list = os.listdir(task_path)
        sequence_list.sort()
        
        for sequence_index in tqdm(sequence_list):
            start_index = index
            image_folder_path = os.path.join(task_path, sequence_index, 'image')
            pickle_folder_path = os.path.join(task_path, sequence_index, 'base_data')
            pickle_name_list = os.listdir(pickle_folder_path)
            pickle_name_list.sort()
            end_index = start_index + len(pickle_name_list) - 1

            past_gripper_open = 1.0 # default gripper state is open. If not, please change
            pickle_data_list = []
            for pickle_name in pickle_name_list:
                # gripper state check to keep grasping frame
                with open(os.path.join(pickle_folder_path, pickle_name), 'rb') as f:
                    pickle_data = pickle.load(f)
                    pickle_data_list.append(pickle_data)
                
                head, ext = os.path.splitext(pickle_name)
                data_dict = {}
                data_dict['image_dir'] = image_folder_path
                data_dict['filename'] = os.path.join(head)
                data_dict['pickle_path'] = pickle_folder_path
                data_dict['start_index'] = start_index
                data_dict['end_index'] = end_index
                self.info_dict["data_list"].append(data_dict)
                # get camera info
                camera_intrinsic = pickle_data['front_intrinsic_matrix']
                data_dict["camera_intrinsic"] = camera_intrinsic
                gripper_open = pickle_data['gripper_open']
                data_dict['gripper_state'] = gripper_open
                
            # image size
            if index == 0:
                rgb_path = os.path.join(data_dict['image_dir'], "front_rgb_{}.png".format(head))
                rgb_image = Image.open(rgb_path)
                image_size = rgb_image.size
            self.info_dict["image_size"] = image_size

            # get gripper info
            pose, rotation = self.get_gripper(pickle_data_list)

            # get uv cordinate and pose image
            uv = self.get_uv(pose, camera_intrinsic)
            uv = self.preprocess_uv(uv, image_size)
            
            for j in range(len(pose)):
                self.info_dict["data_list"][index]["pose"] = pose[j]
                self.info_dict["data_list"][index]["rotation"] = rotation[j]
                self.info_dict["data_list"][index]["uv"] = uv[j]
                self.info_dict["data_list"][index]['current_index'] = index
                index += 1
        
            self.info_dict["sequence_index_list"].append([start_index, end_index])
            
    def preprocess(self):
        print("start preprocess")
        self.info_dict["max_len"] = 0
        self.info_dict["pos_curve_list"] = []
        self.info_dict["rot_curve_list"] = []
        self.info_dict["grasp_state_curve_list"] = []
        self.info_dict["uv_curve_list"] = []
        self.info_dict["z_curve_list"] = []


        for i, (start_index, end_index) in enumerate(self.info_dict["sequence_index_list"]):
            print(f"{i}/{len(self.info_dict['sequence_index_list'])}")

            if self.info_dict["max_len"] < end_index - start_index + 1:
                self.info_dict["max_len"] = end_index - start_index + 1

            index_list = [index for index in range(start_index, end_index+1)]
            time_batch, pose_batch, rotation_batch, grasp_state_batch, uv_batch, z_batch = self.get_list(index_list, start_index, end_index)
            pos_curve, rot_curve, grasp_curve, uv_curve, z_curve = self.get_spline_curve(time_batch, pose_batch, rotation_batch, grasp_state_batch, uv_batch, z_batch)
            
            self.info_dict["pos_curve_list"].append(pos_curve)
            self.info_dict["rot_curve_list"].append(rot_curve)
            self.info_dict["grasp_state_curve_list"].append(grasp_curve)
            self.info_dict["uv_curve_list"].append(uv_curve)
            self.info_dict["z_curve_list"].append(z_curve)


    def preprocess_uv(self, uv, image_size):
        """
        Preprocess includes
        1. convert to torch.tensor
        2. convert none to 0.
        3. normalize uv from [0, image_size] to [-1, 1]
        """
        u, v = uv[:, 0], uv[:, 1]
        h, w = image_size
        u = (u / (w - 1) * 2) - 1
        v = (v / (h - 1) * 2) - 1
        uv = np.stack([u, v], 1)
        return uv
    
    def postprocess_uv(self, uv, image_size):
        """
        Preprocess includes
        1. denormalize uv from [-1, 1] to [0, image_size]
        """
        if uv.dim() == 2:
            u, v = uv[:, 0], uv[:, 1]
        elif uv.dim() == 1:
            u, v = uv[0], uv[1]
        
        h, w = image_size
        
        denorm_u = (u + 1) / 2 * (w - 1)
        denorm_v = (v + 1) / 2 * (h - 1)
        
        denorm_uv = torch.stack([denorm_u, denorm_v], dim=(uv.dim()-1))
        return denorm_uv
        
    def get_gripper(self, pickle_list):
        gripper_pos_WorldCor = np.array([np.append(pickle_data['gripper_pose'][:3], 1) for pickle_data in pickle_list])
        gripper_matrix_WorldCor = np.array([pickle_data['gripper_matrix'] for pickle_data in pickle_list])
        # gripper_open = torch.unsqueeze(gripper_open, 1)

        world2camera_matrix = np.array([pickle_data['front_extrinsic_matrix'] for pickle_data in pickle_list])
        camera2world_matrix = np.linalg.inv(world2camera_matrix)
        
        gripper_pose_CamCor = np.einsum('bij,bj->bi', camera2world_matrix, gripper_pos_WorldCor)
        gripper_matrix_CamCor = np.einsum('bij,bjk->bik', camera2world_matrix, gripper_matrix_WorldCor)
        gripper_rot_CamCor = R.from_matrix(gripper_matrix_CamCor[:,:3,:3])
            
        # return torch.tensor(gripper_pose_CamCor[:, :3], dtype=torch.float), torch.tensor(gripper_rot_CamCor, dtype=torch.float)
        return gripper_pose_CamCor[:, :3], gripper_rot_CamCor

    def update_seed(self):
        # change seed. augumentation will be changed.
        self.seed += 1
        random.seed(self.seed)
        
    @staticmethod
    def get_task_names(task_list):
        for i, task in enumerate(task_list):
            if i == 0:
                task_name = task
            else:
                task_name = task_name + "_" + task
        return task_name

    def get_uv(self, pos_data, intrinsic_matrix):
        # transfer position data(based on motive coordinate) to camera coordinate
        B, _ = pos_data.shape
        z = np.repeat(pos_data[:, 2], 3).reshape((B,3))
        pos_data = pos_data / z # u,v,1
        uv_result = np.einsum('ij,bj->bi', intrinsic_matrix, pos_data)
        return uv_result[:, :2]
    
    def get_list(self, index_list, start_index, end_index):
        pose_batch = []
        rotation_batch = []
        grasp_state_batch = []
        uv_batch = []
        z_batch = []
        time_batch = []

        # get pickle data
        start = time.time()
        for i,index in enumerate(index_list):
            data_dict = self.info_dict["data_list"][index]
            pose_batch.append(data_dict["pose"])
            uv_batch.append(data_dict["uv"])
            z_batch.append(data_dict["pose"][2:])
            grasp_state_batch.append([data_dict['gripper_state']])

            gripper_rot_CamCor = data_dict["rotation"]
            gripper_rot_CamCor = gripper_rot_CamCor.as_matrix()

            rotation_batch.append(gripper_rot_CamCor)

            normalized_time = (index - start_index) / (end_index - start_index)
            time_batch.append(normalized_time)
        
        time_batch = np.array(time_batch)
        pose_batch = np.array(pose_batch)
        rotation_batch = np.array(rotation_batch)
        grasp_state_batch = np.array(grasp_state_batch)
        uv_batch = np.array(uv_batch)
        z_batch = np.array(z_batch)
        return time_batch, pose_batch, rotation_batch, grasp_state_batch, uv_batch, z_batch
        
    def get_spline_curve(self, time_batch, pose_batch, rotation_batch, grasp_state_batch, uv_batch, z_batch):
        pose_batch = pose_batch.transpose((1,0))
        pos_curve = interpolate.interp1d(time_batch, pose_batch, kind="cubic", fill_value="extrapolate")
        # interpolated_pos = pos_curve(output_time).transpose((1,0))

        
        query_rot = R.from_matrix(rotation_batch)
        rot_curve = RotationSpline(time_batch, query_rot)
        # interpolated_rot = spline(output_time).as_matrix()

        grasp_state_batch = grasp_state_batch.transpose((1,0))
        grasp_curve = interpolate.interp1d(time_batch, grasp_state_batch, fill_value="extrapolate")

        
        uv_batch = uv_batch.transpose((1,0))
        uv_curve = interpolate.interp1d(time_batch, uv_batch, kind="cubic", fill_value="extrapolate")

        z_batch = z_batch.transpose((1,0))
        z_curve = interpolate.interp1d(time_batch, z_batch, kind="cubic", fill_value="extrapolate")
        
        return pos_curve, rot_curve, grasp_curve, uv_curve, z_curve