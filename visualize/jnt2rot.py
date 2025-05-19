import torch
import numpy as np
import os
import sys
import smplx
import h5py
from tqdm import tqdm
import argparse
import visualize.utils.rotation_conversions as geometry
from visualize.joints2smpl.src import config
from visualize.joints2smpl.src.smplify import SMPLify3D
from visualize.config import right_hand_pose, left_hand_pose

class joints2smpl:
    def __init__(self, num_frames, device_id, cuda=True):
        self.device = torch.device("cuda:" + str(device_id) if cuda else "cpu")
        # self.device = torch.device("cpu")
        self.batch_size = num_frames
        self.num_joints = 22  # for HumanML3D
        self.joint_category = "AMASS"
        self.num_smplify_iters = 150
        self.fix_foot = False
        
        smplmodel = smplx.create(config.SMPL_MODEL_DIR,
                                 model_type="smpl", gender="neutral", ext="pkl", flat_hand_mean=False, left_hand_pose=left_hand_pose, right_hand_pose=right_hand_pose,
                                 batch_size=self.batch_size).to(self.device)

        # ## --- load the mean pose as original ----
        smpl_mean_file = config.SMPL_MEAN_FILE

        file = h5py.File(smpl_mean_file, 'r')
        self.init_mean_pose = torch.from_numpy(file['pose'][:]).unsqueeze(0).repeat(self.batch_size, 1).float().to(self.device)
        self.init_mean_shape = torch.from_numpy(file['shape'][:]).unsqueeze(0).repeat(self.batch_size, 1).float().to(self.device)
        self.cam_trans_zero = torch.Tensor([0.0, 0.0, 0.0]).unsqueeze(0).to(self.device)
        #

        # # #-------------initialize SMPLify
        self.smplify = SMPLify3D(smplxmodel=smplmodel,
                            batch_size=self.batch_size,
                            joints_category=self.joint_category,
                            num_iters=self.num_smplify_iters,
                            device=self.device)

    def joint2smpl(self, input_joints, init_params=None):
        _smplify = self.smplify # if init_params is None else self.smplify_fast
        pred_pose = torch.zeros(self.batch_size, 72).to(self.device)
        pred_betas = torch.zeros(self.batch_size, 10).to(self.device)  # Always initialize with zeros
        pred_cam_t = torch.zeros(self.batch_size, 3).to(self.device)
        keypoints_3d = torch.zeros(self.batch_size, self.num_joints, 3).to(self.device)

        # run the whole seqs
        num_seqs = input_joints.shape[0]

        # joints3d = input_joints[idx]  # *1.2 #scale problem [check first]
        keypoints_3d = torch.Tensor(input_joints).to(self.device).float()

        # if idx == 0:
        if init_params is None:
            pred_pose = self.init_mean_pose
            pred_cam_t = self.cam_trans_zero
            # pred_betas remains zero
        else:
            pred_pose = init_params['pose']
            pred_cam_t = init_params['cam']
            # pred_betas remains zero

        if self.joint_category == "AMASS":
            confidence_input = torch.ones(self.num_joints)
            # make sure the foot and ankle
            if self.fix_foot == True:
                confidence_input[7] = 1.5
                confidence_input[8] = 1.5
                confidence_input[10] = 1.5
                confidence_input[11] = 1.5
            # make sure the hand and wrist
                confidence_input[20] = 1.5
                confidence_input[21] = 1.5
                confidence_input[22] = 1.5
                confidence_input[25] = 1.5
        else:
            print("Such category not settle down!")

        new_opt_vertices, new_opt_joints, new_opt_pose, new_opt_betas, \
        new_opt_cam_t, new_opt_joint_loss = _smplify(
            pred_pose.detach(),
            pred_betas.detach(),  # This will be zeros
            pred_cam_t.detach(),
            keypoints_3d,
            conf_3d=confidence_input.to(self.device),
            # seq_ind=idx
            seq_ind=1 # exclude betas from grad
        )

        thetas = new_opt_pose.reshape(self.batch_size, 24, 3)
        thetas = geometry.matrix_to_rotation_6d(geometry.axis_angle_to_matrix(thetas))  # [bs, 24, 6]
        root_loc = keypoints_3d[:, 0].clone().detach()  # [bs, 3]
        root_loc = torch.cat([root_loc, torch.zeros_like(root_loc)], dim=-1).unsqueeze(1)  # [bs, 1, 6]
        thetas = torch.cat([thetas, root_loc], dim=1).unsqueeze(0).permute(0, 2, 3, 1)  # [1, 25, 6, 196]
        
        return thetas.clone().detach(), {'pose': new_opt_joints[0, :24].flatten().clone().detach(), 'betas': new_opt_betas.clone().detach(), 'cam': new_opt_cam_t.clone().detach()}
