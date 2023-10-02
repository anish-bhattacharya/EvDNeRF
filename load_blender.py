import os
import torch
import numpy as np
import imageio 
import json
import cv2
import glob
import pickle

trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()

def rodrigues_mat_to_rot(R):
  eps =1e-16
  trc = np.trace(R)
  trc2 = (trc - 1.)/ 2.
  #sinacostrc2 = np.sqrt(1 - trc2 * trc2)
  s = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
  if (1 - trc2 * trc2) >= eps:
    tHeta = np.arccos(trc2)
    tHetaf = tHeta / (2 * (np.sin(tHeta)))
  else:
    tHeta = np.real(np.arccos(trc2))
    tHetaf = 0.5 / (1 - tHeta / 6)
  omega = tHetaf * s
  return omega

def rodrigues_rot_to_mat(r):
  wx,wy,wz = r
  theta = np.sqrt(wx * wx + wy * wy + wz * wz)
  a = np.cos(theta)
  b = (1 - np.cos(theta)) / (theta*theta)
  c = np.sin(theta) / theta
  R = np.zeros([3,3])
  R[0, 0] = a + b * (wx * wx)
  R[0, 1] = b * wx * wy - c * wz
  R[0, 2] = b * wx * wz + c * wy
  R[1, 0] = b * wx * wy + c * wz
  R[1, 1] = a + b * (wy * wy)
  R[1, 2] = b * wy * wz - c * wx
  R[2, 0] = b * wx * wz - c * wy
  R[2, 1] = b * wz * wy + c * wx
  R[2, 2] = a + b * (wz * wz)
  return R

def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w

def load_events_multiview(datadir, split, load_evs=False, half_res=False, single_view=-1, pos_thresh=0.2, neg_thresh=0.2):

    metas = {}
    with open(os.path.join(datadir, f'transforms_{split}.json'), 'r') as fp:
        metas = json.load(fp)

    ims = []
    poses = []
    times = []

    num_views, num_timesteps = None, None

    # adjust conditionals for loading evs and evims
    if load_evs:
        try:
            evs_filename = glob.glob(os.path.join(datadir, f'data_{split}.*'))
            if len(evs_filename) > 1:
                print(f'WARNING: found multiple files matching {datadir}/data_{split}.* ; Either a .npy or .pkl should exist. Exiting.')
                exit()
            if '.npy' in evs_filename[0]:
                evs = np.load(os.path.join(datadir, f'data_{split}.npy'), allow_pickle=True)
            elif '.pkl' in evs_filename[0]:
                with open(os.path.join(datadir, f'data_{split}.pkl'), 'rb') as file:
                    evs = pickle.load(file)
            evs = evs.tolist()
            evs = [torch.from_numpy(ev) for ev in evs]
        except:
            evs = None

    # for val, don't load in evs
    else:
        evs = None

    # print(f'[inside DATALOADER] Loading evims for {split}...')
    try:
        evims = np.load(os.path.join(datadir, f'data_frames_{split}.npy'))
    except:
        evims = None
    # print(f'[inside DATALOADER] Done.')

    if evims is not None:
        num_views, num_timesteps, H, W = evims.shape
        if num_views*num_timesteps < len(metas['frames']):
            num_timesteps += 1
            padded_ev_ims = np.zeros((num_views, num_timesteps, H, W))
            for view_idx in range(num_views):
                padded_ev_ims[view_idx] = np.concatenate((evims[view_idx], np.zeros((1, H, W))), axis=0)
            evims = torch.from_numpy(padded_ev_ims.reshape(-1, evims.shape[2], evims.shape[3]))
        else:
            evims = torch.from_numpy(evims.reshape(-1, evims.shape[2], evims.shape[3]))

    # choose single-view from multi-view data
    if single_view > -1:
        myview = single_view
        evs = [evs[single_view]] if evs is not None else None
        evims = evims[myview*num_timesteps:(myview+1)*num_timesteps, ...] if evims is not None else None
        metas['frames'] = metas['frames'][myview*num_timesteps:(myview+1)*num_timesteps]

    frames = metas['frames']

    # load each frame's rgb image, timestep, and pose
    view_ctr = 0
    for t, frame in enumerate(frames):
        fname = os.path.join(datadir, frame['file_path'] + '.png')
        if 'real' in datadir:
            im = imageio.imread(fname)
            if len(im.shape) == 2:
                im = np.stack((im, im, im), axis=2)
        else:
            # cv2 grayscale conversion was used to make the events for sim data
            im = cv2.imread(fname, cv2.IMREAD_GRAYSCALE) # (H, W)
        ims.append(im)
        poses.append(np.array(frame['transform_matrix']))
        timestep = frame['time']
        times.append(timestep)
        if np.abs((timestep - 1.0)) < 1e-5:
            view_ctr += 1

    num_views = view_ctr if view_ctr > 0 else 1
    num_timesteps = int(len(frames) / num_views)

    if evims is None:
        H, W = imageio.imread(fname).shape[:2]

    times = torch.from_numpy(np.array(times).astype(np.float32))
    if 'real' in datadir:
        ims = torch.from_numpy((np.array(ims) / 255.).astype(np.float32))[...,:3].mean(-1)
    else:
        # preserve the cv2 grayscale conversion for sim data, it's already 1-dimensional
        ims = torch.from_numpy((np.array(ims) / 255.).astype(np.float32))
    poses = torch.from_numpy(np.array(poses).astype(np.float32))

    # extract focal length from metadata
    # NOTE that in keyframing.py simulated data generation, the default camera parameters are:
    # focal_length: float = 50,
    # sensor_width: float = 36,
    # while legacy code sets "camera_angle_x": 0.6911112070083618, which means focal length is calculated to be 355.55
    # the above parameters suggest that sim focal length (in pixel units) should be 50/36 * W = 50/36 * 256 = 355.55
    # ok these agree
    camera_angle_x = float(metas['camera_angle_x'])
    if 'focal' in metas and float(metas['focal']) > 0.0:
        focal = float(metas['focal'])
    else:
        focal = .5 * W / np.tan(.5 * camera_angle_x)

    if half_res:

        # print(f'[inside DATALOADER] Resizing images and events to half-res for {split}...')

        H = H//2
        W = W//2
        focal = focal/2.

        ims = ims.numpy()

        ims_half_res = np.zeros((ims.shape[0], H, W), dtype=np.float32)
        for i, img in enumerate(ims):
            ims_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        ims = torch.from_numpy(ims_half_res)

        if evs is not None:
            # custom resize is done by making thresh 1/4 the value in the training script
            for view_idx in range(len(evs)):
                evs[view_idx][:,1] = evs[view_idx][:,1] // 2
                evs[view_idx][:,2] = evs[view_idx][:,2] // 2

        if evims is not None:
            # custom resize
            evims_half_res = torch.zeros((evims.shape[0], H, W), dtype=torch.float32).to(evims.device)
            sums_half_res = evims[:, 0::2, 0::2] + evims[:, 0::2, 1::2] + evims[:, 1::2, 0::2] + evims[:, 1::2, 1::2]
            pos_evs = (sums_half_res // (4*pos_thresh)) * pos_thresh
            evims_half_res[sums_half_res >= 0] = pos_evs[sums_half_res >= 0].float()
            neg_evs = (sums_half_res // (4*neg_thresh)) * neg_thresh
            evims_half_res[sums_half_res < 0] = neg_evs[sums_half_res < 0].float()
            evims = evims_half_res

            # # automatic resize using interpolation
            # evims = evims.numpy()
            # evims_half_res = np.zeros((evims.shape[0], H, W), dtype=np.float32)
            # for i, evim in enumerate(evims):
            #     evims_half_res[i] = cv2.resize(evim, (W, H), interpolation=cv2.INTER_AREA)
            # evims = torch.from_numpy(evims_half_res)

        # print(f'[inside DATALOADER] Done.')

    target_maxabs_value = evims.abs().max() if evims is not None else None

    return {"images": ims,
            "evims": evims,
            "evs": evs,
            "poses": poses,
            "times": times,
            "intrinsics": [H, W, focal],
            "target_maxabs_value": target_maxabs_value,
            "num_views": num_views,
            "num_timesteps": num_timesteps}

def preload_data(data, device='cpu'):

    data['images'] = data['images'].to(device)
    data['evims'] = data['evims'].to(device) if data['evims'] is not None else None
    data['evs'] = [ev.to(device) for ev in data['evs']] if data['evs'] is not None else None
    data['poses'] = data['poses'].to(device)
    data['times'] = data['times'].to(device)
    data['target_maxabs_value'] = data['target_maxabs_value'].to(device) if data['target_maxabs_value'] is not None else None

    return data
