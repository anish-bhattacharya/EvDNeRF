# This script trains an EvDNeRF with various optional arguments, and periodically saves checkpoints and runs validations.
# For details, please refer to the paper EvDNeRF: Reconstructing Event Data with Dynamic Neural Radiance Fields.

# NOTE for consistency with previous NeRF codebases, we refer to images as "rgb" though we only consider intensity (1-channel) images.

import os, sys, glob
import imageio
import time
from numpy import empty
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
import json

from run_evdnerf_helpers import *
from load_blender import *

from metrics import *

# NOTE this suppresses tensorflow warnings and info
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEBUG = False

def set_seeds(seed):
    np.random.seed(seed)
    # random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    def ret(inputs_pos, inputs_time):
        num_batches = inputs_pos.shape[0]

        out_list = []
        dx_list = []
        for i in range(0, num_batches, chunk):
            out, dx = fn(inputs_pos[i:i+chunk], [inputs_time[0][i:i+chunk], inputs_time[1][i:i+chunk]])
            out_list += [out]
            dx_list += [dx]
        return torch.cat(out_list, 0), torch.cat(dx_list, 0)
    return ret

def run_network(inputs, viewdirs, frame_time, fn, embed_fn, embeddirs_fn, embedtime_fn, netchunk=1024*64,
                embd_time_discr=True):
    """Prepares inputs and applies network 'fn'.
    inputs: N_rays x N_points_per_ray x 3
    viewdirs: N_rays x 3
    frame_time: N_rays x 1
    """
    # assert len(torch.unique(frame_time)) == 1, "Only accepts all points from same time"
    cur_time = torch.unique(frame_time)[0]

    # embed position
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    # embed time
    if embd_time_discr:
        B, N, _ = inputs.shape
        input_frame_time = frame_time[:, None].expand([B, N, 1])
        input_frame_time_flat = torch.reshape(input_frame_time, [-1, 1])
        embedded_time = embedtime_fn(input_frame_time_flat)
        embedded_times = [embedded_time, embedded_time]

    else:
        assert NotImplementedError

    # embed views
    if viewdirs is not None:
        input_dirs = viewdirs[:,None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat, position_delta_flat = batchify(fn, netchunk)(embedded, embedded_times)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    position_delta = torch.reshape(position_delta_flat, list(inputs.shape[:-1]) + [position_delta_flat.shape[-1]])
    return outputs, position_delta

def batchify_rays(rays_flat, chunk=1024*32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret

def render(H, W, focal, chunk=1024*32, rays=None, c2w=None, ndc=True,
                  near=0., far=1., frame_time=None,
                  use_viewdirs=False, c2w_staticcam=None,
                  **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for 
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, focal, c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, focal, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    sh = rays_d.shape # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, focal, 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()

    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    if len(frame_time.shape) <= 1:
        frame_time = frame_time * torch.ones_like(rays_d[...,:1])
    rays = torch.cat([rays_o, rays_d, near, far, frame_time], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape
    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]

def render_path(render_poses, render_times, hwf, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0, save_also_gt=False, i_offset=0, scaling_factor=None, do_evim=False, render_scaler=torch.ones(1), starting_idx=0, pos_thresh=0.2, neg_thresh=0.2):

    H, W, focal = hwf

    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    if savedir is not None:
        save_dir_estim = os.path.join(savedir, "estim")
        save_dir_gt = os.path.join(savedir, "gt")
        if not os.path.exists(save_dir_estim):
            os.makedirs(save_dir_estim)
        if save_also_gt and not os.path.exists(save_dir_gt):
            os.makedirs(save_dir_gt)
        
        if do_evim:
            os.makedirs(os.path.join(save_dir_estim, 'rgb'))
            os.makedirs(os.path.join(save_dir_estim, 'evim'))
            if save_also_gt:
                os.makedirs(os.path.join(save_dir_gt, 'rgb'))
                os.makedirs(os.path.join(save_dir_gt, 'evim'))

    if do_evim:

        rgbs0 = []
        disps0 = []
        evims = []

        for i in tqdm(range(starting_idx, len(render_poses)-1)):
            rgb0, disp0, acc0, _ = render(H, W, focal, chunk=chunk, c2w=render_poses[i][:3,:4], frame_time=render_times[i], **render_kwargs)
            rgb1, disp1, acc1, _ = render(H, W, focal, chunk=chunk, c2w=render_poses[i+1][:3,:4], frame_time=render_times[i+1], **render_kwargs)
            pred_evim = compute_pred_ev(rgb1, rgb0, 1/torch.abs(render_scaler))

            # scale rgbs as necessary
            # if render_scaler is negative, scale rgbs to 1
            if render_scaler < 0:
                rgb0 /= rgb0.max()
                rgb1 /= rgb1.max()

            rgbs0.append(rgb0)
            disps0.append(disp0)
            evims.append(pred_evim)

            if savedir is not None:
                # save rgb0
                rgb8_estim = to8b(rgbs0[-1].cpu().numpy())
                filename = os.path.join(save_dir_estim, 'rgb', '{:06d}.png'.format(i+i_offset))
                imageio.imwrite(filename, rgb8_estim)
                # if last one, then save the rgb1 as well so we have a complete rgb image dataset
                if i == len(render_poses)-2:
                    rgb8_estim = to8b(rgb1.cpu().numpy())
                    filename = os.path.join(save_dir_estim, 'rgb', '{:06d}.png'.format(i+i_offset+1))
                    imageio.imwrite(filename, rgb8_estim)

                # save evim
                pred_evim_vis = visualize_evim(pred_evim, darken_factor=1.0, pos_thresh=pos_thresh, neg_thresh=neg_thresh)
                filename = os.path.join(save_dir_estim, 'evim', '{:06d}.png'.format(i+i_offset))
                imageio.imwrite(filename, pred_evim_vis)

                if save_also_gt:
                    rgb8_gt = to8b(gt_imgs['images'][i].cpu().numpy())
                    filename = os.path.join(save_dir_gt, 'rgb', '{:06d}.png'.format(i+i_offset))
                    imageio.imwrite(filename, rgb8_gt)
                    # if last one, then save the rgb1 as well so we have a complete rgb image dataset
                    if i == len(render_poses)-2:
                        rgb8_gt = to8b(gt_imgs['images'][i+1].cpu().numpy())
                        filename = os.path.join(save_dir_gt, 'rgb', '{:06d}.png'.format(i+i_offset+1))
                        imageio.imwrite(filename, rgb8_gt)

                    # also save test evim
                    gt_evim_vis = visualize_evim(gt_imgs['evims'][i], darken_factor=1.0, pos_thresh=pos_thresh, neg_thresh=neg_thresh)
                    filename = os.path.join(save_dir_gt, 'evim', '{:06d}.png'.format(i+i_offset))
                    imageio.imwrite(filename, gt_evim_vis)

        rgbs0 = torch.stack(rgbs0, 0)
        disps0 = torch.stack(disps0, 0)
        evims = torch.stack(evims, 0)

        return rgbs0, disps0, evims

    else:

        rgbs = []
        disps = []

        for i, (c2w, frame_time) in enumerate(zip(tqdm(render_poses), render_times)):
            rgb, disp, acc, _ = render(H, W, focal, chunk=chunk, c2w=c2w[:3,:4], frame_time=frame_time, **render_kwargs)
            rgbs.append(rgb.cpu().numpy())
            disps.append(disp.cpu().numpy())

            if savedir is not None:
                rgb8_estim = to8b(rgbs[-1])
                filename = os.path.join(save_dir_estim, '{:06d}.png'.format(i+i_offset))
                imageio.imwrite(filename, rgb8_estim)
                if save_also_gt:
                    rgb8_gt = to8b(gt_imgs[i])
                    filename = os.path.join(save_dir_gt, '{:06d}.png'.format(i+i_offset))
                    imageio.imwrite(filename, rgb8_gt)

        rgbs = np.stack(rgbs, 0)
        disps = np.stack(disps, 0)

        return rgbs, disps

def create_nerf(args):
    """Instantiate NeRF's MLP model.
    """
    embed_fn, input_ch = get_embedder(args.multires, 3, args.i_embed)
    embedtime_fn, input_ch_time = get_embedder(args.multires, 1, args.i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, 3, args.i_embed)

    # NOTE output_ch not used when view_dirs=True (always)
    output_ch = 5 if args.N_importance > 0 else 4

    # NOTE adding output_color_ch argument to change it for rgb or gray
    if args.dataset_type == 'blender':
        output_color_ch = 3
    elif args.dataset_type == 'gray' or 'events' in args.dataset_type:
        output_color_ch = 1
    else:
        sys.exit(f'{args.dataset_type} dataset type not recognized for setting output_color_ch')

    skips = [4]
    model = NeRF.get_by_name(args.nerf_type, D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, input_ch_time=input_ch_time,
                 use_viewdirs=args.use_viewdirs, embed_fn=embed_fn,
                 zero_canonical=not args.not_zero_canonical, output_color_ch=output_color_ch,
                 ).to(device)
    grad_vars = list(model.parameters())

    model_fine = None
    if args.use_two_models_for_fine:
        model_fine = NeRF.get_by_name(args.nerf_type, D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, input_ch_time=input_ch_time,
                          use_viewdirs=args.use_viewdirs, embed_fn=embed_fn,
                          zero_canonical=not args.not_zero_canonical).to(device)
        grad_vars += list(model_fine.parameters())

    network_query_fn = lambda inputs, viewdirs, ts, network_fn : run_network(inputs, viewdirs, ts, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                embedtime_fn=embedtime_fn,
                                                                netchunk=args.netchunk,
                                                                embd_time_discr=args.nerf_type!="temporal")

    # create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_file is not None and args.ft_file != 'None':
        ckpts = [args.ft_file]
    else:
        if args.ft_path is not None and args.ft_path != 'None':
            ckptdir = args.ft_path
        else:
            ckptdir = os.path.join(args.workspace, expname)
        isExist = os.path.exists(ckptdir)
        sorted_files = sorted(os.listdir(ckptdir)) if isExist else []
        does_tar_exist = ['hit!' for f in sorted_files if 'tar' in f]
        ckpts = [os.path.join(ckptdir, f) for f in sorted_files if 'tar' in f] if does_tar_exist else []

    if not args.no_reload:
        if len(ckpts) > 0:
            ckpt_path = ckpts[-1] # choose the last weights file
            mylogger(logfile, f'[CKPT] Reloading from {ckpt_path}')
            ckpt = torch.load(ckpt_path)

            start = ckpt['global_step']
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])

            # Load model
            model.load_state_dict(ckpt['network_fn_state_dict'])
            if model_fine is not None:
                model_fine.load_state_dict(ckpt['network_fine_state_dict'])
        else:
            mylogger(logfile, f'No ckpt found at directory {ckptdir}')
    else:
        mylogger(logfile, f'No reload!')

    ##########################

    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'network_fine': model_fine,
        'N_samples' : args.N_samples,
        'network_fn' : model,
        'use_viewdirs' : args.use_viewdirs,
        'white_bkgd' : args.white_bkgd,
        'raw_noise_std' : args.raw_noise_std,
        'use_two_models_for_fine' : args.use_two_models_for_fine,
    }

    # NDC only good for LLFF-style forward facing data
    # if args.dataset_type != 'llff' or args.no_ndc:
    if not args.three_view or args.no_ndc:
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer

def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10], device=device).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    rgb = torch.sigmoid(raw[...,:1])  # [N_rays, N_samples, 1]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
            noise = torch.Tensor(noise, device=device)

    alpha = raw2alpha(raw[...,-1] + noise, dists, torch.sigmoid)  # [N_rays, N_samples]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])
        # rgb_map = rgb_map + torch.cat([acc_map[..., None] * 0, acc_map[..., None] * 0, (1. - acc_map[..., None])], -1)

    return rgb_map, disp_map, acc_map, weights, depth_map

def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False,
                z_vals=None,
                use_two_models_for_fine=False):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """

    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
    viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 9 else None
    bounds = torch.reshape(ray_batch[...,6:9], [-1,1,3])
    near, far, frame_time = bounds[...,0], bounds[...,1], bounds[...,2] # [-1,1]
    z_samples = None
    rgb_map_0, disp_map_0, acc_map_0, position_delta_0 = None, None, None, None

    if z_vals is None:
        t_vals = torch.linspace(0., 1., steps=N_samples)
        if not lindisp:
            z_vals = near * (1.-t_vals) + far * (t_vals)
        else:
            z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

        z_vals = z_vals.expand([N_rays, N_samples])

        if perturb > 0.:
            # get intervals between samples
            mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
            upper = torch.cat([mids, z_vals[...,-1:]], -1)
            lower = torch.cat([z_vals[...,:1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape)

            # Pytest, overwrite u with numpy's fixed random numbers
            if pytest:
                np.random.seed(0)
                t_rand = np.random.rand(*list(z_vals.shape))
                t_rand = torch.Tensor(t_rand, device=device)

            z_vals = lower + (upper - lower) * t_rand

        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]

        if N_importance <= 0:
            raw, position_delta = network_query_fn(pts, viewdirs, frame_time, network_fn)
            rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

        else:
            if use_two_models_for_fine:
                raw, position_delta_0 = network_query_fn(pts, viewdirs, frame_time, network_fn)
                rgb_map_0, disp_map_0, acc_map_0, weights, _ = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

            else:
                with torch.no_grad():
                    raw, _ = network_query_fn(pts, viewdirs, frame_time, network_fn)
                    _, _, _, weights, _ = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

            z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
            z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest)
            z_samples = z_samples.detach()
            z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)

    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]
    run_fn = network_fn if network_fine is None else network_fine
    raw, position_delta = network_query_fn(pts, viewdirs, frame_time, run_fn)
    rgb_map, disp_map, acc_map, weights, _ = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map, 'z_vals' : z_vals, 'position_delta' : position_delta, 'weights' : weights}
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        if rgb_map_0 is not None:
            ret['rgb0'] = rgb_map_0
        if disp_map_0 is not None:
            ret['disp0'] = disp_map_0
        if acc_map_0 is not None:
            ret['acc0'] = acc_map_0
        if position_delta_0 is not None:
            ret['position_delta_0'] = position_delta_0
        if z_samples is not None:
            ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            mylogger(logfile, f"! [Numerical Error] {k} contains nan or inf.")

    return ret

def form_eventframe_batched(evs, views, H, W, times0, times1=None, N_ev_rays=512, N=None, device='cpu', is_half_res=False, precrop_frac=1.0, pos_thresh=0.2, neg_thresh=0.2, center=None):

    if center is None:
        center = [H//2, W//2]

    if precrop_frac > 0.0:
        dH = int(H//2 * precrop_frac)
        dW = int(W//2 * precrop_frac)
    else:
        dH = int(H//2)
        dW = int(W//2)

    x = torch.zeros(views.shape[0])
    y = torch.zeros(views.shape[0])
    times1_ = torch.zeros(views.shape[0])
    target = torch.zeros(views.shape[0])

    # for each target value, we need to choose a single event pixel or random pixel
    # this would be more efficient if calculated each viewpoint's target values as a batch
    # sorted_indices = np.argsort(views)
    # views = views[sorted_indices]
    # times0 = times0[sorted_indices]
    # curr_view_i = -1
    for i in range(views.shape[0]):

        if i < N_ev_rays: # choose from nonzero events

            # if views[i] > curr_view_i:
            view_events = evs[views[i]]
                # curr_view_i = views[i]
            vet = view_events[view_events[:,0] >= times0[i]*1e9][:N]
            if precrop_frac > 0.0:
                precrop_mask = torch.bitwise_and( torch.bitwise_and( vet[:,2] >= (center[0] - dH) , vet[:,2] < (center[0] + dH) ) , torch.bitwise_and( vet[:,1] >= (center[1] - dW) , vet[:,1] < (center[1] + dW) ) )
                vet = vet[precrop_mask]

            # this is biased towards locations with lots of events
            # rand_px = np.random.randint(0, vet.shape[0])
            # rand_x = vet[rand_px, 1]
            # rand_y = vet[rand_px, 2]
            # so instead, choose randomly from the unique values
            unique_xs = torch.unique(vet[:,1])
            rand_x = unique_xs[np.random.randint(0, unique_xs.shape[0])]
            unique_ys = torch.unique(vet[:,2])
            rand_y = unique_ys[np.random.randint(0, unique_ys.shape[0])]

        else: # choose randomly

            rand_x = np.random.randint(max(center[0]-dW, 0), min(center[0]+dW, W))
            rand_y = np.random.randint(max(center[1]-dH, 0), min(center[1]+dH, H))

        x[i] = rand_x
        y[i] = rand_y

        rand_evs_idxs = torch.bitwise_and( vet[:,1]==rand_x , vet[:,2]==rand_y )
        rand_negevs_idxs = torch.bitwise_and( rand_evs_idxs , vet[:,3]<0 )
        rand_posevs_idxs = torch.bitwise_and( rand_evs_idxs , vet[:,3]>0 )
        target_value = pos_thresh*rand_posevs_idxs.sum() - neg_thresh*rand_negevs_idxs.sum()
        target[i] = target_value

        times1_[i] = (vet[-1,0]+1) / 1e9

    # if is_half_res:
    #     target /= 4.0

    # un-sort the outputs to undo our original sorting
    # x = x[np.argsort(sorted_indices)]
    # y = y[np.argsort(sorted_indices)]
    # target = target[np.argsort(sorted_indices)]
    # times1_ = times1_[np.argsort(sorted_indices)]

    return x.long(), y.long(), target.to(device), times1_.to(device)

def form_eventframe(view_events, H, W, times0, times1=None, N=None, device='cpu', is_half_res=False, pos_thresh=0.2, neg_thresh=0.2):
    if times1 is not None:
        # extract the time-sliced events
        # it's likely that view_events is on cpu
        valid_ev_idxs_timed = torch.bitwise_and(view_events[:,0] >= times0.cpu()*1e9, view_events[:,0] < times1[0].cpu()*1e9)
        view_events_timed = view_events[valid_ev_idxs_timed]
    elif N is not None:
        view_events_timed = view_events[view_events[:,0] >= times0*1e9][:N]
        times1 = (view_events_timed[-1,0]+1) / 1e9
    else:
        raise ValueError("form_eventframe() requires either times1 or N to be not None")

    view_events_timed_pos = view_events_timed[view_events_timed[:,-1] > 0]
    view_events_timed_neg = view_events_timed[view_events_timed[:,-1] < 0]
    frame = pos_thresh*np.histogram2d(view_events_timed_pos[:,1].cpu().numpy(), view_events_timed_pos[:,2].cpu().numpy(), bins=(W, H), range=[[0, W], [0, H]])[0] - neg_thresh*np.histogram2d(view_events_timed_neg[:,1].cpu().numpy(), view_events_timed_neg[:,2].cpu().numpy(), bins=(W, H), range=[[0, W], [0, H]])[0]

    # # if continuous eventstream was made half-res, then 4x events have coalesced into a single pixel.
    # # make up for it by dividing the final sum per pixel by 4.
    # if is_half_res:
    #     frame = (frame / 4.0) // pos_thresh * pos_thresh

    return torch.Tensor(frame.T).to(device), torch.Tensor(times1).to(device)

def bin_evim(evim, target_maxabs_value, pos_thresh=0.2, neg_thresh=0.2):
    binned_evim = evim * target_maxabs_value
    pos_ids = evim > 0
    neg_ids = evim < 0
    binned_evim[pos_ids] = evim[pos_ids] // pos_thresh
    binned_evim[neg_ids] = evim[neg_ids] // neg_thresh
    return binned_evim

def visualize_evim(evim, pos_thresh=0.2, neg_thresh=0.2, darken_factor=0.7):
    # darken_factor=1.0 means no extra darkening

    frame = np.ones((*evim.shape, 3))
    binned = bin_evim(evim, target_maxabs_value=1.0, pos_thresh=pos_thresh, neg_thresh=neg_thresh)
    binned = binned.cpu()

    # R or B pixel based on net of all events at pixel during timeframe
    neg_ids = (binned<0).nonzero()
    pos_ids = (binned>0).nonzero()
    # note in below, binned[neg_ids] < 0 and binned[pos_ids] > 0
    frame[neg_ids[:,0], neg_ids[:,1], 0] = darken_factor + binned[neg_ids[:,0], neg_ids[:,1]]/binned.abs().max()/(1/darken_factor)
    frame[neg_ids[:,0], neg_ids[:,1], 1] = darken_factor + binned[neg_ids[:,0], neg_ids[:,1]]/binned.abs().max()/(1/darken_factor)
    frame[pos_ids[:,0], pos_ids[:,1], 1] = darken_factor - binned[pos_ids[:,0], pos_ids[:,1]]/binned.abs().max()/(1/darken_factor)
    frame[pos_ids[:,0], pos_ids[:,1], 2] = darken_factor - binned[pos_ids[:,0], pos_ids[:,1]]/binned.abs().max()/(1/darken_factor)

    return (frame*255.0).astype(np.uint8)

def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument("--expname", type=str, 
                        help='experiment name')
    parser.add_argument("--workspace", type=str, default='./logs/', 
                        help='where to store ckpts and logs')
    parser.add_argument("--suffix", type=str, default='', 
                        help='(optional) suffix to append to workspace tag')
    parser.add_argument("--root_path", type=str, default='',
                        help='path of main file')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern',
                        help='input data directory')

    # training options
    parser.add_argument("--nerf_type", type=str, default="original",
                        help='nerf network type')
    parser.add_argument("--N_iter", type=int, default=500000,
                        help='num training iterations')
    parser.add_argument("--netdepth", type=int, default=8, 
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256, 
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8, 
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256, 
                        help='channels per layer in fine network')
    parser.add_argument("--N_rays", type=int, default=1024, 
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--frac_bg_rays", type=float, default=0.1, 
                        help='fraction of N_rays dedicated for random selection (bg)')
    parser.add_argument("--do_half_precision", action='store_true',
                        help='(no longer implemented) do half precision training and inference')
    parser.add_argument("--lrate", type=float, default=5e-4, 
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=float, default=500e3, 
                        help='exponential learning rate decay')
    parser.add_argument("--chunk", type=int, default=1024*32, 
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64, 
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true', 
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true', 
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None, 
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--ft_file", type=str, default=None, 
                        help='specific weights .tar file to reload for coarse network')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64, 
                        help='number of coarse samples per ray')
    parser.add_argument("--not_zero_canonical", action='store_true',
                        help='if set zero time is not the canonic space')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true', 
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0, 
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10, 
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4, 
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0., 
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')
    parser.add_argument("--use_two_models_for_fine", action='store_true',
                        help='use two models for fine results')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_iters_time", type=int, default=0,
                        help='number of steps to train on central time')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops')
    parser.add_argument("--add_tv_loss", action='store_true',
                        help='evaluate tv loss')
    parser.add_argument("--tv_loss_weight", type=float,
                        default=1.e-4, help='weight of tv loss')

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff', 
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=4,
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    ## deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek', 
                        help='options : armchair / cube / greek / vase')

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true', 
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true', 
                        help='load data and make it half the resolution by interpolation')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8, 
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true', 
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true', 
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true', 
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8, 
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=100,
                        help='frequency of console printout and metric logging')
    parser.add_argument("--i_img",     type=int, default=10000,
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=10000000,
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=10000000,
                        help='frequency of render_poses video saving')

    # my arguments
    parser.add_argument("--random_seed", type=int, default=0,
                        help='set the random seed to use')

    parser.add_argument("--keyframing", type=int, default=0,
                        help='set the highest keyframing split, to reduce by factors of 2')

    parser.add_argument("--near", type=float, default=2.0,
                        help='ray distance bounds: near')

    parser.add_argument("--far", type=float, default=6.0,
                        help='ray distance bounds: far')

    parser.add_argument("--render_only", action='store_true', 
                        help='do not optimize; reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true', 
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0, 
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    parser.add_argument("--render_test_path", type=str, default='None',
                        help='directory containing transforms_test.json with requested test path')

    parser.add_argument("--render_scaler", type=float, default=1.0,
                        help='Scaler to scale up image (exp) and evim (mult) renders')

    parser.add_argument("--weight_init", type=str, default=None,
                        help='weight initialization of DNeRF')

    parser.add_argument("--single_view", type=int, default=-1,
                        help="load only a single viewpoint of a multiview dataset")

    parser.add_argument("--N_batched_evs", type=int, default=0,
                        help='(1 means randomize) train on N-batch of events instead of the datasets consecutive timesteps')

    parser.add_argument("--randomized_t0", action='store_true', 
                        help='randomize index0 instead of choosing from the [0, T-2]dataset timesteps')

    parser.add_argument("--time_window", type=float, default=0.0,
                        help='(work in progress) time window duration of events to train on, if using randomized_t0 and not using N_batched_evs; if 0.0 then randomly chosen')

    parser.add_argument("--lr_warmup_iters", type=int, default=0,
                        help='how many steps to do a linear warmup over')

    parser.add_argument("--grad_clipping", type=float, default=np.inf,
                        help='what value to clip grad norms to (0.5 was used for real-world data)')

    parser.add_argument("--batch_over_images", action='store_true', 
                        help='batching over multiple training images instead of rays in single training images')

    parser.add_argument("--gamma", type=float, default=-1.0,
                        help='gamma correction, meant for color-to-grayscale conversion')

    parser.add_argument("--evim_darken_factor", type=float, default=1.0,
                        help='darkening factor to manually make the saved events image colored correctly or visually discernable')

    parser.add_argument("--boi_dataselect", type=str, default='pcitt_uniform', 
                        help='pcitt increases training horizon gradually, and sawtooth increases likelihood of seeing the new, later training samples')

    parser.add_argument("--lr_decay_rate", type=float, default=0.1,
                        help='exponential learning rate decay factor (reached after lr_decay*1000 iters)')

    # NOTE not using args.dnerf option -- instead, directly using old run_dnerf.py script
    parser.add_argument("--dnerf", action='store_true', 
                        help='whether to train rgb0 only, i.e. dnerf')

    parser.add_argument("--num_views", type=int, default=18,
                        help='if dnerf, then need to manually specify number of training views in multiview dataset')

    parser.add_argument("--num_timesteps", type=int, default=32,
                        help='if dnerf, then need to manually specify number of training timesteps in multiview dataset')

    parser.add_argument("--is_e2vid", action='store_true', 
                        help='Rendered t=t0 from model corresponds to ground truth t=t1 (used for E2Vid->DNeRF baseline)')

    parser.add_argument("--render_train", action='store_true', 
                        help='Render on training dataset rather than testing, usually to generate an image dataset on which to train a dnerf model')

    parser.add_argument("--run_eval", type=int, default=0,
                        help='Eval on periodically saved weights over course of training; 0 means no eval, positive number is the skip factor when reading weights files (1 means eval on every saved weights file)')

    parser.add_argument("--three_view", action='store_true', 
                        help='whether to use only the first 3 views of training/val datasets, to emulate the 3-view real-data setup')

    parser.add_argument("--render_validation", action='store_true', 
                        help='Just run evaluation() function as would be done during training')

    parser.add_argument("--ray_jitter", action='store_true', 
                        help='Use ray jittering to improve validation viewpoint rendering')

    parser.add_argument("--starting_idx", type=int, default=0,
                        help='Starting index for render function')

    parser.add_argument("--dist_loss", type=float, default=0.0,
                        help='Use distortion regularization loss described in Mip-NeRF 360; dist_loss arg is the lambda regularizer on this term')

    parser.add_argument("--ev_threshold_loss", type=float, default=0.0,
                        help='Use custom piecewise loss for event-based reconstruction that sets loss to 0 for the per-pixel correct number of generated events and MSE everywhere else')

    parser.add_argument("--rgb_loss", type=float, default=0.0,
                        help='Use RGB loss (MSE, an actually single-channel) on both predicted images; rgb_loss arg is the weight on this term')

    parser.add_argument("--pos_thresh", type=float, default=0.2,
                        help='Positive event threshold that is surpassed in the following inequality to generate a positive event: log(i1) - log(i0) > pos_thresh')

    parser.add_argument("--neg_thresh", type=float, default=0.2,
                        help='Negative event threshold that is surpassed in the following inequality to generate a negative event: log(i1) - log(i0) < -neg_thresh')

    parser.add_argument("--path_to_gt_ims", type=str, default='', 
                        help='Path to ground truth images ONLY to serve for prediction image correction (affine transformation)')

    parser.add_argument("--affineshift_predims", action='store_true', 
                        help='Perform an affine transformation on predicted images to match white balance of ground truth images')

    parser.add_argument("--target_scaledown", type=float, default=0.0,
                        help='(0.0 reverts to scaling by target.max()) On MSELoss between target and predicted evim, scale down the target by this factor.')

    parser.add_argument("--num_views_training", type=int, default=0,
                        help='(0 means use all views) Number of interpolated views to use for training (usually of 18 views total), starting from view0.')

    parser.add_argument("--render_test_path_testskip", type=int, default=1,
                        help='specifically for render_test_path, how many frames to skip between each rendered frame')

    parser.add_argument("--render_test_path_start", type=int, default=0,
                        help='specifically for render_test_path, index to start rendering at')

    return parser

def my_lr_scheduler(global_step, lrate, lrate_decay, lr_warmup_iters, decay_rate):

    if global_step < lr_warmup_iters:
        lr = (0.9*lrate)/lr_warmup_iters * global_step + 0.1*lrate

    else:
        decay_steps = lrate_decay
        lr = lrate * (decay_rate ** ((global_step-lr_warmup_iters) / decay_steps))

    return lr

def mylogger(logfile, msg):
    print(msg)
    logfile.write(msg+'\n')

def train():

    #################################
    ### Experiment initialization ###
    #################################

    init_time = time.time()

    args = config_parser().parse_args()

    # set seeds
    set_seeds(args.random_seed)

    # Create log dir and file, and copy the args, config file
    from datetime import datetime
    expname = args.expname+datetime.now().strftime('_d%m_%d_t%H_%M')
    if args.suffix != '':
        expname += '_' + args.suffix
    args.workspace = os.path.join(args.workspace, expname)
    folder_name = args.workspace
    counter = 1
    while True:
        if not os.path.exists(args.workspace):
            break
        args.workspace = f"{folder_name}_{counter}"
        counter += 1
    os.makedirs(args.workspace, exist_ok=False)
    f = os.path.join(args.workspace, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(args.workspace, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())
    f = os.path.join(args.workspace, 'log.txt')
    global logfile
    logfile = open(f, 'w')

    # Summary writers
    writer = SummaryWriter(args.workspace)

    # save transforms train and val
    try:
        from shutil import copyfile
        copyfile(os.path.join(args.root_path, args.datadir, 'transforms_train.json'), os.path.join(args.workspace, 'transforms_train.json'))
        copyfile(os.path.join(args.root_path, args.datadir, 'transforms_val.json'), os.path.join(args.workspace, 'transforms_val.json'))
        if os.path.exists(os.path.join(args.root_path, args.datadir, 'transforms_test.json')):
            copyfile(os.path.join(args.root_path, args.datadir, 'transforms_test.json'), os.path.join(args.workspace, 'transforms_test.json'))
    except:
        pass
    
    mylogger(logfile, f'[PREP] Running experiment {os.path.basename(args.workspace)}')

    ###################
    ### Dataloading ###
    ###################

    # evdnerf training; only events-multiview dataset handling is implemented for both evdnerf training and dnerf (args.dnerf=True) training

    if args.dataset_type == 'events-multiview':

        datadir = os.path.join(args.root_path, args.datadir)

        if not args.render_test:

            mylogger(logfile, f'[DATALOADER] Loading TRAINset {args.datadir}')
            st = time.time()
            data_train = load_events_multiview(datadir, 'train', args.N_batched_evs>0 or args.randomized_t0 or args.time_window>0.0, args.half_res, single_view=args.single_view)
            data_train = preload_data(data_train, device)
            mylogger(logfile, f"[DATALOADER] Done ({time.time()-st:.3f}s).")

            mylogger(logfile, f'[DATALOADER] Loading VALset {args.datadir}')
            st = time.time()
            data_val = load_events_multiview(datadir, 'val' if not args.single_view>-1 else 'train', False, args.half_res, single_view=args.single_view)
            data_val = preload_data(data_val, device)
            mylogger(logfile, f"[DATALOADER] Done ({time.time()-st:.3f}s).")

            if args.three_view:
                # NOTE manually editing data to be 3 training viewpoints, 5 validation viewpoints
                num_ts = data_train['num_timesteps']
                data_train['images'] = data_train['images'][:3*num_ts, ...]
                data_train['evims'] = data_train['evims'][:3*num_ts, ...]
                data_train['poses'] = data_train['poses'][:3*num_ts, ...]
                data_train['times'] = data_train['times'][:3*num_ts, ...]
                data_train['num_views'] = 3
                num_ts = data_val['num_timesteps']
                data_val['images'] = data_val['images'][:5*num_ts, ...]
                data_val['evims'] = data_val['evims'][:5*num_ts, ...]
                data_val['poses'] = data_val['poses'][:5*num_ts, ...]
                data_val['times'] = data_val['times'][:5*num_ts, ...]
                data_val['num_views'] = 5
            
            if args.num_views_training > 0:
                num_ts = data_train['num_timesteps']
                num_vs = data_train['num_views']

                ids = np.arange(num_ts * num_vs)
                des_ids = []
                des_vs = [int(vi) for vi in np.arange(0, num_vs, num_vs/args.num_views_training)]
                mylogger(logfile, f'[DATALOADER] Using only views {des_vs} for training')
                for dv in des_vs:
                    group = ids[dv*num_ts:dv*num_ts+num_ts]
                    des_ids.extend(group)

                data_train['images'] = data_train['images'][des_ids, ...]
                data_train['evims'] = data_train['evims'][des_ids, ...]
                data_train['poses'] = data_train['poses'][des_ids, ...]
                data_train['times'] = data_train['times'][des_ids, ...]
                data_train['num_views'] = args.num_views_training

            mylogger(logfile, f"[DATALOADER] Loaded TRAINset events-multiview {data_train['images'].shape}, {data_train['intrinsics']}, {args.datadir}")
            mylogger(logfile, f"[DATALOADER] Loaded VALset events-multiview {data_val['images'].shape}, {data_val['intrinsics']}, {args.datadir}")

            num_views = data_train['num_views']
            num_timesteps = data_train['num_timesteps']
            len_train = num_views * num_timesteps

        near = args.near
        far = args.far

        pos_thresh = args.pos_thresh
        neg_thresh = args.neg_thresh
        # if we are using half-res continuous event stream, thresholds should be scaled down at training time
        if args.half_res and (args.N_batched_evs > 0 or args.randomized_t0 or args.time_window > 0.0):
            pos_thresh /= 4.0
            neg_thresh /= 4.0

    else:
        mylogger(logfile, f'[DATALOADER] Unknown dataset type {args.dataset_type}; exiting.')
        return

    if not args.render_test:
        hwf = data_train['intrinsics']
        [H, W, focal] = hwf

    #############################################
    ### Externally define evaluation function ###
    #############################################

    def evaluation(iteration):
        torch.cuda.empty_cache()

        # save prediction and gt
        rgb_validation_path = os.path.join(args.workspace, 'validation', f'iter_{str(iteration).zfill(5)}', 'rgb')
        os.makedirs(rgb_validation_path)
        acc_validation_path = os.path.join(args.workspace, 'validation', f'iter_{str(iteration).zfill(5)}', 'acc')
        os.makedirs(acc_validation_path)
        depth_validation_path = os.path.join(args.workspace, 'validation', f'iter_{str(iteration).zfill(5)}', 'depth')
        os.makedirs(depth_validation_path)
        pred_evim_validation_path = os.path.join(args.workspace, 'validation', f'iter_{str(iteration).zfill(5)}', 'evim')
        os.makedirs(pred_evim_validation_path)
        target_evim_validation_path = os.path.join(args.workspace, 'validation', f'iter_{str(iteration).zfill(5)}', 'target_evim')
        os.makedirs(target_evim_validation_path)
        target_rgb_validation_path = os.path.join(args.workspace, 'validation', f'iter_{str(iteration).zfill(5)}', 'target_rgb')
        os.makedirs(target_rgb_validation_path)

        for im_i in tqdm(range(0, data_val['images'].shape[0], args.testskip)):

            if (im_i+1) % data_val['num_timesteps'] == 0:
                im_i -= 1

            target_rgb = data_val['images'][im_i]
            pose0 = data_val['poses'][im_i, :3,:4]
            frame_time0 = data_val['times'][im_i]
            pose1 = data_val['poses'][im_i+1, :3,:4]
            frame_time1 = data_val['times'][im_i+1]
            with torch.no_grad():
                rgb0, disp0, acc0, extras0 = render(H, W, focal, chunk=args.chunk, c2w=pose0, frame_time=frame_time0, **render_kwargs_test)
                rgb1, disp1, acc1, extras1 = render(H, W, focal, chunk=args.chunk, c2w=pose1, frame_time=frame_time1, **render_kwargs_test)

            # transform predicted image via affine transformation to match ground truth images
            if args.affineshift_predims:
                rgb0_corrected = correct_prediction_image(rgb0, target_rgb, args.path_to_gt_ims)
            else:
                rgb0_corrected = rgb0

            imageio.imwrite(
                os.path.join(rgb_validation_path, f"rgb_{str(im_i).zfill(5)}.png"),
                (to8b(rgb0_corrected.cpu().numpy())),
            )

            # save target rgb
            imageio.imwrite(
                os.path.join(target_rgb_validation_path, f"targetrgb_{str(im_i).zfill(5)}.png"),
                (to8b(target_rgb.cpu().numpy())),
            )

            imageio.imwrite(
                os.path.join(acc_validation_path, f"acc_{str(im_i).zfill(5)}.png"),
                ((acc0 > 0).float().cpu().numpy() * 255).astype(np.uint8),
            )
            imageio.imwrite(
                os.path.join(depth_validation_path, f"depth_{str(im_i).zfill(5)}.png"),
                (((1/disp0)/(1/disp0).max()).float().cpu().numpy() * 255).astype(np.uint8),
            )

            gamma = 1 if args.gamma < 0.0 else (args.gamma/data_val['evims'].max() if data_val['evims'].max()>0.0 else data_train['evims'].max())
            pred_evim = compute_pred_ev(rgb1, rgb0, gamma)
            pred_evim_vis = visualize_evim(pred_evim, darken_factor=args.evim_darken_factor, pos_thresh=args.pos_thresh, neg_thresh=args.neg_thresh)
            imageio.imwrite(
                os.path.join(pred_evim_validation_path, f"evim_{str(im_i).zfill(5)}.png"),
                pred_evim_vis,
            )
            
            # also save target evim for comparison
            if data_val['evims'] is not None:
                target_evim = data_val['evims'][im_i]
                target_evim_vis = visualize_evim(target_evim, darken_factor=args.evim_darken_factor, pos_thresh=args.pos_thresh, neg_thresh=args.neg_thresh)
                imageio.imwrite(
                    os.path.join(target_evim_validation_path, f"target_{str(im_i).zfill(5)}.png"),
                    target_evim_vis,
                )

            # log evaluation metrics
            for metric in all_metrics:
                # both intensity image and event images
                # for event frame metrics, compare in the original, unscaled space
                if "Int" in metric.name:
                    metric.update(rgb0_corrected, target_rgb.reshape(rgb0.shape))
                elif "EI" in metric.name:
                    if not im_i % data_val['num_timesteps'] == data_val['num_timesteps']-1:
                        metric.update(pred_evim.unsqueeze(-1), target_evim.unsqueeze(-1), frame_time0, im_i) # unnormalized events

        # log metrics for this evaluation
        mylogger(logfile, '[EVAL]')
        for metric in all_metrics:
            mylogger(logfile, metric.report())
            metric.write(writer, global_step, prefix="eval")
            metric.clear()
    
        mylogger(logfile, "[EVAL] finish summary")
        writer.flush()

    #######################################
    ### Create NeRF model and optimizer ###
    #######################################

    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    global_step = start
    bds_dict = {
        'near' : near,
        'far' : far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # need to load val just to get target_maxabs_value for PSNR metric
    if args.render_test:
        datadir = os.path.join(args.root_path, args.datadir)
        mylogger(logfile, f'[DATALOADER] Loading VALset {args.datadir}')
        st = time.time()
        data_val = load_events_multiview(datadir, 'val' if not args.single_view>-1 else 'train', False, args.half_res, single_view=args.single_view)
        mylogger(logfile, f"[DATALOADER] Done ({time.time()-st:.3f}s).")

    # all metrics
    if args.render_test_path == 'None' and not args.dnerf:
        all_metrics = [PSNRMeter("Int_PSNR", 1, device), VarDiffMeter("Int_VarDiff"), MAEMeter("Int_MAE"), SSIMMeter("Int_SSIM", 1), LPIPSMeter("Int_LPIPS"), PSNRMeter("EI_PSNR", data_range=2*data_val['target_maxabs_value'], device=device), VarDiffMeter("EI_VarDiff"), MAEMeter("EI_MAE"), SSIMMeter("EI_SSIM"), LPIPSMeter("EI_LPIPS")] #, ChamferDistMeter("EI_ChamferDist", num_timesteps-1, testskip=args.testskip if not args.render_only else 1)]
    else:
        all_metrics = [PSNRMeter("Int_PSNR", 1, device), VarDiffMeter("Int_VarDiff"), MAEMeter("Int_MAE"), SSIMMeter("Int_SSIM", 1), LPIPSMeter("Int_LPIPS")]

    ###############################
    ### Render_test or Run_eval ###
    ###############################

    if args.render_only:
        mylogger(logfile, f'[RENDER_ONLY] RENDER_ONLY={args.render_only} or RUN_EVAL={args.run_eval}')
        with torch.no_grad():
            images = None
            if args.render_test or args.render_train:

                if args.render_train:
                    data_test = data_train
                else:
                    mylogger(logfile, '[RENDER_TEST] Deleting val dataset and emptying torch cache to free up GPU memory')
                    del data_val
                    torch.cuda.empty_cache()

                    # load the testset
                    data_test = load_events_multiview(datadir, 'test', args.N_batched_evs>0, args.half_res, single_view=args.single_view)
                    data_test = preload_data(data_test, device)

                    hwf = data_test['intrinsics']
                    [H, W, focal] = hwf

                mylogger(logfile, f"[RENDER_ONLY] Loaded TESTset of events-multiview, {data_test['images'].shape}, {data_test['intrinsics']}, {args.datadir}")

                render_poses = data_test['poses']
                render_times = data_test['times']

            elif args.run_eval > 0:
                # find all weights saved (use ft.path to set ft.file)
                tarfiles = sorted(glob.glob(os.path.join(args.ft_path, '*.tar')))
                mylogger(logfile, f'[RENDER_ONLY, RUN_EVAL] Found {len(tarfiles)} weights files')
                tarfiles = tarfiles[::args.run_eval]
                mylogger(logfile, f'[RENDER_ONLY, RUN_EVAL] Reduced to {len(tarfiles)} weights files')
                
                if len(tarfiles) == 0:
                    mylogger(logfile, '[RENDER_ONLY, RUN_EVAL] run_eval=True but found no logfiles! Exiting.')
                    sys.exit()

                render_poses = data_val['poses']
                render_times = data_val['times']

            elif args.render_validation:
                # create nerf model according to ft_file
                render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
                global_step = start
                bds_dict = {
                    'near' : near,
                    'far' : far,
                }
                render_kwargs_train.update(bds_dict)
                render_kwargs_test.update(bds_dict)

                # render_function()
                evaluation(int(os.path.basename(args.ft_file)[:-4]))

                mylogger(logfile, '[RENDER_ONLY, RENDER_VALIDATION] Finished render_validation!')

            elif args.render_test_path != 'None':
                # load render poses and times from direct path
                with open(args.render_test_path, 'r') as fp:
                    metas = json.load(fp)
                render_poses = [np.array(frame['transform_matrix']) for frame in metas['frames']]
                render_times = [frame['time'] for frame in metas['frames']]
                render_poses = torch.tensor(render_poses).to(device)
                render_times = torch.tensor(render_times).to(device)
                if args.render_test_path_testskip > 1:
                    # apply render_test_path_testskip
                    mylogger(logfile, f'[RENDER_ONLY, RENDER_TEST_PATH] Applying render_test_path_testskip={args.render_test_path_testskip}')
                    # extract pairs of frames according to render_test_path_testskip
                    idxs0 = list(range(0, len(render_poses)-1, args.render_test_path_testskip))
                    idxs1 = list(range(1, len(render_poses), args.render_test_path_testskip))
                    idxs = []
                    for x, y in zip(idxs0, idxs1):
                        idxs.extend([x, y])
                    render_poses = render_poses[idxs]
                    render_times = render_times[idxs]
                if args.render_test_path_start > 0:
                    render_poses = render_poses[args.render_test_path_start:]
                    render_times = render_times[args.render_test_path_start:]

            # need to load a fresh model and loop+save-to-writer stuff when doing run_eval to evaluate at various checkpoints

            def render_function():
                testsavedir = os.path.join(args.workspace, expname, 'renderonly_{}_{:06d}'.format('test' if args.render_test else ('val' if args.run_eval else 'path'), start))
                os.makedirs(testsavedir, exist_ok=True)

                do_evim = True

                render_scaler = torch.ones(1).to(device) if args.gamma<0 else torch.Tensor([22.6/args.gamma]).to(device)

                rgbs, disps, evims = render_path(render_poses, render_times, hwf, args.chunk, render_kwargs_test, gt_imgs=data_test if args.render_test and data_test['evims'] is not None else images, savedir=testsavedir, render_factor=args.render_factor, save_also_gt=True if args.render_test and data_test['evims'] is not None else False, do_evim=do_evim, render_scaler=render_scaler, starting_idx=args.starting_idx, pos_thresh=args.pos_thresh, neg_thresh=args.neg_thresh)
                mylogger(logfile, f'[RENDER_FUNCTION] Done rendering {testsavedir}')

                relevant_idxs = range(evims.shape[0])

                # do pred image correction
                if args.affineshift_predims:
                    for pred_im_i in range(len(rgbs)):
                        rgbs[pred_im_i] = correct_prediction_image(rgbs[pred_im_i], data_test['images'][pred_im_i], args.path_to_gt_ims)

                for ii, i_render in enumerate(relevant_idxs):
                    target_rgb = data_test['images'][i_render+args.is_e2vid].reshape(rgbs[i_render].shape) if args.render_test else torch.zeros_like(rgbs[i_render])
                    for metric in all_metrics:
                        mylogger(logfile, f'[RENDER_FUNCTION] Updating {metric.name}')
                        if "Int" in metric.name:
                            metric.update(rgbs[i_render], target_rgb)
                        elif "EI" in metric.name:
                            metric.update(evims[i_render].unsqueeze(-1), data_test['evims'][i_render+args.is_e2vid].unsqueeze(-1), [render_times[i_render+args.is_e2vid], render_times[relevant_idxs[ii+1]+args.is_e2vid] if ii+1<len(relevant_idxs) and relevant_idxs[ii+1]+args.is_e2vid<len(render_times) else None])

                # log metrics for this evaluation
                for metric in all_metrics:
                    mylogger(logfile, f'[RENDER_FUNCTION] Reporting {metric.name}')
                    mylogger(logfile, metric.report())
                    metric.write(writer, global_step, prefix="eval")
                    metric.clear()

            if not args.run_eval:

                ##################
                ## NOT run_eval ##
                ##################

                render_function()

                mylogger(logfile, '[RENDER_ONLY] Finished render!')

            else:

                ##############
                ## run_eval ##
                ##############

                for i, weightsfile in enumerate(tarfiles):

                    mylogger(logfile, f'[RENDER_ONLY, RUN_EVAL] Evaluating on {os.path.basename(weightsfile)}, weights file {i+1}/{len(tarfiles)}')

                    # check to make sure the weightsfile isn't empty
                    if os.path.getsize(weightsfile) < 1e3:
                        mylogger(logfile, f'[RENDER_ONLY, RUN_EVAL] Weightsfile is empty! Size: {os.path.getsize(weightsfile)} bytes. Continuing to next file.')
                        continue

                    # create nerf model according to ft_file
                    args.ft_file = weightsfile
                    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
                    global_step = start
                    bds_dict = {
                        'near' : near,
                        'far' : far,
                    }
                    render_kwargs_train.update(bds_dict)
                    render_kwargs_test.update(bds_dict)

                    # render_function()
                    evaluation(int(os.path.basename(weightsfile)[:-4]))

                mylogger(logfile, '[RENDER_ONLY, RUN_EVAL] Finished run_eval!')

            return

    # for EvDNeRF training, assume we always have evim frames
    # find "center" of nonzero events in evim frames to center the precrop during training
    centery = min(max((torch.abs(data_train['evims'][0])>min(pos_thresh, neg_thresh)).nonzero()[:,0].double().mean(), int(H//2 * args.precrop_frac)), H-int(H//2 * args.precrop_frac))
    centerx = min(max((torch.abs(data_train['evims'][0])>min(pos_thresh, neg_thresh)).nonzero()[:,1].double().mean(), int(W//2 * args.precrop_frac)), W-int(W//2 * args.precrop_frac))
    center = [int(centery), int(centerx)]

    # in dnerf baseline, all rays should be selected randomly
    if args.dnerf:
        args.frac_bg_rays = 1.0
        args.dataset_type = 'gray'
        args.rgb_loss = 1.0

    N_iters = args.N_iter + 1
    mylogger(logfile, '[TRAIN] Beginning at iteration {}'.format(start))
    
    running_timer = 0.

    running_rayctr = 0
    
    target_maxabs_value = None

    for i in range(start, N_iters+start):
        time0 = time.time()

        im_id1 = None

        ####################################
        ## NOT Batch over multiple images ##
        ####################################

        if not args.batch_over_images:

            ###############################
            ## Training index0 selection ##
            ###############################

            if not args.randomized_t0:

                # Choose an initial index im_id0

                # Either by linearly introducing more timesteps...
                if i >= args.precrop_iters_time:
                    im_id0 = np.random.choice(len_train)

                else:

                # Or all the datapoints at once...
                    if args.keyframing == 0:

                        # kf=0 reverts to the original tngp way of gradually introducing another timestep in the trajectory every pc_its_time/num_timesteps iters.
                        # It starts with the first 3 timesteps.

                        skip_factor = i / float(args.precrop_iters_time) * num_timesteps
                        max_sample = max(int(skip_factor), 3)
                        im_id0 = np.random.choice(max_sample)+num_timesteps*np.random.choice(num_views)

                # Or by keyframing.
                    else:

                        # keyframing method splits training time below pc_its_time to {num_kf_splits} equal-length periods.
                        # if keyframing = 4, then the first period will sample from all viewpoints but in time will skip ::4. The next period will be ::2, then the next will be post-pc_its_time so will not skip any.
                        # kf = 1 sees all data immediately, effectively bypassing the pc_its_time parameter.

                        num_kf_splits = np.log2(args.keyframing)

                        if num_kf_splits % 1 != 0:
                            sys.exit(f'keyframing argument is not a power of 2. args.keyframing={args.keyframing}')

                        kf_skip_factor = int(2**(num_kf_splits-np.floor(i/(args.precrop_iters_time/(num_kf_splits+1e-10)))))
                        view_i = np.random.choice(list(range(0,num_views)))
                        timestep_i = np.random.choice(list(range(0,num_timesteps-1,kf_skip_factor)))
                        im_id0 = view_i*num_timesteps + timestep_i

                # Ensure that im_id0 is not the last timestep
                if (im_id0+1) % num_timesteps == 0:
                    im_id0 -= 1

                # Save timestep0 training data
                timestep0 = data_train['times'][im_id0]
                im0 = data_train['images'][im_id0] # first gt image
                view_idx = im_id0 // num_timesteps
                pose0 = data_train['poses'][im_id0, :3, :4]
                pose1 = pose0.clone() if im_id1 is None else data_train['poses'][im_id1, :3, :4]

            # Else, choose a randomized t0
            else:

                precrop_iters_and_bound = min(max(i / args.precrop_iters_time, 0.1), 0.95)
                timestep0 = torch.Tensor([precrop_iters_and_bound*np.random.random()], device=device) # hardcoding some temporal padding here
                im0 = None
                view_idx = np.random.choice(num_views)
                pose0 = data_train['poses'][view_idx*num_timesteps, :3, :4]
                pose1 = pose0.clone() if im_id1 is None else data_train['poses'][view_idx*num_timesteps, :3, :4]

            ######################
            ## Target selection ##
            ######################

            if args.N_batched_evs > 0:

                if args.N_batched_evs > 1:
                    N_batched_evs = args.N_batched_evs
                else:
                    num_evs_per_frame = data_train['evs'][view_idx].shape[0] // (num_timesteps-1)
                    deviation = min((global_step-start)/30e3, .75) * num_evs_per_frame
                    if global_step % 100 == 0:
                        mylogger(logfile, f'[TRAIN] Deviation is {deviation}')
                    N_batched_evs = np.random.randint(num_evs_per_frame-deviation, num_evs_per_frame+deviation+1)
                
                view_events = data_train['evs'][view_idx]
                target, timestep1 = form_eventframe(view_events, H, W, times0=timestep0, N=N_batched_evs, device=device, is_half_res=args.half_res, pos_thresh=pos_thresh, neg_thresh=neg_thresh)

                # setting target_maxabs_value once will probably make training smoother
                if target_maxabs_value is None:
                    target_maxabs_value = target.abs().max()

            elif args.time_window > 0.0:

                view_events = data_train['evs'][view_idx]

                # we set a time window
                if args.time_window > 0.0:
                    timestep1 = timestep0 + args.time_window

                # randomly choose time window on each iter
                else:
                    default_time_window = 1.0 / (num_timesteps - 1)
                    timestep1 = min(timestep0 + 1.5 * np.random.random() * default_time_window, 1.0)

                target, timestep1 = form_eventframe(view_events, H, W, times0=timestep0, times1=[timestep1], device=device, is_half_res=args.half_res, pos_thresh=pos_thresh, neg_thresh=neg_thresh)
                target_maxabs_value = target.abs().max()

            # evims
            else:

                target = data_train['evims'][im_id0] # events
                target_maxabs_value = data_train['target_maxabs_value']

                # Choose an im_id1 that is the next timestep after im_id0
                im_id1 = im_id0 + 1

                # Save timestep1 training data
                timestep1 = data_train['times'][im_id1]
                im1 = data_train['images'][im_id1]

            ###################
            ## Ray selection ##
            ###################

            rays_o0, rays_d0 = get_rays(H, W, focal, pose0, args.ray_jitter)  # (H, W, 3), (H, W, 3)
            rays_o1, rays_d1 = get_rays(H, W, focal, pose1, args.ray_jitter)  # (H, W, 3), (H, W, 3)

            # original ray selection:
            if i < args.precrop_iters:
                dH = int(H//2 * args.precrop_frac)
                dW = int(W//2 * args.precrop_frac)
                coords = torch.stack(
                    torch.meshgrid(
                        torch.linspace(center[0] - dH, center[0] + dH - 1, 2*dH), 
                        torch.linspace(center[1] - dW, center[1] + dW - 1, 2*dW)
                    ), -1)
                if i == start:
                    mylogger(logfile, f"[Config] Center @ {center} cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}")
            else:
                coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1) # (H, W, 2)

            # ray selection scheme
            if "events" in args.dataset_type:

                # more bg rays prevents time-scattered hallucinations
                N_bg_rays = int(args.frac_bg_rays * args.N_rays)
                N_ev_rays = args.N_rays - N_bg_rays

                coords = coords.reshape(-1, coords.shape[-1])  # (H * W, 2)
                coords_ev = torch.nonzero(target) # target is (256, 256), coords_ev is (N, 2)
                if i < args.precrop_iters:
                    precrop_ev_mask = torch.bitwise_and( torch.bitwise_and(coords_ev[:,1] >= (center[1] - dW), coords_ev[:,1] < (center[1] + dW)), 
                                                         torch.bitwise_and(coords_ev[:,0] >= (center[0] - dH), coords_ev[:,0] < (center[0] + dH)) )
                    coords_ev = coords_ev[precrop_ev_mask]
                select_inds_ev = np.random.choice(coords_ev.shape[0], size=[min(N_ev_rays, coords_ev.shape[0])], replace=False)
                select_coords_ev = coords_ev[select_inds_ev]
                ev_ct = select_coords_ev.shape[0]
                select_bg_inds = np.random.choice(coords.shape[0], size=[max(N_bg_rays, 64)], replace=False)  # (N_rays,)
                select_coords_rand = coords[select_bg_inds].long()  # (N_rays, 2) # original
                select_coords = torch.cat((select_coords_ev, select_coords_rand), dim=0)

                rays_o0 = rays_o0[select_coords[:, 0], select_coords[:, 1]]  # (N_rays, 3)
                rays_d0 = rays_d0[select_coords[:, 0], select_coords[:, 1]]  # (N_rays, 3)
                batch_rays0 = torch.stack([rays_o0, rays_d0], 0)
                rays_o1 = rays_o1[select_coords[:, 0], select_coords[:, 1]]  # (N_rays, 3)
                rays_d1 = rays_d1[select_coords[:, 0], select_coords[:, 1]]  # (N_rays, 3)
                batch_rays1 = torch.stack([rays_o1, rays_d1], 0)
                target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rays, 3)
                target_rgb = im0[select_coords[:, 0], select_coords[:, 1]] if im0 is not None else None

            elif args.dataset_type == "gray" or args.dataset_type == "blender":

                coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
                select_inds = np.random.choice(coords.shape[0], size=[args.N_rays], replace=False)  # (N_rays,)
                select_coords = coords[select_inds].long()  # (N_rays, 2) # original

                rays_o0 = rays_o0[select_coords[:, 0], select_coords[:, 1]]  # (N_rays, 3)
                rays_d0 = rays_d0[select_coords[:, 0], select_coords[:, 1]]  # (N_rays, 3)
                batch_rays0 = torch.stack([rays_o0, rays_d0], 0)
                target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rays, 3)
                target_rgb = im0[select_coords[:, 0], select_coords[:, 1]]

            else:

                sys.exit(f'ray selection scheme not specified for dataset type {args.dataset_type}')

        ## END: NOT Batch over multiple images

        ################################
        ## Batch over multiple images ##
        ################################

        else:

            ###############################
            ## Training index0 selection ##
            ###############################

            # calculate how many images to select rays from for the batch
            # at first let's just assume we're doing all views at once?

            skip_factor = min(i / float(args.precrop_iters_time) * num_timesteps, num_timesteps-1)
            max_time_id = max(skip_factor, 3)

            candidate_im_ids = np.concatenate([np.arange(view_idx*num_timesteps, view_idx*num_timesteps+max_time_id) for view_idx in range(num_views)])

            # probability sawtooth
            if i < 2*args.precrop_iters_time and 'pcitt' in args.boi_dataselect:
                if 'sawtooth' in args.boi_dataselect:
                    full_probs = np.tile(np.concatenate((np.linspace(.25, 1.0, num_timesteps-1), np.zeros(1))), num_views)
                elif 'uniform' in args.boi_dataselect:
                    full_probs = np.ones(len_train)
                    full_probs[num_timesteps-1::num_timesteps] = 0
                else:
                    raise ValueError("pcitt boi_dataselect only accepts _sawtooth or _uniform distribution types")
                non_candidate_mask = np.ones_like(full_probs)
                non_candidate_mask[candidate_im_ids.astype(np.int32)] = 0
                full_probs[non_candidate_mask.astype(np.bool_)] = 0
            else:
                full_probs = np.ones(len_train)
                full_probs[num_timesteps-1::num_timesteps] = 0
            full_probs /= full_probs.sum()

            # for now just set a im_id0 even though the images selected won't correspond if using a randomized_t0
            im_id0 = np.random.choice(len_train, size=(args.N_rays), p=full_probs)
            im_id1 = None
            pose0 = data_train['poses'][im_id0, :3, :4]

            if not args.randomized_t0:
                timestep0 = data_train['times'][im_id0].unsqueeze(-1)
            else:
                precrop_iters_and_bound = min(max(i / args.precrop_iters_time, 0.1), 0.95)
                timestep0 = torch.Tensor([precrop_iters_and_bound*np.random.random(size=(args.N_rays))], device=device).T # hardcoding some temporal padding here

            ###################
            ## Ray selection ##
            ###################

            N_bg_rays = int(args.frac_bg_rays * args.N_rays)
            N_ev_rays = args.N_rays - N_bg_rays

            if args.N_batched_evs > 0:

                if args.N_batched_evs > 1:
                    N_batched_evs = args.N_batched_evs
                else:
                    num_evs_per_frame = data_train['evs'][0].shape[0] // (num_timesteps-1)
                    deviation = min((global_step-start)/30e3, .75) * num_evs_per_frame
                    if global_step % 100 == 0:
                        mylogger(logfile, f'[TRAIN] Deviation is {deviation}')
                    N_batched_evs = np.random.randint(num_evs_per_frame-deviation, num_evs_per_frame+deviation+1)

                x, y, target_s, timestep1 = form_eventframe_batched(data_train['evs'], im_id0//num_timesteps, H, W, timestep0, N_ev_rays=N_ev_rays, N=N_batched_evs, device=device, is_half_res=args.half_res, precrop_frac=args.precrop_frac if i<args.precrop_iters else -1.0, pos_thresh=pos_thresh, neg_thresh=neg_thresh, center=center)

                if target_maxabs_value is None:
                    target_maxabs_value = target_s.abs().max()

            else:

                im_id1 = im_id0 + 1

                timestep1 = data_train['times'][im_id1].unsqueeze(-1)
                
                coords_ev = torch.nonzero(data_train['evims'][im_id0[:N_ev_rays]])
                if i < args.precrop_iters:
                    dH = int(H//2 * args.precrop_frac)
                    dW = int(W//2 * args.precrop_frac)
                    precrop_ev_mask = torch.bitwise_and( torch.bitwise_and(coords_ev[:,2] >= (center[1] - dW), coords_ev[:,2] < (center[1] + dW)), torch.bitwise_and(coords_ev[:,1] >= (center[0] - dH), coords_ev[:,1] < (center[0] + dH)) )
                    coords_ev = coords_ev[precrop_ev_mask]

                    if i == start:
                        mylogger(logfile, f"[Config] Center @ {center} cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}")

                _, ev_lens = torch.unique(coords_ev[:,0], return_counts=True)
                ev_lens = ev_lens.cpu().numpy()
                select_im_coords_ev = torch.Tensor(np.random.randint(0, ev_lens, size=[ev_lens.shape[0]]), device=device)
                cumidxsums = np.hstack((0, np.cumsum(ev_lens[:-1])))
                select_im_coords_ev += torch.Tensor(cumidxsums, device=device)
                x = coords_ev[select_im_coords_ev.long(), 2]
                y = coords_ev[select_im_coords_ev.long(), 1]
                
                # add in some random bg rays
                if i < args.precrop_iters:
                    x = torch.hstack((x, torch.randint(center[1] - dW, center[1] + dW, size=(N_bg_rays,), device=device)))
                    y = torch.hstack((y, torch.randint(center[0] - dH, center[0] + dH, size=(N_bg_rays,), device=device)))
                else:
                    x = torch.hstack((x, torch.randint(0, W, size=(N_bg_rays,), device=device)))
                    y = torch.hstack((y, torch.randint(0, H, size=(N_bg_rays,), device=device)))

                # in the case there are fewer events than expected, trim image indices appropriately
                if x.shape[0] < im_id0.shape[0]:
                    im_id0 = im_id0[:x.shape[0]]
                    im_id1 = im_id1[:x.shape[0]]
                    pose0 = pose0[:x.shape[0]]
                    timestep0 = timestep0[:x.shape[0]]
                    timestep1 = timestep1[:x.shape[0]]
                target_s = data_train['evims'][im_id0, y, x]
                target_maxabs_value = data_train['target_maxabs_value']

            pose1 = pose0.clone() if im_id1 is None else data_train['poses'][im_id1, :3, :4]

            target_rgb = data_train['images'][im_id0, y, x]

            rays_o0, rays_d0 = get_batched_rays(H, W, focal, pose0, x, y, args.ray_jitter)  # (H, W, 3), (H, W, 3)
            rays_o1, rays_d1 = get_batched_rays(H, W, focal, pose1, x, y, args.ray_jitter)  # (H, W, 3), (H, W, 3)

            batch_rays0 = torch.stack([rays_o0, rays_d0], 0)
            batch_rays1 = torch.stack([rays_o1, rays_d1], 0)

            running_rayctr += batch_rays0.shape[1] + batch_rays1.shape[1]

        ## END: Batch over multiple images

        ##############################
        ##  Core optimization loop  ##
        ##############################

        optimizer.zero_grad()
        
        rgb0, disp0, acc0, extras0 = render(H, W, focal, chunk=args.chunk, rays=batch_rays0,
                                            frame_time=timestep0,
                                            verbose=i < 10, retraw=True,
                                            **render_kwargs_train)

        if args.dnerf:
            rgb_loss = img2mse(rgb0, target_rgb)
            rgb_psnr = mse2psnr(rgb_loss)

            ev_loss = torch.zeros(1)[0]
            ev_psnr = torch.zeros(1)

            # set ev recon loss term to 0 for dnerf training
            recon_loss = torch.zeros(1)[0]

            # calculate distortion regularization loss
            dist_loss0 = compute_dist_loss(extras0['weights'], extras0['z_vals'])
            dist_loss1 = torch.zeros(1)[0]

        else:
            
            rgb1, disp1, acc1, extras1 = render(H, W, focal, chunk=args.chunk, rays=batch_rays1,
                                                frame_time=timestep1,
                                                verbose=i < 10, retraw=True,
                                                **render_kwargs_train)

            if args.target_scaledown == 0.0:
                target_s_normalized = target_s / target_maxabs_value
            else:
                target_s_normalized = target_s / args.target_scaledown
            pred_ev = compute_pred_ev(rgb1, rgb0)

            pred_maxabs_value = 1.0 if args.target_scaledown == 1.0 else (target_maxabs_value if args.gamma < 0.0 else args.gamma)

            # sometimes scaling target and/or predictions can help with training
            ev_loss = img2mse(pred_ev/pred_maxabs_value, target_s_normalized)
            recon_loss = ev_loss
            ev_psnr = mse2psnr(ev_loss if args.target_scaledown == 1.0 else ev_loss*(target_maxabs_value**2) , 2*target_maxabs_value)

            if target_rgb is not None:
                rgb_loss = img2mse(rgb0, target_rgb)
                rgb_psnr = mse2psnr(rgb_loss)
                if im_id1 is not None:
                    target_rgb1 = data_train['images'][im_id1, y, x]
                    rgb_loss += img2mse(rgb1, target_rgb1)
                    rgb_loss /= 2.0
            else:
                rgb_loss = torch.zeros(1)[0]
                rgb_psnr = torch.zeros(1)

            # calculate distortion regularization loss
            dist_loss0 = compute_dist_loss(extras0['weights'], extras0['z_vals'])
            dist_loss1 = compute_dist_loss(extras1['weights'], extras1['z_vals'])

            # optionally overwrite ev_loss with ev_threshold_loss
            if args.ev_threshold_loss > 0.0:
                ev_loss = ev_threshold_loss(pred_ev, target_s, pos_thresh=pos_thresh, neg_thresh=neg_thresh)
                recon_loss = args.ev_threshold_loss * ev_loss
        
        dist_loss = dist_loss0 + dist_loss1
        loss = recon_loss + args.dist_loss * dist_loss + args.rgb_loss * rgb_loss

        loss.backward()

        gradnorm = torch.nn.utils.clip_grad_norm_(grad_vars, max_norm=args.grad_clipping)
        gradnorm_clipped = torch.nn.utils.clip_grad_norm_(grad_vars, max_norm=torch.inf) #extract gradnorm again for logging clipped

        optimizer.step()

        ##########################
        ## Update learning rate ##
        ##########################

        new_lrate = my_lr_scheduler(global_step-start, args.lrate, args.lrate_decay, args.lr_warmup_iters, args.lr_decay_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate

        #############
        ## Logging ##
        #############

        # save to tf writer every 50 iterations
        if i % 50 == 0:
            writer.add_scalar('train/loss', ev_loss, i)
            writer.add_scalar('train/dist_loss', dist_loss, i)
            writer.add_scalar('train/psnr', ev_psnr, i)
            writer.add_scalar('train/rgb_loss', rgb_loss, i)
            writer.add_scalar('train/rgb_psnr', rgb_psnr, i)
            writer.add_scalar('train/gradnorm', gradnorm, i)
            writer.add_scalar('train/gradnorm_clipped', gradnorm_clipped, i)
            writer.add_scalar('train/lr', new_lrate, i)
            writer.add_scalar('train/rayctr', running_rayctr, i)

        # print relevant stats to console
        dt = time.time()-time0
        running_timer += dt
        if i > 0 and i % args.i_print == 0:
            stats_txt = f"[TRAIN] Iter: {i}, {args.i_print/running_timer:.1f} it/s (T {int(time.time()-init_time):d} s) | EV Loss: {ev_loss:.3f} PSNR: {ev_psnr[0]:.3f} | RGB Loss: {rgb_loss:.3f} PSNR: {rgb_psnr[0]:.3f} | lr: {new_lrate:.3e}"
            mylogger(logfile, stats_txt)
            running_timer = 0

        # save checkpoint weights
        if i > 0 and i % args.i_weights==0:
            path = os.path.join(args.workspace, '{:06d}.tar'.format(i))
            save_dict = {
                'global_step': global_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            if render_kwargs_train['network_fine'] is not None:
                save_dict['network_fine_state_dict'] = render_kwargs_train['network_fine'].state_dict()

            torch.save(save_dict, path)
            mylogger(logfile, f'Saved checkpoints at {path}')

        # save validation image predictions and metrics
        if i > start and i % args.i_img==0:
            evaluation(i)

        ###########################
        ## Increment global step ##
        ###########################

        global_step += 1

    #########################
    ### Training complete ###
    #########################

    mylogger(logfile, f'[TRAIN] Training complete, experiment at {args.workspace}')
    logfile.close()

if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train()
