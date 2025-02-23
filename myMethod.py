from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs
import mast3r.utils.path_to_dust3r
from dust3r.inference import inference
from dust3r.utils.image import load_images
import numpy as np
import cv2
from utils import scale_intrinsics, run_pnp

def processPair(model, device, anchor_path, query_path, K, n_matches):
    #finds the transformation from query to anchor given a pair of {anchor,query} images
    # Load images
    pair_imgs = load_images([anchor_path, query_path], size=512, verbose=False)
    
    # Run inference
    output = inference([tuple(pair_imgs)], model, device, batch_size=1, verbose=False)
    
    view1 = output['view1'] #some form of the image to size of the point cloud -- used to get the size of the matches 
    view2 = output['view2'] 

    pred1 = output['pred1'] #3d point cloud 
    pred2 = output['pred2']

    desc1 = pred1['desc'].squeeze(0).detach() #local features
    desc2 = pred2['desc'].squeeze(0).detach()
    
    # Fast reciprocal NNs
    matches_im0, matches_im1 = fast_reciprocal_NNs(desc1, desc2, subsample_or_initxy1=8, device=device, dist='dot', block_size=2**13)
    
    # Filter matches
    ignore = 0
    boarder = 3
    H0, W0 = view1['true_shape'][0]
    H1, W1 = view2['true_shape'][0]
    
    valid_matches_im0 = (matches_im0[:, 0] >= boarder) & (matches_im0[:, 0] < int(W0) - boarder) & (
        matches_im0[:, 1] >= boarder) & (matches_im0[:, 1] < int(H0) - boarder - ignore)
    valid_matches_im1 = (matches_im1[:, 0] >= boarder) & (matches_im1[:, 0] < int(W1) - boarder) & (
        matches_im1[:, 1] >= boarder) & (matches_im1[:, 1] < int(H1) - boarder - ignore)
    
    valid_matches = valid_matches_im0 & valid_matches_im1
    matches_im0 = matches_im0[valid_matches]
    matches_im1 = matches_im1[valid_matches]
    
    pts3d_im0 = pred1['pts3d'].squeeze(0).detach().cpu().numpy() 
    pts3d_im1 = pred2['pts3d_in_other_view'].squeeze(0).detach().cpu().numpy() 

    conf_im0 = pred1['conf'].squeeze(0).detach().cpu().numpy() #confidence 
    conf_im1 = pred2['conf'].squeeze(0).detach().cpu().numpy()

    desc_conf_im0 = pred1['desc_conf'].squeeze(0).detach().cpu().numpy()
    desc_conf_im1 = pred2['desc_conf'].squeeze(0).detach().cpu().numpy()
    
    # Extract confidence scores for the matches
    match_conf_im0 = conf_im0[matches_im0[:, 1], matches_im0[:, 0]]
    match_conf_im1 = conf_im1[matches_im1[:, 1], matches_im1[:, 0]]

    
    combined_conf = np.minimum(match_conf_im0, match_conf_im1)
    top_indices = np.argsort(combined_conf)[::-1][:n_matches]
    
    filtered_matches_im0 = matches_im0[top_indices]
    filtered_matches_im1 = matches_im1[top_indices]
    
    # Run PnP
    #COMMENT: change this to dynamically adjust to query mast3r resolution
    w, h = cv2.imread(query_path).shape[:2][::-1]
    K_scaled = scale_intrinsics(K, w, h, 512, 384)
    ret_val, transformation = run_pnp(
        filtered_matches_im1.astype(np.float32),
        pts3d_im0[filtered_matches_im0[:, 1], filtered_matches_im0[:, 0], :].astype(np.float32),
        K_scaled.astype(np.float32)
    )
    
    n_matches_total = len(matches_im0)
    n_matches_filtered = len(filtered_matches_im0)
    return ret_val, transformation, n_matches_total, n_matches_filtered




