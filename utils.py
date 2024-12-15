from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs
import mast3r.utils.path_to_dust3r
from dust3r.inference import inference
from dust3r.utils.image import load_images

#imports for visualizing matches
import numpy as np
import torch
import torchvision.transforms.functional
from matplotlib import pyplot as pl
from mpl_toolkits.axes_grid1 import make_axes_locatable

import cv2 #for pnp
from pyproj import Proj, transform #cartographic transformations and coordinate conversions

#supressing unnecessary warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


import time
import matplotlib.pyplot as plt
import csv
import os
from PIL import Image

import folium
from folium.plugins import MarkerCluster
import base64
import pandas as pd
import io
from io import BytesIO
from IPython.display import display



def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def min_max_normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def makeHistogram(match_conf_im0,match_conf_im1,lowest_confidence_im0,lowest_confidence_im1):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    # Histogram for Image 0 matches
    ax1.hist(match_conf_im0, bins=50, edgecolor='black', color='skyblue')
    ax1.axvline(lowest_confidence_im0, color='red', linestyle='--')
    ax1.set_title('Histogram of Confidence Scores (Anchor Matches)')
    ax1.set_xlabel('Confidence Value')
    ax1.set_ylabel('Frequency')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Histogram for Image 1 matches
    ax2.hist(match_conf_im1, bins=50, edgecolor='black', color='lightgreen')
    ax2.axvline(lowest_confidence_im1, color='red', linestyle='--', label='Confidence Cutoff')
    ax2.set_title('Histogram of Confidence Scores (Query Matches)')
    ax2.set_xlabel('Confidence Value')
    ax2.set_ylabel('Frequency')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.show()    


def visualize2Dmatches(conf_im0, conf_im1,matches_im0,matches_im1,view1,view2,n_viz=20):
        
        num_matches = matches_im0.shape[0]
        print("Number of matches before confidence mask: ",num_matches)
        match_idx_to_viz = np.round(np.linspace(0, num_matches - 1, n_viz)).astype(int)
        viz_matches_im0, viz_matches_im1 = matches_im0[match_idx_to_viz], matches_im1[match_idx_to_viz]

        image_mean = torch.as_tensor([0.5, 0.5, 0.5], device='cpu').reshape(1, 3, 1, 1)
        image_std = torch.as_tensor([0.5, 0.5, 0.5], device='cpu').reshape(1, 3, 1, 1)

        viz_imgs = []
        for i, view in enumerate([view1, view2]):
            rgb_tensor = view['img'] * image_std + image_mean
            viz_imgs.append(rgb_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy())

        H0, W0, H1, W1 = *viz_imgs[0].shape[:2], *viz_imgs[1].shape[:2]
        img0 = np.pad(viz_imgs[0], ((0, max(H1 - H0, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
        img1 = np.pad(viz_imgs[1], ((0, max(H0 - H1, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
        img = np.concatenate((img0, img1), axis=1)
        pl.figure()
        pl.imshow(img)
        cmap = pl.get_cmap('jet')
        for i in range(n_viz):
            (x0, y0), (x1, y1) = viz_matches_im0[i].T, viz_matches_im1[i].T
            pl.plot([x0, x1 + W0], [y0, y1], '-+', color=cmap(i / (n_viz - 1)), scalex=False, scaley=False)
        pl.show(block=True)


        # Create the figure
        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot the image
        im = ax.imshow(img)
        ax.set_title('Image Matches with Confidence (Anchor - left, Query - Right)')

        # Create scatter plots of matches with color-coded confidence
        scatter_im0 = ax.scatter(matches_im0[:, 0], matches_im0[:, 1], 
                                c=conf_im0[matches_im0[:, 1], matches_im0[:, 0]], 
                                cmap='viridis', s=10, alpha=0.7)
        scatter_im1 = ax.scatter(matches_im1[:, 0] + W0, matches_im1[:, 1], 
                                c=conf_im1[matches_im1[:, 1], matches_im1[:, 0]], 
                                cmap='viridis', s=10, alpha=0.7)


        # Create a divider for the existing axes instance
        divider = make_axes_locatable(ax)

        # Add an axes to the right of the main axes
        cax = divider.append_axes("right", size="5%", pad=0.05)

        plt.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(vmin=conf_im0.min(), vmax=conf_im0.max()), 
                    cmap='viridis'), cax=cax, label='Confidence')


        plt.tight_layout()
        plt.show()

def getMasterOutout(anchor_image, query_image,n_matches,visualizeMatches=False): 
    #inputs known image and unknown image paths to return mast3r output

    device = 'cuda:5'
    schedule = 'cosine'
    lr = 0.01
    niter = 300
    boarder = 3

    model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    # you can put the path to a local checkpoint in model_name if needed
    ##load model and run inference
    model = AsymmetricMASt3R.from_pretrained(model_name).to(device)
    images = load_images([anchor_image, query_image], size=512)
    #print("Images shape: ",images[1]['true_shape'])
    mast3r_inference_start = time.time()
    output = inference([tuple(images)], model, device, batch_size=1, verbose=False)
    mast3r_inference_stop = time.time()
    mast3r_inference_time = mast3r_inference_stop-mast3r_inference_start
    print(f"Mast3r Inference Time: {mast3r_inference_time:.4f} seconds.")
    # at this stage, you have the raw dust3r predictions 
    #dust3r predictions are the original two heads outputing 3d point cloud and confidence
    #master adds another that includes an additional head for local features


    view1 = output['view1'] #some form of the image to size of the point cloud -- used to get the size of the matches 
    view2 = output['view2'] 

    pred1 = output['pred1'] #3d point cloud 
    pred2 = output['pred2']

    desc1 = pred1['desc'].squeeze(0).detach() #local features
    desc2 = pred2['desc'].squeeze(0).detach()

    # find 2D-2D matches between the two images
    #feature based mapping to recover point correspdances/matches
    point_matches_start=time.time()
    matches_im0, matches_im1 = fast_reciprocal_NNs(desc1, desc2, subsample_or_initxy1=8,
                                                    device=device, dist='dot', block_size=2**13)
    point_matches_stop = time.time()
    point_matches_time = point_matches_stop-point_matches_start
    print(f"Point Matches Time: {point_matches_time:.4f} seconds.")
    

    ignore = 0 #to ignore correspondances in the lower half of the image
    # ignore small border around the edge
    H0, W0 = view1['true_shape'][0]
    valid_matches_im0 = (matches_im0[:, 0] >= boarder) & (matches_im0[:, 0] < int(W0) - boarder) & (
        matches_im0[:, 1] >= boarder) & (matches_im0[:, 1] < int(H0) - boarder -ignore)

    H1, W1 = view2['true_shape'][0]
    valid_matches_im1 = (matches_im1[:, 0] >= boarder) & (matches_im1[:, 0] < int(W1) - boarder) & (
        matches_im1[:, 1] >= boarder) & (matches_im1[:, 1] < int(H1) - boarder -ignore)

    valid_matches = valid_matches_im0 & valid_matches_im1

    # matches are Nx2 image coordinates.
    matches_im0 = matches_im0[valid_matches]
    matches_im1 = matches_im1[valid_matches]

    # Convert the other outputs to numpy arrays
    #point correspondances with respect to the coordinate system of the first image 
    pts3d_im0 = pred1['pts3d'].squeeze(0).detach().cpu().numpy() 
    pts3d_im1 = pred2['pts3d_in_other_view'].squeeze(0).detach().cpu().numpy() 

    conf_im0 = pred1['conf'].squeeze(0).detach().cpu().numpy() #confidence 
    conf_im1 = pred2['conf'].squeeze(0).detach().cpu().numpy()

    desc_conf_im0 = pred1['desc_conf'].squeeze(0).detach().cpu().numpy()
    desc_conf_im1 = pred2['desc_conf'].squeeze(0).detach().cpu().numpy()
    
    # Extract confidence scores for the matches
    match_conf_im0 = conf_im0[matches_im0[:, 1], matches_im0[:, 0]]
    match_conf_im1 = conf_im1[matches_im1[:, 1], matches_im1[:, 0]]

    if visualizeMatches:
        visualize2Dmatches(conf_im0,conf_im1,matches_im0,matches_im1,view1,view2)

    # Sort matches by confidence scores
    sorted_indices_im0 = np.argsort(match_conf_im0)[::-1]
    sorted_indices_im1 = np.argsort(match_conf_im1)[::-1]

    top_matches_im0 = sorted_indices_im0[:min(n_matches, len(sorted_indices_im0))]
    top_matches_im1 = sorted_indices_im1[:min(n_matches, len(sorted_indices_im1))]

    # Find the lowest confidence score among the top n_matches
    lowest_confidence_im0 = match_conf_im0[top_matches_im0[-1]]
    lowest_confidence_im1 = match_conf_im1[top_matches_im1[-1]]

    #simple thresholding
    # conf_mask = (conf_im0[matches_im0[:, 1], matches_im0[:, 0]] > threshold) & \
    #             (conf_im1[matches_im1[:, 1], matches_im1[:, 0]] > threshold)

    # # Apply the mask to filter matches and other data
    # matches_im0 = matches_im0[conf_mask] #query
    # matches_im1 = matches_im1[conf_mask] #map

    # Create a mask for the union of top matches from both images
    conf_mask = np.zeros(len(matches_im0), dtype=bool)
    conf_mask[top_matches_im0] = True
    conf_mask[top_matches_im1] = True

    # Apply the mask to filter matches
    filtered_matches_im0 = matches_im0[conf_mask]
    filtered_matches_im1 = matches_im1[conf_mask]

    print("Number of matches after confidence mask: ", filtered_matches_im0.shape[0])

    print("Lowest Confidence Value: ", lowest_confidence_im0, lowest_confidence_im1)

    #print("Normalization: ", np.max(normalized_conf_im0), np.min(normalized_conf_im0))
    

    if visualizeMatches:
        makeHistogram(match_conf_im0,match_conf_im1,lowest_confidence_im0,lowest_confidence_im1)



    return filtered_matches_im0,filtered_matches_im1,matches_im0, matches_im1, pts3d_im0, pts3d_im1, conf_im0, conf_im1, desc_conf_im0, desc_conf_im1


def scale_intrinsics(K: np.ndarray, prev_w: float, prev_h: float, master_w: float, master_h: float) -> np.ndarray:
    """Scale the intrinsics matrix by a given factor .

    Args:
        K (NDArray): 3x3 intrinsics matrix
        scale (float): Scale factor

    Returns:
        NDArray: Scaled intrinsics matrix
    """
    #3024x4032 --> 384x512

    assert K.shape == (3, 3), f"Expected (3, 3), but got {K.shape=}"

    scale_w = master_w / prev_w  # sizes of the images in the Mast3r dataset
    scale_h = master_h / prev_h  # sizes of the images in the Mast3r dataset

    K_scaled = K.copy()
    K_scaled[0, 0] *= scale_w
    K_scaled[0, 2] *= scale_w
    K_scaled[1, 1] *= scale_h
    K_scaled[1, 2] *= scale_h

    return K_scaled

def CameraMatrix(fx,fy,cx,cy):
    return np.array([[fx,  0, cx],
                     [ 0, fy, cy],
                     [ 0,  0, 1]])

def cameraMatrixMapillary(focal,width,height): #converting open sfm intrinsics to standard
    K = np.array([ [focal * width, 0, width / 2],
      [0, focal * height, height / 2],
      [0, 0, 1] ])
    
    return K

def run_pnp(pts2D, pts3D, K, distortion=None): 
    """
    intrinsics= K

    mode='cv2'
    """

    # print("pts3D shape:", pts3D.shape)
    # print("pts2D shape:", pts2D.shape)

    success, r_pose, t_pose, _ = cv2.solvePnPRansac(pts3D, pts2D, K, distortion, flags=cv2.SOLVEPNP_SQPNP,
                                                    iterationsCount=10_000,
                                                    reprojectionError=3,
                                                    confidence=0.9999) #returns 3d to 2d transfromation, known to unknown 
    if not success:
        print("Failed to find transform")
        return False, None
    r_pose = cv2.Rodrigues(r_pose)[0]  # world2cam == world2cam2
    RT = np.r_[np.c_[r_pose, t_pose], [(0,0,0,1)]] # world2cam2 #anchor to query

    return True, np.linalg.inv(RT)  # cam2toworld #query to world

def get_rotation_from_compass(compass_angle):
    """ Create a rotation matrix based on compass angle (in radians). """
    return np.array([
        [np.cos(compass_angle), -np.sin(compass_angle), 0],
        [np.sin(compass_angle), np.cos(compass_angle), 0],
        [0, 0, 1]
    ])

def pnp_to_relative_global_coords(pnp_rotation, pnp_translation, ref_lat, ref_lon, compass_angle, ref_alt=0):
    # Define the reference point in UTM coordinates
    utm_proj = Proj(proj='utm', zone=getUTMzone(ref_lon), ellps='WGS84') 
    ref_x, ref_y = utm_proj(ref_lon, ref_lat)

    # Convert the rotation matrix to a 3x3 matrix if it's a vector
    if pnp_rotation.shape == (3,):
        R, _ = cv2.Rodrigues(pnp_rotation)
    else:
        R = pnp_rotation

    #print("Rotation: ", R, " Translation: ", pnp_translation )

    # The transformation we have is from known to unknown camera 
    compass_rotation = get_rotation_from_compass(np.deg2rad(compass_angle))
    T_world_to_camera = np.eye(4)
    T_world_to_camera[:3, :3] = compass_rotation[:3, :3]
    T_world_to_camera[:3, 3] = np.array([ref_x, ref_y, 0])
    R_cam_to_world = np.array([[1, 0, 0],
                               [0, 0, 1],
                               [0, -1, 0]])

    query_camera_in_anchor_frame = R_cam_to_world  @ pnp_translation

    query_camera_position = np.linalg.inv(compass_rotation) @  query_camera_in_anchor_frame


    # Add this position to the reference UTM coordinates
    new_x = ref_x + query_camera_position[0]
    new_y = ref_y + query_camera_position[1]
    global_alt = ref_alt + query_camera_position[2]

    # Transform back to latitude and longitude
    global_lon, global_lat = utm_proj(new_x, new_y, inverse=True)


    return global_lat, global_lon, global_alt

def getUTMzone(longitude):
    return int((longitude + 180) / 6) + 1



def getImageFromIndex(index, image_folder):
    filename = image_folder + '/metadata.csv'
    with open(filename, 'r', newline='') as csvfile:
        csvreader = csv.DictReader(csvfile)
        
        for row in csvreader:
            if row['id'] == str(index):
                image_path = os.path.join(image_folder, row['image_name'])
                return row, image_path
        
        return None  # ID not found

def getSequenceImageFromIndex(image_id, image_folder): #for mapillary data
    filename = image_folder + 'metadata.csv'
    with open(filename, 'r', newline='') as csvfile:
        csvreader = csv.DictReader(csvfile)
        
        for row in csvreader:
            if row['id'] == str(image_id):
                image_path = os.path.join(image_folder, f"{row['id']}.jpg")
                return row, image_path
        
        return None  # ID not found
    


def plotImages(image_indices,image_folder, rotate=False, title=None):
    # Get image paths
    image_paths = [getImageFromIndex(index, image_folder)[1] for index in image_indices]
    
    # Open images
    images = [Image.open(path) for path in image_paths]
    
    # Create figure
    fig, axes = plt.subplots(1, len(images), figsize=(20, 5))
    if len(images) == 1:
        axes = [axes]
    
    # Plot images
    for ax, img, index in zip(axes, images, image_indices):
        if rotate:
            img = img.rotate(-90, expand=True)
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f'Image {index}')
    
    # Set main title if provided
    if title:
        fig.suptitle(title, fontsize=16)
    
    plt.tight_layout()
    plt.show()


def shiftOrigin(points, x, y):
    return [[p[0] + x, p[1] + y] for p in points]


def get_image_html(img_path, width=50, rotate=True):
    # Load and resize the image
    with Image.open(img_path) as img:
        if rotate:
            img = img.rotate(-90, expand=True)
        img.thumbnail((width, width))
        # Convert the image to base64
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return f'<img src="data:image/jpeg;base64,{img_str}" width="{width}" height="{width}">'



def plot_images_on_map(csv_path, image_folder,pin_locations,visualizePins=True, visualizeImages=True,output_map='map.html'):
    # Load data from CSV
    data = pd.read_csv(csv_path)
    
    # Initialize the map centered on average coordinates
    avg_lat = data['lat'].mean()
    avg_lon = data['long'].mean()
    folium_map = folium.Map(location=[avg_lat, avg_lon], zoom_start=12)

    # Initialize a marker cluster
    # marker_cluster = MarkerCluster().add_to(folium_map)


    if visualizePins:
        # Add the custom pins to the map
        for index, location in enumerate(pin_locations):
            folium.Marker(
                location=location,
                popup=f'Index: {index}<br>Location: {location}',  # Customized popup with index
                tooltip=f'Pin {index}',  # Tooltip showing index on hover
                icon=folium.Icon(color='red', icon='info-sign')  # Red icon with an 'i' symbol
            ).add_to(folium_map)

    if visualizeImages:        
        # Loop through each image data entry
        for _, row in data.iterrows():
            # Get the file path for the image
            image_path = os.path.join(image_folder, f"{row['image_name']}")
            lat, lon = row['lat'], row['long']
            image_id = row['id']
            image_name = row['image_name']
            orientation = row['orientation']
            # Create the HTML for the image
            image_html = get_image_html(image_path)

            folium.Marker(
                location = [lat,lon],
                popup=folium.Popup(image_html, max_width=300),
                icon=folium.Icon(color='blue'),
                tooltip=image_id
            ).add_to(folium_map)

        # Create a legend HTML
    legend_html = """
    <div style="position: fixed; 
                bottom: 50px; left: 50px; 
                width: 180px; height: auto; 
                z-index:9999; font-size:14px; 
                background-color:white; 
                border:2px solid grey; 
                padding: 10px;">
        <b>Legend</b><br>
        <i class="fa fa-map-marker" style="color:blue"></i>&nbsp; Images<br>
        <i class="fa fa-map-marker" style="color:red"></i>&nbsp; Manually Sorted Locations<br>
    </div>
    """

    # Add the legend to the map using DivIcon
    folium_map.get_root().html.add_child(folium.Element(legend_html))

    # Save the map to an HTML file
    folium_map.save(output_map)
    print(f"Map saved as {output_map}")
    display(folium_map)
    
# Function to create an arrow marker
def create_orientation_arrow(location, angle, color='blue'):
    # Unicode arrow pointing north by default
    arrow_symbol = "↑"
    
    # CSS to rotate the arrow based on the orientation angle
    html = f"""
    <div style="
        transform: rotate({angle}deg);
        font-size: 30px;
        color: {color};
        width: 30px;
        height: 30px;
        text-align: center;
        line-height: 1.5;
    ">{arrow_symbol}</div>
    """
    return folium.Marker(
    location=location,
    icon=folium.DivIcon(html=html))

def plotImages(image_indices,image_folder, title=None):
    # Get image paths
    image_paths = [getImageFromIndex(index, image_folder)[1] for index in image_indices]
    
    # Open images
    images = [Image.open(path) for path in image_paths]
    
    # Create figure
    fig, axes = plt.subplots(1, len(images), figsize=(20, 5))
    if len(images) == 1:
        axes = [axes]
    
    # Plot images
    for ax, img, index in zip(axes, images, image_indices):
        img = img.rotate(-90, expand=True)
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f'Image {index}')
    
    # Set main title if provided
    if title:
        fig.suptitle(title, fontsize=16)
    
    plt.tight_layout()
    plt.show()