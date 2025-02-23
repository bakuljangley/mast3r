#the plotting code is not yet finalized honestly -- i'm adding all functions here to clean up the notebooks

from sevenScenesDatasets import loadPose7scenes
import matplotlib.pyplot as plt
import numpy as np
from utils import computePoseError
from scipy.ndimage import gaussian_filter1d
from matplotlib.colors import LogNorm
import ipywidgets as widgets
from ipywidgets import interact

def plotSceneResults(scene_indices, pair_dataset, results, title=None):
    #function uses a custom load ground truth function
    #the scene_indices are ordered in a way to ensure queries arent plot 9 times
    #expects the results to work in the results dataset way
    #log normalization -- over n_matches (in colors) 

    scene_indices = np.array(scene_indices).reshape(int(len(scene_indices)/9), 9)
    n_matches = results.n_matches_total
    
    # Initialize lists to store coordinates and n_matches
    gt_x, gt_z = [], []  # Ground truth
    est_x, est_z = [], []  # MAST3R estimations
    est_n_matches = []  # Number of matches for each estimation

    # Iterate through all queries in the scene
    for queryframes in scene_indices:
        gt_pose = loadPose7scenes(pair_dataset[queryframes[0]]['query_path'])
        gt_x.append(gt_pose[0, 3])
        gt_z.append(gt_pose[2, 3])

        for frame in queryframes:
            mast3r_pose = results.getPairResults(frame)['mast3r_q2world']
            if mast3r_pose is not None:
                est_x.append(mast3r_pose[0, 3])
                est_z.append(mast3r_pose[2, 3])
                est_n_matches.append(n_matches[frame])

    # Normalize the number of matches using log normalization
    est_n_matches = np.array(est_n_matches)
    normalized_matches = np.log1p(est_n_matches)  # log1p(x) = log(1 + x)
    normalized_matches = (normalized_matches - normalized_matches.min()) / (normalized_matches.max() - normalized_matches.min())

    # Create the scatter plot
    plt.figure(figsize=(10, 10))
    
    # Plot MAST3R estimations with colormap
    scatter = plt.scatter(est_x, est_z, c=normalized_matches, s=10, alpha=0.5, 
                          cmap='Spectral', label='MAST3R Estimation')

    # Plot ground truth
    plt.scatter(gt_x, gt_z, c='r', s=5, alpha=1, label='Ground Truth Query Position')
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Normalized Log Number of Matches')

    plt.title(f'Query Poses for Scene {title}')
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.xlim(-2.5, 2.5)
    plt.ylim(-2.5, 2.5)

    # Add text for min and max number of matches
    plt.text(0.02, 0.98, f'Min matches: {est_n_matches.min()}', transform=plt.gca().transAxes, verticalalignment='top')
    plt.text(0.02, 0.94, f'Max matches: {est_n_matches.max()}', transform=plt.gca().transAxes, verticalalignment='top')

    plt.show()


def plotLocalizationVsMatches(scene_indices, pair_dataset, results, title=None):
    scene_indices = np.array(scene_indices).reshape(int(len(scene_indices)/9), 9)
    n_matches = results.n_matches_total
    
    # Initialize lists to store data
    est_n_matches = []
    localization_errors = []
    x_errors, y_errors, z_errors = [], [], []

    # Iterate through all queries in the scene
    for queryframes in scene_indices:
        gt_pose = loadPose7scenes(pair_dataset[queryframes[0]]['query_path'])

        for frame in queryframes:
            mast3r_pose = results.getPairResults(frame)['mast3r_q2world']
            if mast3r_pose is not None:
                est_n_matches.append(n_matches[frame])
                pos_error, _ = computePoseError(mast3r_pose, gt_pose)
                localization_errors.append(pos_error)
                
                # Calculate individual x, y, z errors
                x_errors.append(abs(mast3r_pose[0, 3] - gt_pose[0, 3]))
                y_errors.append(abs(mast3r_pose[1, 3] - gt_pose[1, 3]))
                z_errors.append(abs(mast3r_pose[2, 3] - gt_pose[2, 3]))

    est_n_matches = np.array(est_n_matches)
    localization_errors = np.array(localization_errors)
    x_errors, y_errors, z_errors = np.array(x_errors), np.array(y_errors), np.array(z_errors)

    # Create the scatter plot
    plt.figure(figsize=(10, 6))
    
    # Color-code points based on their error using a logarithmic scale
    scatter = plt.scatter(est_n_matches, localization_errors, c=localization_errors, s=5, alpha=0.4, 
                          cmap='viridis', norm=LogNorm(), label='Estimations')

    # Add colorbar for error
    cbar = plt.colorbar(scatter)
    cbar.set_label('Localization error (m)')

    # Add moving average trendline for overall error
    sorted_indices = np.argsort(est_n_matches)
    sorted_matches = est_n_matches[sorted_indices]
    sorted_errors = localization_errors[sorted_indices]
    smoothed_errors = gaussian_filter1d(sorted_errors, sigma=50)
    plt.plot(sorted_matches, smoothed_errors, color='red', linewidth=2, label='Overall Trend')

    # Add trendlines for x, y, z errors
    smoothed_x_errors = gaussian_filter1d(x_errors[sorted_indices], sigma=50)
    smoothed_y_errors = gaussian_filter1d(y_errors[sorted_indices], sigma=50)
    smoothed_z_errors = gaussian_filter1d(z_errors[sorted_indices], sigma=50)
    
    plt.plot(sorted_matches, smoothed_x_errors, color='cyan', linewidth=1.5, linestyle='--', label='X Error Trend')
    plt.plot(sorted_matches, smoothed_y_errors, color='magenta', linewidth=1.5, linestyle='--', label='Y Error Trend')
    plt.plot(sorted_matches, smoothed_z_errors, color='yellow', linewidth=1.5, linestyle='--', label='Z Error Trend')

    plt.title(f'Localization Error vs Number of Matches for Scene {title}')
    plt.xlabel('Number of Matches')
    plt.ylabel('Localization Error (m)')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')  # Use log scale for error to better visualize the range

    # Add text for statistics
    plt.text(0.02, 0.98, f'Min matches: {est_n_matches.min()}', transform=plt.gca().transAxes, verticalalignment='top')
    plt.text(0.02, 0.94, f'Max matches: {est_n_matches.max()}', transform=plt.gca().transAxes, verticalalignment='top')
    plt.text(0.02, 0.90, f'Mean error: {np.mean(localization_errors):.3f}m', transform=plt.gca().transAxes, verticalalignment='top')
    plt.text(0.02, 0.86, f'Median error: {np.median(localization_errors):.3f}m', transform=plt.gca().transAxes, verticalalignment='top')


pos_thresholds = [0.25, 0.5]
rot_thresholds = [5, 10]

def evaluateScene(indices, pair_dataset, results, pos_thresholds=pos_thresholds, rot_thresholds=rot_thresholds, confidence_threshold=0):
    scene_indices = np.array(indices).reshape(int(len(indices)/9), 9)

    total_estimations = len(scene_indices) * 9  # 9 estimations per query
    accuracies = {f"{p}m_{r}deg": 0 for p in pos_thresholds for r in rot_thresholds}
    complete_fail_count = 0
    below_threshold_count = 0
    above_threshold_count = 0
    pos_errors = []
    rot_errors = []

    for queryframes in scene_indices:
        gt_pose = loadPose7scenes(pair_dataset[queryframes[0]]['query_path'])
        
        for frame in queryframes:
            mast3r_output = results.getPairResults(frame)
            mast3r_pose = mast3r_output['mast3r_q2world']
            n_matches = mast3r_output['n_matches_total']
            ret_val = mast3r_output['ret_val']

            if not ret_val:
                complete_fail_count += 1
            elif n_matches < confidence_threshold:
                below_threshold_count += 1
            else:
                above_threshold_count += 1
                if mast3r_pose is not None:
                    pos_error, rot_error = computePoseError(mast3r_pose, gt_pose)
                    pos_errors.append(pos_error)
                    rot_errors.append(rot_error)
                    for pos_thresh in pos_thresholds:
                        for rot_thresh in rot_thresholds:
                            if pos_error <= pos_thresh and rot_error <= rot_thresh:
                                accuracies[f"{pos_thresh}m_{rot_thresh}deg"] += 1

    # Calculate percentages
    percent_complete_fail = (complete_fail_count / total_estimations) * 100
    percent_below_threshold = (below_threshold_count / total_estimations) * 100
    percent_above_threshold = (above_threshold_count / total_estimations) * 100

    # Calculate accuracies for estimations above threshold
    for key in accuracies:
        accuracies[key] = (accuracies[key] / above_threshold_count) * 100 if above_threshold_count > 0 else 0

    # Calculate mean errors
    mean_pos_error = np.mean(pos_errors) if pos_errors else 0
    mean_rot_error = np.mean(rot_errors) if rot_errors else 0

    return {
        **accuracies,
        'percent_complete_fail': percent_complete_fail,
        'percent_below_threshold': percent_below_threshold,
        'percent_above_threshold': percent_above_threshold,
        'above_threshold_count': above_threshold_count,
        'total_estimations': total_estimations,
        'mean_pos_error': mean_pos_error,
        'mean_rot_error': mean_rot_error
    }


import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

def createSceneHistogram(results, dataset, indices, bin_width=50, fail_error=-1.0, title=None):

    n_matches = []
    pos_errors = []

    
    for i in indices:
        pair_result = results.getPairResults(i)
        n_matches.append(pair_result['n_matches_total'])
        
        mast3r_pose = pair_result['mast3r_q2world']
        gt_pose = loadPose7scenes(dataset[i]['query_path'])
        
        if mast3r_pose is not None: 
            pos_error, _ = computePoseError(mast3r_pose, gt_pose)
        else:
            pos_error = fail_error  # Negative value for failed localizations
        
        pos_errors.append(pos_error)

    n_matches = np.array(n_matches)
    pos_errors = np.array(pos_errors)

    max_matches = max(n_matches)
    bins = np.arange(0, max_matches + bin_width, bin_width)

    mean_errors = []
    fails_per_bin = []
    successes_per_bin = []

    for i in range(len(bins) - 1):
        mask = (n_matches >= bins[i]) & (n_matches < bins[i+1])
        errors_in_bin = pos_errors[mask]
        successful_errors = errors_in_bin[errors_in_bin >= 0]
        mean_error = np.mean(successful_errors) if len(successful_errors) > 0 else 0
        mean_errors.append(mean_error)
        fails_per_bin.append(np.sum(errors_in_bin < 0))
        successes_per_bin.append(np.sum(errors_in_bin >= 0))

    # Create the stacked histogram
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax2 = ax1.twinx()

    bin_centers = (bins[:-1] + bins[1:]) / 2
    width = bin_width * 0.8  # Adjust bar width

    ax1.bar(bin_centers, fails_per_bin, width, label='Fails', color='red', alpha=0.7)
    ax1.bar(bin_centers, successes_per_bin, width, bottom=fails_per_bin, label='Successes', color='green', alpha=0.7)

    ax1.set_xlabel('Number of Matches')
    ax1.set_ylabel('Count')
    ax1.legend(loc='upper left')

    # Calculate trend line
    valid_indices = pos_errors >= 0
    sorted_indices = np.argsort(n_matches[valid_indices])
    sorted_matches = n_matches[valid_indices][sorted_indices]
    sorted_errors = pos_errors[valid_indices][sorted_indices]
    smoothed_errors = gaussian_filter1d(sorted_errors, sigma=50)

    ax2.plot(sorted_matches, smoothed_errors, color='blue', linewidth=2, label='Error Trend')
    ax2.set_ylabel('Localization Error (m)', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    ax2.set_yscale('log')  # Set y-axis to logarithmic scale

    # Add legend for trend line
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(ax1.get_legend_handles_labels()[0] + lines2, ax1.get_legend_handles_labels()[1] + labels2, loc='upper left')

    plt.title(f'Stacked Histogram of Localization Results with Error Trend - {title.capitalize()}')
    plt.tight_layout()
    plt.show()




def plot_arrow(ax, pose, color, label):
    position = pose[:3, 3]
    direction = pose[:3, 2]

    ax.quiver(position[0], position[2], direction[0], direction[2],
              color=color,
              scale=4,         # Adjust for longer arrows
              scale_units='inches',  # Crucial for consistent scaling
              headwidth=8,      # Adjust for slimmer head width, was 8
              headlength=8,    # Adjust for shorter head length, was 10
              headaxislength=5,  # Adjust to control the fineness of the arrow tip, was 5
              alpha = 0.5,
              label=label)

def plotQuery(queryNeighbourhood, dataset, results): #this will plot a query and all 9 of it's anchors
    #queryNeighbourhood is the list of all indices that localize the query -- so we have all query anchor frames
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal', adjustable='box')

    #plot the query pose
    query_pose = loadPose7scenes(dataset[queryNeighbourhood[0]]['query_path']) #load query location from the first frame in queryFrames
    ax.scatter(query_pose[0, 3], query_pose[2, 3], c='r', s=50, label='True Query Position')
    plot_arrow(ax, query_pose, 'r', label='')

    #plot all anchors 
    anchor_x, anchor_z = [], []
    estimated_x, estimated_z = [], []
    for i,anchor in enumerate(queryNeighbourhood):
        
        #plot anchor pose
        anchor_pose = loadPose7scenes(dataset[anchor]['anchor_path'])
        plot_arrow(ax, anchor_pose, 'b', label='') #, 'Anchor Direction' if i == 0 else ''
        anchor_x.append(anchor_pose[0, 3])
        anchor_z.append(anchor_pose[2, 3])

        ax.text(anchor_pose[0, 3]+0., anchor_pose[2, 3], str(anchor), fontsize=10, ha='right', va='bottom')
        ax.text(anchor_pose[0, 3]+0., anchor_pose[2, 3]-0.1, str(results.getPairResults(anchor)['n_matches_total']), fontsize=10, color='blue', ha='right', va='bottom')

        #plot mast3r estimated query pose
        mast3r_q2world = results.getPairResults(anchor)['mast3r_q2world']
        if mast3r_q2world is not None:
            ax.plot([anchor_pose[0, 3], mast3r_q2world[0, 3]],
                    [anchor_pose[2, 3], mast3r_q2world[2, 3]], 
                    'k--', linewidth=0.7, alpha=0.3)
            # plot_arrow(ax, mast3r_q2world, 'g', 'Estimated Direction' if i == 0 else '')
            
            estimated_x.append(mast3r_q2world[0, 3])
            estimated_z.append(mast3r_q2world[2, 3])
        else:
            ax.text(anchor_pose[0, 3], anchor_pose[2, 3] + 0.1, "Fail", fontsize=10,color='red', ha='right', va='bottom')
    ax.scatter(anchor_x, anchor_z, c='b', s=25, label='True Anchor Position')
    ax.scatter(estimated_x, estimated_z, c='g', s=25, label='MASt3R Estimated Query Position')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    # ax.set_xlim(-3, 3)
    # ax.set_ylim(-3, 3)
    ax.legend()
    ax.grid(True)
    ax.set_title(f'Query and All Anchors with Corresponding Localizations')
    plt.show()




