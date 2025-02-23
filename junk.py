##THIS IS OLD CODE (THAT I MIGHT NEED, maybe idk) NOT CURRENTLY USED IN ANY OF MY CODE -- 

#contains dataset class, load pose and load depth map functions for the 7scenes dataset


##generally imports ahead of every thing here are specific trash -- 








##################################################################################################################

#this is from sevenScenesDatasets.py -- old code to read the dataset and pairs file -- moved 23/02/2025


from torch.utils.data import Dataset
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class SevenScenesNBDataset(Dataset):
    def __init__(self, root_dir, pairs_file, mast3r_output = None):
        self.root_dir = root_dir #directory containing 7scsnes
        self.scenes_dict = {i: scene for i, scene in enumerate(['chess', 'fire', 'heads', 'office', 'pumpkin', 'redkitchen', 'stairs'])}
        
        self.query_paths, self.anchor_paths, self.scene_ids = self._read_pairs_txt(pairs_file)
        if mast3r_output:
            self.mast3r_q2a, self.mast3r_query_poses = self._read_results_txt(mast3r_output)
        else:
            self.mast3r_q2a, self.mast3r_query_poses = None, None
    def _read_pairs_txt(self, pairs_file):
        query_paths, anchor_paths, scene_ids = [], [], []
        with open(pairs_file, 'r') as f:
            for line in f:
                anchors = []
                chunks = line.strip().split(' ')
                scene_id = chunks[-1]
                
                scene_ids.append(scene_id)
                query_paths.append(os.path.join(self.root_dir, self.scenes_dict[int(scene_id)], chunks[0][1:]))
                for i in range(1,10):
                    anchors.append(os.path.join(self.root_dir, self.scenes_dict[int(scene_id)], chunks[8*i][1:]))
                anchor_paths.append(anchors)

        return query_paths, anchor_paths, scene_ids
    

    def _read_results_txt(self, results_file):
        mast3r_q2a = {}
        mast3r_query_poses = {}
        with open(results_file, 'r') as f:
            for line in f:
                chunks = line.strip().split(' ')
                query_path = chunks[0]
                transforms = []
                poses = []
                for i in range(9):  # 9 anchors
                    start_transform = 1 + i * 32
                    end_transform = start_transform + 16
                    start_pose = end_transform
                    end_pose = start_pose + 16
                    
                    transform = np.array([float(x) for x in chunks[start_transform:end_transform]]).reshape(4, 4)
                    pose = np.array([float(x) for x in chunks[start_pose:end_pose]]).reshape(4, 4)
                    
                    transforms.append(transform if not np.all(transform == 0) else None)
                    poses.append(pose if not np.all(pose == 0) else None)
                
                mast3r_q2a[query_path] = transforms
                mast3r_query_poses[query_path] = poses
        return mast3r_q2a, mast3r_query_poses
    
    def _load_pose(self, image_path):
        pose_path = image_path.replace('.color.png', '.pose.txt')
        pose = np.loadtxt(pose_path)
        return pose
    
    def __getitem__(self, idx):
        query_path = self.query_paths[idx]
        anchors_path = self.anchor_paths[idx]
        scene_id = self.scene_ids[idx]
        query_pose = self._load_pose(query_path)
        anchor_poses = []
        for anchor in anchors_path:
            anchor_poses.append(np.array(self._load_pose(anchor)))

        mast3r_q2a = self.mast3r_q2a[query_path] if self.mast3r_q2a else [None] * 9
        mast3r_query_pose = self.mast3r_query_poses[query_path] if self.mast3r_query_poses else [None] * 9

        return{
            'query_path': query_path,
            'anchors_path': anchors_path,
            'scene_id': scene_id,
            'gt_query_pose': np.array(query_pose),
            'anchor_poses': anchor_poses,
            'mast3r_q2a': mast3r_q2a,
            'mast3r_query_pose': mast3r_query_pose

        }
    def __len__(self):
        return len(self.query_paths)
    
    def show_query_and_anchors(self, idx, anchor_idx=None):
        data = self[idx]
        query_path = data['query_path']
        anchors_path = data['anchors_path']

        query_img = Image.open(query_path).convert('RGB')

        if anchor_idx is not None:
            # Display query and specific anchor
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))
            ax1.imshow(query_img)
            ax1.set_title('Query Image')
            ax1.axis('off')

            anchor_img = Image.open(anchors_path[anchor_idx]).convert('RGB')
            ax2.imshow(anchor_img)
            ax2.set_title(f'Anchor Image {anchor_idx}')
            ax2.axis('off')

        else:
            # Display query and all anchors
            fig, axes = plt.subplots(3, 4, figsize=(4, 3))
            axes = axes.flatten()

            axes[0].imshow(query_img)
            axes[0].set_title('Query Image')
            axes[0].axis('off')

            for i, anchor_path in enumerate(anchors_path):
                anchor_img = Image.open(anchor_path).convert('RGB')
                axes[i+1].imshow(anchor_img)
                axes[i+1].set_title(f'Anchor {i}')
                axes[i+1].axis('off')

            # Remove extra subplots
            for i in range(len(anchors_path) + 1, 12):
                fig.delaxes(axes[i])

        plt.tight_layout()
        plt.show()
    
    def get_query_anchor_pair(self, query_idx, anchor_idx):
        query_path = self.query_paths[query_idx]
        anchor_path = self.anchor_paths[query_idx][anchor_idx]
        anchor_pose = self.anchor_poses[query_idx][anchor_idx]
        return query_path, anchor_path, anchor_pose
    

##################################################################################################################