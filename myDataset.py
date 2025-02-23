
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import ipywidgets as widgets
from IPython.display import display



class ImagePairDataset(Dataset):
    #one item is essentially a pair of images --> geared towards finding the anchor to query transform
    #ISSUE : right now the code makes no use of the results loaded here, maybe i should change this?
    def __init__(self, anchor_paths, query_paths, mast3r_output=None):
        assert len(anchor_paths) == len(query_paths), "All inputs must have the same length"
        self.anchor_paths = anchor_paths
        self.query_paths = query_paths
        self.mast3r_output = self.mast3r_output if mast3r_output else None

    def __len__(self):
        return len(self.anchor_paths)

    def __getitem__(self, idx):
        anchor_path = self.anchor_paths[idx]
        query_path = self.query_paths[idx]

        item = {
            'anchor_path': anchor_path,
            'query_path': query_path,
        }

        if self.mast3r_output: #if master output then read query to world transform, and query to anchor transform
            mast3r_result = self.mast3r_output.get((query_path, anchor_path))
            item['mast3r_transform'] = mast3r_result['transform'] if mast3r_result else None
            item['mast3r_pose'] = mast3r_result['pose'] if mast3r_result else None

        return item
    
    def plotItem(self,idx):
        item = self[idx]
        anchor = Image.open(item['anchor_path'])
        query = Image.open(item['query_path'])
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 4))
        
        ax1.imshow(anchor)
        ax1.set_title('Anchor Image')
        ax1.axis('off')
        
        ax2.imshow(query)
        ax2.set_title('Query Image')
        ax2.axis('off')
        
        plt.tight_layout()
        plt.show()

    def visualizePairs(self, indices): 
        #indices == list of indices to plot
        #plots an interactive slider with anchor, query pair list

        if len(indices) == 0:
            print("No Failed Queries")
        else:
            output = widgets.Output()

            def update(index):
                output.clear_output(wait=True)
                with output:
                    idx = indices[index]
                    self.plotItem(idx)  # Use plotItem method here
                    item = self[idx]
                    print(f"Dataset Index: {idx}")

            slider = widgets.IntSlider(value=0, min=0, max=len(indices)-1, step=1, description='Fail:')
            
            widgets.interactive(update, index=slider)
            
            display(widgets.VBox([slider, output]))


class ResultsDataset:
    def __init__(self):
        self.n_matches_total = []
        self.n_matches_filtered = []
        self.ret_vals = []
        self.transformations = []
        self.query2world_poses = []

    def add_result(self, n_total, n_filtered, ret_val, transformation, query2world_pose):
        self.n_matches_total.append(n_total)
        self.n_matches_filtered.append(n_filtered)
        self.ret_vals.append(ret_val)
        self.transformations.append(transformation)
        self.query2world_poses.append(query2world_pose)

    def __len__(self):
        return len(self.n_matches_total)

    def getPairResults(self, index):
        return {
            'n_matches_total': self.n_matches_total[index],
            'n_matches_filtered': self.n_matches_filtered[index],
            'ret_val': self.ret_vals[index],
            'mast3r_q2a': self.transformations[index],
            'mast3r_q2world': self.query2world_poses[index]
        }

    def printSummary(self):
        print(f"Total number of pairs processed: {len(self)}")
        print(f"Number of successful matches: {sum(self.ret_vals)}")
        print(f"Average number of total matches: {np.mean(self.n_matches_total):.2f}")
        print(f"Average number of filtered matches: {np.mean(self.n_matches_filtered):.2f}")


    def getFails(self):
        """Return indices where mast3r_q2world is None (i.e., localization failed)."""
        return [i for i, pose in enumerate(self.query2world_poses) if pose is None]

    def getMatchesBelow(self, threshold):
        """Return indices where the number of total matches is below the threshold."""
        return [i for i, n_matches in enumerate(self.n_matches_total) if n_matches < threshold]
    
    def getMatchesWithin(self, lower_bound, upper_bound):
        """
        Return indices where the number of total matches is between lower_bound and upper_bound (inclusive).
        """
        return [i for i, n_matches in enumerate(self.n_matches_total) 
                if lower_bound <= n_matches <= upper_bound]


def readResultsFile(file_path):
    #reads the output results file and saves it in order into a dataset with the item being all logged outputs
    results = ResultsDataset()

    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            n_total = int(parts[2])
            n_filtered = int(parts[3])
            ret_val = parts[4].lower() == 'true'



            if ret_val:
                transformation = np.array([float(x) for x in parts[5:21]]).reshape(4, 4)
                query2world_pose = np.array([float(x) for x in parts[21:37]]).reshape(4, 4)
                # Check if the matrices are all zeros, and set to None if they are
                transformation = None if np.all(transformation == 0) else transformation
                query2world_pose = None if np.all(query2world_pose == 0) else query2world_pose

            else:
                transformation = None
                query2world_pose = None

            results.add_result(n_total, n_filtered, ret_val, transformation, query2world_pose)

    return results

