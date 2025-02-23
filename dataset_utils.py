from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from PIL import Image

import ipywidgets as widgets
from IPython.display import display


class ImagePairResults(Dataset):  #renamed to image pair results, can support additional plotting functionality
    #this can be edited to store the n_matches, etc -- or maybe i just have a results class?? 

    
    #one item is essentially a pair of images, works with processer to get localization results
    def __init__(self, anchor_paths, query_paths, scene_ids, mast3r_transform, mast3r_pose):
        assert len(anchor_paths) == len(query_paths) == len(mast3r_transform)==len(mast3r_pose), "All inputs must have the same length"
        self.anchor_paths = anchor_paths 
        self.query_paths = query_paths
        self.mast3r_q2a = mast3r_transform
        self.mast3r_q2world = mast3r_pose
        self.scene_ids = scene_ids

    def __len__(self):
        return len(self.anchor_paths)

    def __getitem__(self, idx):
        item = {
            'anchor_path': self.anchor_paths[idx],
            'query_path': self.query_paths[idx],
            'scene_id': self.scene_ids[idx],
            'mast3r_q2a': self.mast3r_q2a[idx],
            'mast3r_q2world': self.mast3r_q2world[idx]
        }
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

    def getFails(self):
        #return indices of failed queries
        return [idx for idx, item in enumerate(self) if item['mast3r_q2a'] is None]

    def visualizeFails(self):
        fails = self.getFails()

        if len(fails) == 0:
            print("No Failed Queries")
        else:
            output = widgets.Output()

            def update(index):
                output.clear_output(wait=True)
                with output:
                    idx = fails[index]
                    self.plotItem(idx)  # Use plotItem method here
                    item = self[idx]
                    print(f"Dataset Index: {idx}, Scene: {item['scene_id']}")

            slider = widgets.IntSlider(value=0, min=0, max=len(fails)-1, step=1, description='Fail:')
            
            widgets.interactive(update, index=slider)
            
            display(widgets.VBox([slider, output]))
