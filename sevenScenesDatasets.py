
import os
import numpy as np
from PIL import Image

#dictionary to scene_id keys : defined the same in the relposenet codebase + this is how i read scene_ids for 7scenes 
scenes_dict = {i: scene for i, scene in enumerate(['chess', 'fire', 'heads', 'office', 'pumpkin', 'redkitchen', 'stairs'])}

#to divide the 7scenes dataset into scenes 

def getSceneIndices(scene_id, scene_ids):
    #scene_id : id of scene to find
    #scene_ids : list of all scene_ids for (anchor,query) pairs
    return np.array([i for i, sid in enumerate(scene_ids) if int(sid) == scene_id] )

def loadPose7scenes(image_path): #expects absolute image path
    pose_path = image_path.replace('.color.png', '.pose.txt')
    pose = np.loadtxt(pose_path)
    return pose

def loadDepth7scenes(image_path):
    depth_path = image_path.replace('.color.png', '.depth.png')
    depth = np.array(Image.open(depth_path))
    return depth

def readMultiImageRelPoseNetPairsFile(pairs_file, root_dir):
        #returns query_paths, anchor_paths, scene_ids for the pairs file 
        #custom made for pairs file
        scenes_dict = {i: scene for i, scene in enumerate(['chess', 'fire', 'heads', 'office', 'pumpkin', 'redkitchen', 'stairs'])}
        
        query_paths, anchor_paths, scene_ids_tmp = [], [], []
        with open(pairs_file, 'r') as f:
            for line in f:
                anchors = []
                chunks = line.strip().split(' ')
                scene_id = chunks[-1]
                
                scene_ids_tmp.append(scene_id)
                query_paths.append(os.path.join(root_dir, scenes_dict[int(scene_id)], chunks[0][1:]))
                for i in range(1,10):
                    anchors.append(os.path.join(root_dir, scenes_dict[int(scene_id)], chunks[8*i][1:]))
                anchor_paths.append(anchors)

        #convert to anchor, query pairs
        # final_index = 9*len(query_paths)
        anchors = []
        queries = []
        scene_ids = []
        for index in range(len(query_paths)):
            anchors.extend(anchor_paths[index])
            queries.extend([query_paths[index]] * len(anchor_paths[index]))
            scene_ids.extend([scene_ids_tmp[index]] * len(anchor_paths[index]))

        return anchors, queries, scene_ids