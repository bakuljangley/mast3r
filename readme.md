# Introduction

This repository is a fork from the [original codebase for MASt3R by Naver Labs](https://github.com/naver/mast3r). My main aim is to use MASt3R for relative pose estimation given two input images.


#### Installation instructions and other required downloads:
1. **MASt3R**: Follow the installation instructions provided in the original code base to set up an environment for MASt3R. I use the `MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric` with resolution 512x384. 
2. [**7scenes**:](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/) Download the entire dataset with the resolution (640x580)


## To add to readme
1. Details on how to use PnP for relative pose estimation
2. Details on how to use Mast3r for relative pose estimation
3. Details on how my functionality works -- classes, functions, visualizations 
4. Add a example notebook for finding the transformation given two input images

## Current Bugs
1. Mast3r visualizer with confidence scores does not account for the fact that the confidence maps -- on the images wouln't be common. Confidence values of one pointmap do not mean anything to the either.
2. Intrinsics scaling isn't dynamically finding mast3r resolution -- there is some logic to sort this out according to image dimensions 



   




 

