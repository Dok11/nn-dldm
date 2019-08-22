# NNCAP — Neural Network Complex Approach to Photogrammetry 
Neural network for photogrammetry and reconstruction 3d model from photos

Current neural network training performance

![progress](/docs/train-results.png)

# Roadmap

Work in progress but currently my plan includes this parts to realisation:

1. System to ordering images by similarity that was image pairs will be together
2. Reconstruct camera positions from image pairs (previous and current)
3. Create depth map based on current image and two most nearset images (or large if I will rich)
4. Convert are separately depth maps to common points cloud and filter them from trash points
5. Create mesh from points cloud and texture them

## 1. Ordering images

Main feature of this section is defining nearest images together for next step.  
The idea is comparing each image with any other or calulcate features of images and ordering images by them. Last way in prioritet because looks like more efficient.  
I think here we can use existed models what are use for CIFAR dataset and get flatten layer for estimate image fetures which we will compare in out network.

Currently status of this section — zero. Just idea.

## 2. Reconstruct camera positions or calculate camera deltas

Here we need define cameras position in 3d space to know from we will estimate depth maps.

Scripts for trainig placed in the `/src/camera_deltas/`  
Dataset placed in the `/data/camera_deltas/`  
Prepared data in npz file placed in the `/train_data/camera_deltas/`  
Saved model and their weights placed in the `/models/camera_deltas/`  

Currently status of this section — almost working demo with acceptable quality.

### Upcoming plans

1. Verify network quality on the synthetic data.
2. To generate dataset for each normal task (indoor photos, exterior, object shooting) with photorealistic images sith sizes around 450×300. And to include complex (for SIFT method) surfaces like shiny, mirrors, metallic.
Currently uses synthetic images like this:
![synt images](/docs/camera_deltas__synt-images.png)
3. To do augmentation methods like:
3.1. Camera lens distortion
3.2. Noise like from high ISO
3.3. Small motion blur
3.4. Gaussian blur (like defocus)
3.5. JPEG-glitches
3.6. Fly on camera lens
3.7. Different brightness/contrast/hue
4. Train network for fast result (like 120×90, bw)
5. Train network for strong result (lik 450×300, rgb)

## 3. Create depth map (bw image)

In this section we will get target image and two most nearest, then calculate depth map which looks like this:
![depth map example](/docs/depth_map__example.png)

Currently status of this section — ready method for creation dataset of images in Blender 3D.
![method for creation dataset with depth maps](/docs/depth_map__method.png)

## 4. Convert are separately depth maps to common points cloud

In this section we will scale depth map from relative it camera until it better matching from neighboring cameras.

## 5. Create mesh from points cloud and texture them

Classic method for creating mesh from poitns cloud.
