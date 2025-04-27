# INF367 project: Identifying Deforestation Drivers


### Competition Description:
The competition is based on identifying deforestation drivers using image segmentation and IoU for the metric.
The images are from the Sentinel-2 satelite dataset during the period: 2017/1/1 to 2024/6/1 and containing 12 bands: 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12'.
It has 4 classes of deforestation: Looging, mining, grassland shrubland and plantations.
So for each image the goal is to predict the outline of the deforestastion segment, and get the best Mean IoU for the evaluation images.
[Link to competition](https://solafune.com/competitions/68ad4759-4686-4bb3-94b8-7063f755b43d?menu=data&tab=)


### Individual Implementations to Project:

**Band Selection Implementation:**
The different steps of the band selection are split up into different functions. First, each 12-band image is downsampled and projected onto its top three PCA components to speed up spatial segmentation, then made into ~500 SLIC superpixels. We filter out tiny or mixed-class patches and, for each remaining superpixel, compute 13 Haralick texture features per original band—yielding one 156-dimensional descriptor per patch. These descriptors form a Train/Val/Test set, over which a simple evolutionary loop (UMDA) samples binary “which-band” vectors, trains a small RBF-SVM on Train, scores on Val via balanced accuracy, and updates its band-inclusion probabilities from the top performers. After 30 generations, we select bands whose appearance frequency exceeds the median, retrain on Train+Val, and report the final balanced accuracy on Test. The best set of bands are also saved and stored as a json file called "best_bands.json" inside the "BandSelec_impl" folder with their given indices. This makes it possible for us to extract the correct bands to our model. 

**SatMAE implementation:**
We train the masked autoencoder on the training images. We group the bands as RGB, NIR and SWIR as same embedding. We then mask 0.25 % of the patches. The weights for the encoder are then stored to be implemented into the main pipeline.

**SatSynth implementation:**
For full details on the SatSynth implementation, look for the `README.md` file in the `SatSynth_impl` folder.


Group Members:
- Tobias Husebø
- Lasse Holt
- Halfdan Hesthammer
