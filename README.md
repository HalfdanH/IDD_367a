# INF367 project: Identifying Deforestation Drivers


### Competition Description:
The competition is based on identifying deforestation drivers using image segmentation and IoU for the metric.
The images are from the Sentinel-2 satelite dataset during the period: 2017/1/1 to 2024/6/1 and containing 12 bands: 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12'.
It has 4 classes of deforestation: Looging, mining, grassland shrubland and plantations.
So for each image the goal is to predict the outline of the deforestastion segment, and get the best Mean IoU for the evaluation images.
[Link to competition](https://solafune.com/competitions/68ad4759-4686-4bb3-94b8-7063f755b43d?menu=data&tab=)


### Individual Implementations to Project:

**Band Selection Implementation:**
...

**SatMAE implementation:**
For implementing the encoder into the main pipeline, we define a class Encoder, then we train it in the autoencoder file, saving the weights, then adding it to a model we are going to train in the main pipeline as a before the original structure. The first thing we do is to

**SatSynth implementation:**
For full details on the SatSynth implementation, look for the `README.md` file in the `SatSynth_impl` folder.


Group Members:
- Tobias Husebø
- Lasse Holt
- Halfdan Hesthammer
