This project investigates the interplay between color augmentation and adversarial feature learning to address the variability in tissue preparation methods that make substantial changes in the appearance of  digitized histology images and that hampers the performance of deep learning networks when applied to computational pathology tasks.

For this purpose, we design a domain adversarial framework in which hard-color augmentation is used in conjunction with domain invariant training of the networks. We test our approach in three open access datasets and provide the steps and code to reproduce each result reported in the paper.

Here are the steps to reproduce each set of experiments:

Mitosis detection in TUPAC:
* Have in a local path the ([TUPAC dataset](http://tupac.tue-image.nl/node/3))
* Create the patches using the coordinates located in patches_coordinates.txt
* Run either baseline.py or dann_experiment.py

Gleason grading using the Zurich prostate TMA dataset and patches from diagnostic WSI of TCGA:
* Have in your local path the ([TMA images folder](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/OCYCMP)) 
* Separate the patches folder using patches_partitions.txt
