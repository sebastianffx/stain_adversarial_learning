This project investigates the interplay between color augmentation and adversarial feature learning to address the variability in histopathology images.
Here are the steps to reproduce each set of experiments:

Mitosis detection in TUPAC:
* Have in your local storage the TUPAC dataset
* Create the patches using the coordinates located in patches_coordinates.txt
* Run either baseline.py or dann_experiment.py

Gleason grading on Zurich prostate TMA dataset:
* Have in your local storage the TMA_images folder
* Separate the patches folder using patches_partitions.txt

Multi-organ nuclei segmentation on Kumar patch dataset:
* Have in your local storage the nuclei and 

