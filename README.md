This project investigates the interplay between color augmentation and adversarial feature learning to address the variability in tissue preparation methods that make substantial changes in the appearance of  digitized histology images and that hampers the performance of deep learning networks when applied to computational pathology tasks.

For this purpose, we design a domain adversarial framework in which color augmentation is used in conjunction with domain invariant training of deep convolutional networks. We test our approach in two open access datasets and provide the steps and code to reproduce each result reported in the paper.

Here are the steps to reproduce each set of experiments:

Mitosis detection in TUPAC:
* Have in a local path the ([TUPAC dataset](http://tupac.tue-image.nl/node/3))
* Create the patches using the coordinates located in datasets_utils/tupac

Gleason grading using the Zurich prostate TMA dataset and patches from diagnostic WSI of TCGA:
* Download the [dataset](https://wetransfer.com/downloads/b33c6eda5df597b2fe375a2162be535f20190719142214/25afc2d4546196eb08825d48316c0c8720190719142214/d04b56) (if you have problems downloading it, drop me an email to juan.otalora [AT] etu.unige.ch)

* Run either baseline.py or dann_experiment.py
