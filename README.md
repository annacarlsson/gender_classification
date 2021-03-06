# Gender classification using convolutional neural networks
### Our final project in the Chalmers course TMS016: Spatial statistics and image analysis

## Abstract
In this project, three different neural network architectures (MobileNet, Inception V3, and VGG16) have been trained to perform gender classification on facial images. All network architectures have been pretrained on the ImageNet database.

Two datasets have been used for training and network validation; the UTKFace dataset and the Adience dataset. Both datasets were split into 80/15/5 training/validation/test sets. Data augmentation (horizontal flips, rotation and shear) was used to increase the size of the dataset. Random search was used to select the best set of model hyperparameters.

The models were trained for 15 epochs with all layers but the top layers frozen, and then for additional 30 epochs with all layers unfreezed. This was done for both the UTKFace and Adience datasets. Then, the model which performed best in terms of test accuracy on the UTKFace dataset of each sort was selected and additionally trained on the Adience dataset using transfer learning.

The best performing model on the UTKFace dataset was the VGG16 architecture based, which reached a test accuracy of 94.4 \%. The same model also performed best in terms of test accuracy on the Adience dataset, with a test accuracy of 94.9 \%. The networks which were first trained on UTKFace and then trained on Adience did not perform nearly as good as the best network which was directly trained on Adience. In this case, the VGG16 architecture reached a maximal test accuracy of 88.1 \%. This suggests that the extra training step did not contribute with any significant features which was useful for the networks for classification of the images in the Adience dataset.

## Contact
Anton Andrée: <ananton@student.chalmers.se>  
Anna Carlsson: <annca@student.chalmers.se>  
Jan Liu: <lijan@student.chalmers.se>
