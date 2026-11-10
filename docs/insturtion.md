Workshop on Nov. 10:

In this workshop, we produce the classification baseline from the AffectNet paper on the freely available FER2013 dataset.

In the code, we have an ImageNet-pretrained model fine-tuned to classify expressions (the paper used AlexNet), a class-weighted loss for the imbalanced classes, 8 expression classes (with FER+ labels), and evaluation with accuracy and macro-F1. 

In the code, we don't have AffectNet's valence/arousal prediction. FER2013 has no valence/arousal labels.

Data: FER+ labels are downloaded automatically from Microsoft's FERPlus GitHub repository. They only work with the CSV version of FER2013. train.py downloads the original fer2013.tar.gz (about 92 MB) from a public Hugging Face mirror. 

***For all exercises, look at F1 as the main score and use accuracy for comparing with published papers.

*** The exercises take about 6–8 hours in total. Divide the work between you and your groupmate so you can run experiments in parallel.

Exercises:

1) Run the baseline model as follows. The baseline uses the ResNet18 as the pretrained model. (One Run)

!python train.py

2) Right now, 48×48 faces are enlarged to 128×128. Adapt the network to small images instead of enlarging them. Report the baseline results and compare them with the new results (One Run)

3) Find the best published results on FER2013 dataset. Try to improve the accuracy of this code with one or a combination of following methods. (You can come up with your own idea to improve the accuracy for at least 1-2 points) Report the baseline results and compare them with the new results and explain your proposed method.

Train for 50–100+ epochs with a better tuned learning rate
Add mixup/CutMix,
Keep a running average of the weights (EMA)
4) In this code, we load torchvision’s standard ImageNet weights: 

ResNet18_Weights.DEFAULT. ImageNet is about 1.2 million photos sorted into 1,000 everyday categories such as dog breeds, birds, vehicles, tools and food... People do appear in many of the photos, but “face” and “expression” aren’t among the categories, so the network was never trained to tell a smile from a frown.

a) Run the model without any pretrained network. (One Run)
b) Run the model with AlexNet (already implemented in the code). (One Run) 

!python train.py --model alexnet --epochs 30 

c) Compare the train loss, val loss, val acc, val macro-F1 during the training with AlexNet and ResNet18 as the pretrained models respectively and explain how results are different and why.

d) Find a network pretrained on face recognition or expression data to improve F1 and accuracy. Report the baseline results and compare them with the new results from 4a, 4b, and 4d. Explain which pre-trained network you use and why. 

5)  Predict the expression in your own photos in different moods with the best model that you got. For example, with resnet18 we have:

!python predict.py --checkpoint /content/drive/MyDrive/fer_runs/resnet18_ferplus_majority_128px_sqrt_inverse/best.pt yourphoto.jpg