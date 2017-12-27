# Semantic-Segmentation

Term 3 out of 3
Project 2

![road_gif](https://github.com/JLee21/Semantic-Segmentation/blob/master/img/road_social.gif)


### Hyperparmeters

High-level tuning paramters such as number of epochs and learning rate can be found in `config.py`

### Training Progress

A few things to note:
I found that it is too easy to exhaust the GPU memory of a GTX 1050 while implementing a Fully Convoultional Network.
With this in mind, the loss value is computed on the same single image and the Mean_IOU accuracy is computed on the same batch of 10 images. To avoid this potential bias in the future, it would be ideal to increase the working memory of the GPU. Also, randoming choosing the sample images would help generalize the progress (accuracy/loss) the model makes.
