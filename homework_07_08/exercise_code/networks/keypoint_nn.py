"""Models for facial keypoint detection"""

import torch
import torch.nn as nn
import pytorch_lightning as pl

# TODO: Choose from either model and uncomment that line
# class KeypointModel(nn.Module):
class KeypointModel(pl.LightningModule):
    """Facial keypoint detection model"""
    def __init__(self, hparams):
        """
        Initialize your model from a given dict containing all your hparams
        Warning: Don't change the method declaration (i.e. by adding more
            arguments), otherwise it might not work on the submission server
            
        NOTE: You could either choose between pytorch or pytorch lightning, 
            by switching the class name line.
        """
        super().__init__()
        self.save_hyperparameters(hparams)
        ########################################################################
        # TODO: Define all the layers of your CNN, the only requirements are:  #
        # 1. The network takes in a batch of images of shape (Nx1x96x96)       #
        # 2. It ends with a linear layer that represents the keypoints.        #
        # Thus, the output layer needs to have shape (Nx30),                   #
        # with 2 values representing each of the 15 keypoint (x, y) pairs      #
        #                                                                      #
        # Some layers you might consider including:                            #
        # maxpooling layers, multiple conv layers, fully-connected layers,     #
        # and other layers (such as dropout or batch normalization) to avoid   #
        # overfitting.                                                         #
        #                                                                      #
        # We would really recommend to make your code generic, so that you     #
        # automate the calculation of the number of parameters at each layer.  #
        # You're probably going to try different architecutres, and that will  #
        # allow you to be quick and flexible.                                  #
        ########################################################################
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=6, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Dropout(p=0.2),
            
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.3),
        
            nn.Conv2d(64, 128, kernel_size=4, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.4),
        
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.5)
        )
        
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            
            nn.Linear(256 * 5 * 5, 500),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            
            nn.Linear(500, 30)
        )

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self, x):
        
        # check dimensions to use show_keypoint_predictions later
        if x.dim() == 3:
            x = torch.unsqueeze(x, 0)
        ########################################################################
        # TODO: Define the forward pass behavior of your model                 #
        # for an input image x, forward(x) should return the                   #
        # corresponding predicted keypoints.                                   #
        # NOTE: what is the required output size?                              #
        ########################################################################

        x = self.conv_layers(x)
        
        x = self.fc_layers(x)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return x
    
    def training_step(self, batch, batch_idx):
        images, targets = batch['image'], batch['keypoints']
        targets = targets.view(images.shape[0], -1)
        preds = self.forward(images)
        loss = nn.MSELoss()(preds, targets)
        self.log('loss',loss)
        return loss

    def configure_optimizers(self):
        # Define your optimizer
        return torch.optim.Adam(self.parameters(), lr=0.001)


class DummyKeypointModel(pl.LightningModule):
    """Dummy model always predicting the keypoints of the first train sample"""
    def __init__(self):
        super().__init__()
        self.prediction = torch.tensor([[
            0.4685, -0.2319,
            -0.4253, -0.1953,
            0.2908, -0.2214,
            0.5992, -0.2214,
            -0.2685, -0.2109,
            -0.5873, -0.1900,
            0.1967, -0.3827,
            0.7656, -0.4295,
            -0.2035, -0.3758,
            -0.7389, -0.3573,
            0.0086, 0.2333,
            0.4163, 0.6620,
            -0.3521, 0.6985,
            0.0138, 0.6045,
            0.0190, 0.9076,
        ]])

    def forward(self, x):
        return self.prediction.repeat(x.size()[0], 1, 1, 1)
