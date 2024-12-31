class Classifier6(nn.Module):

    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            ## Convolitional Layer 1
                nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=1), # 1 input 16 filters, padding so same dim
                nn.ReLU(), # ReLU introduce non-linearity
                nn.MaxPool2d(2, 2), #pool
 
                ## Convolutional Layer 2
                nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=1), # 16 inputs 32 filters
                nn.ReLU(),
                nn.MaxPool2d(2, 2),   
 
                ## feed forward layer w/ 1024 neurons, regular layer
                nn.Flatten(),
                nn.Linear(800, 1024),    ## see how to get 800 below on last cell
                nn.ReLU(),

                nn.Linear(1024, 10), # maps to output w/ 10 classes
                nn.LogSoftmax(dim=1)
        )
   
    def forward(self, inputs):
        return self.model(inputs)