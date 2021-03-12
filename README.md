
### File Description

+ `checkpoints`: contains saved weights of the model.

+ `dataset`: contains three folders for the three classes (person with-mask, person without-mask, and not a person). also a detailed description file is included there. 

+ `logs`: tensorboard logs used for including the training charts in the report.

+ `data.py`: for loading the dataset and creating instances of "torch.utils.data.Dataset".

+ `model.py`: contains the neural network's architecture.

+ `train.py`: main script for training the model. It uses pytorch lightning to for training, logging and saving checkpoint. Run the file without additional arguments for the training to start. The working directory must the project's root directory.

+ `evaluation.py`: main code used for evaluationg the model and generating performance metrics like classification report by sklearn and confusion matrix. Run the file without additional arguments. The working directory must be the project's root directory.
 
 
 

 










