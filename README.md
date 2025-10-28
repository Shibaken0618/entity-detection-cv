# Folder Guide:
The root folder contains the following folders:
  - data: (this folder is empty), folder that would contain the train and val datasets (TODO, use floor plan data)
  - models: folder that contains the final output model after training (TODO)
  - results: folder containing a few (5 to be exact) samples images comparing ground truth to prediction output images side-by-side. Also includes json files of metrics used for evaluation. (TODO)
  - src: folder containing the source code used to complete this project, such as data preprocessing, model implementation, training, evaluation, and visualization.



# Key Results and Summary:
### Performance:
- Analyzed performance (Precision, Recall, F1-score) at 4 different confidence thresholds (0.3, 0.5, 0.7, 0.9)
- Best performance was at 0.7 confidence threshold

- Overall F1-Score: 94.3 %
- Title Block Detection F1-Score: 99.8 %
- Note Detection F1-Score: 100 %
- View Detection F1-Score: 91.2 %

{Specific values taken from output detection_model_output_results.json file...}
 "Conf_0.7": {
    "Class_Metrics": {
      "TitleBlock": {
        "TP": 200,
        "FP": 1,
        "FN": 0,
        "Precision": 0.9950248756218906,
        "Recall": 1.0,
        "F1-Score": 0.9975062344139651
      },
      "Note": {
        "TP": 200,
        "FP": 0,
        "FN": 0,
        "Precision": 1.0,
        "Recall": 1.0,
        "F1-Score": 1.0
      },
      "View": {
        "TP": 657,
        "FP": 99,
        "FN": 28,
        "Precision": 0.8690476190476191,
        "Recall": 0.9591240875912409,
        "F1-Score": 0.9118667591950034
      }
    },
    "Overall_Metrics": {
      "TP": 1057,
      "FP": 100,
      "FN": 28,
      "Precision": 0.9135695764909249,
      "Recall": 0.9741935483870968,
      "F1-Score": 0.9429081177520071
    }
  }  

### Training: (current nums are arbitrary)
- x training samples, y validation samples
- Stochastic gradient descent with 0.9 momentum and 0.0005 gradient decay
- Learning rate of 0.002 with step decay using scheduler (0.1 every 3 epochs)
- Batch size of 8
- Added color jitter and grayscaling data augmentation 
- training number of epochs is 10

### Model Choice:
Used Faster R-CNN with ResNet50 feature pyramid network (pulled from torchivision) to detect 4 classes (TitleBlock, Note, View, Background).Chose this model specifically since Faster R-CNN ResNet50 is used often for object detection requiring accurate bounding boxes. It also seems to work well on complex shapes and small objects. Since we are only dealing with 2000 images, ResNet50 was chosen as it seems to have a good balance between accuracy and computational cost.There would be multiple objects per image, with bounding boxes and classification for each object, therefore Feature Pyramid Network (FPN) would be useful. 



# Setup:
- Coded in Python 3.11 using PyTorch.
- Used given pyproject.toml file to create virtual environment.
- Will have to copy into the data folder, the full dataset before use.
