#### Image Classifier Part 2

This folder contains all the files for the app that trains a user defined classifier and predicts the top $k$ class probabilities for a given image.

The following table gives a brief descripition of the files:

| File               | Description                                                  |
| ------------------ | ------------------------------------------------------------ |
| train.py           | Contains all the functions and logic to train the network.   |
| predict.py         | Contains all the functions and logic to make a prediction.   |
| default_network.py | A network class that builds on top of the user defined inputs. If none of the optional arguments are given, it builds the `default classifier`. |

#### Train.py

```python
train.py data_directory [--save_dir path_to_save] [--arch model] [-learning_rate float]
                        [--hidden_units int] [--epochs int] [--gpu]
                
                
Required named arguments:
  
  data_directory        Specify data directory where training data is stored.
                                                

Optional named arguments:
  
  --save_dir            Directory for saving checkpoints
  
  --arch                Specify network architecture
  
  --learning_rate       Specify a learning rate (default: 0.001)
  
  --hidden_units        Specify number of hidden units (default: 2)
                                                
  --epochs              Number of epochs to train (default: 4)
                                                                                            
  --gpu                 Enables CUDA training                                              
```



#### Predict.py

```python
usage: predict.py path_to_image  checkpoint [--top_k int]
                                            [--category_names path_to_json]
                                            [--gpu]


Required named arguments:
  
  path_to_image        Path to image for prediction
                                                
  checkpoint           Path to checkpoint object with trained model
                                                

Optional named arguments:
                                                
  --top_k             Top K predictions to return (default: 1)
                                                
  --category_names    Specify the path for category to name mappings (default: None)
                                                
  --gpu               Enables CUDA
```

