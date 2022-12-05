# TE-Conv: Event Time Prediction for Dynamic Graphs

> GNN,  Dynamic Graph, TPP




## Brief Introduction

Graph is everywhere in our life and it is also a topic that attracts lots of attention of researchers. Graph Neural Network is a powerful type of neural network models that is aimed at learning to calculate vector representations for graph nodes. These vector representations can be used in downstream tasks such as node classification and link prediction and have shown excellent performance. What’s more, in real world, graphs are often dynamic. They can evolve over time. Discrete-Time Dynamic Graph (DTDG) and Continuous-Time Dynamic Graph (CTDG) are two ways to model dynamic graphs. Among them, CTDG enable the novel task of edge event time prediction, which means to predict the time when an edge will form again between two given nodes. Temporal Point Process has been a widely adopted model for temporal events but it has been rarely applied on graphs. We combine Graph Neural Network with a newly proposed Temporal Point Process model, UNIPoint, to address the problem of event time prediction on graphs.



## Running

### Requirements

The environment used by the author can be specified as below:

```{bash}
networkx==2.8.4
numpy==1.23.3
pandas==1.4.3
python==3.8.13
scikit-learn==1.1.2
scipy==1.9.1
torch==1.10.0+cu102
torch-cluster==1.5.9
torch-geometric==2.1.0.post1
torch-scatter==2.0.9
torch-sparse==0.6.12
torch-spline-conv==1.2.1
tqdm==4.64.1
```

### Datasets

Our dataset wikipedia is downloaded from the stanford web page for [Jodie](http://snap.stanford.edu/jodie/). We have included the preprocessed csv file in the folder ```data/```. 

Other datasets (e.g. ia-enron-employees) are from
[networkrepository.com](https://networkrepository.com/)


### Example Running Command

An example command for running our code is as below:
```{bash}
python main.py --dataset_size 200000 --learning_rate 0.001 --cpu_only False --mode tgn  --basis cos --name wikipedia --num_epoch 5 --n_coefficient 1 --mse_loss_coefficient 10 --use_dyrepabstractsum False --use_memory True
```


#### Arguments and Their Meanings

```{txt}
optional arguments:
--mode type=str, default="tgn", the model name, choices=["tgn", "dyrep", "tgat", "jodie", "distance_encoding"]


--name type=str, default="wikipedia", used dataset name


--num_epoch type=int, default=500, number of training epochs


--learning_rate type=float, default=0.001, learning rate


--memory_dim type=int, default=100, the dimension number of memory vectors, effective only if use_memory is true


--time_dim type=int, default=100, the dimension number of time encoding


--embedding_dim type=int, default=100, the dimension number of finally obtained node embedding


--time_encoding type=str2bool, default=True, whether to use time encoding



--n_coefficient type=int, default=2, loss weight for negative sample loss


--mse_loss_coefficient type=int, default=2, loss weight for the mse loss used in time prediction


--num_basis type=int, default=64, number of basis in time predictor


--basis default="powerlaw", choices=["exp", "sigmoid", "cos", "relu", "powerlaw", "mixed"], the type of basis function adopted in time predictor


--use_memory type=str2bool, default=True, whether to use memory vectors


--neighbor_size type=int, default=10, the number of a node's neighbors returned by neighbor_finder or neighbor_loader


--dataset_size type=int, default=200000, the number of samples to draw from the dataset


--cpu_only type=str2bool, default=True, whether to run the model only on CPU (to run without GPU)

```

## TODOs 
* Make the code more simplified: For the convenience of development, some code was not deleted but commented while I programmed the project. Some of these comments should be removed to make the code more organized.


