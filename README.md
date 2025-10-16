##Affincraft-nn

Affincraft-nn is a deep learning model designed to predict the binding affinity of protein–small molecule and protein–protein complexes. Its input is a preprocessed protein–ligand graph (a dictionary stored in a .pkl file), which contains information about nodes, edges, and molecular fingerprints. The model takes a series of complex .pkl files as input and outputs the predicted affinity values. An example of the content of these complex .pkl files can be found in preprocessed_data/read_pkl.ipynb.

Affincraft-nn is built based on Graphormer.



Graphormer is a deep learning package that allows researchers and developers to train custom models for molecule modeling tasks. It aims to accelerate the research and application in AI for molecule science, such as material discovery, drug discovery, etc. [Project website](https://www.microsoft.com/en-us/research/project/graphormer/).

