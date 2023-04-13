# GraphGONet

From the article entitled **GraphGONet: a self-explaining neural network encapsulating the Gene Ontology graph for phenotype prediction on gene expression** (submitted to Bioinformatics) by Victoria Bourgeais, Farida Zehraoui, and Blaise Hanczar.

---

## Description

GraphGONet is a self-explaining neural network integrating the Gene Ontology into its hidden layers.

## Get started

The code is implemented in Python (3.6.7) using [PyTorch v1.7.1](https://pytorch.org/) and [PyTorch-geometric v1.6.3](https://pytorch-geometric.readthedocs.io/en/1.6.3/modules/nn.html) (see [requirements.txt](https://forge.ibisc.univ-evry.fr/vbourgeais/GraphGONet/blob/master/requirements.txt) for more details about the additional packages used).

## Dataset

The full microarray dataset can be downloaded on ArrayExpress database under the id [E-MTAB-3732](https://www.ebi.ac.uk/arrayexpress/experiments/E-MTAB-3732/). 

TCGA dataset can be downloaded from [GDC portal](https://portal.gdc.cancer.gov/). 

Here, you can find the pre-processed training, validation and test sets with additional files for NN architecture to test the network: https://entrepot.ibisc.univ-evry.fr/d/d4764174275347f09862/

## Usage

Example on TCGA dataset:
<!--
There exists 3 functions (flag *processing*): one is dedicated to the training of the model (*train*), another one to the evaluation of the model on the test set (*evaluate*), and the last one to the prediction of the outcomes of the samples from the test set (*predict*).
-->

### Train

<!-- On the microarray dataset:
```bash
python3 scripts/GraphGONet.py --save --n_inputs=36834 --n_nodes=10663 --n_nodes_annotated=8249 --n_classes=1 --selection_op="top" --selection_ratio=0.001 --n_epochs=50 --es --patience=5 --class_weight 
```
-->

```bash
python3 scripts/GraphGONet.py --save --n_inputs=18427 --n_nodes=10636 --n_nodes_annotated=8288 --n_classes=12 --selection_op="top" --selection_ratio=0.001 --n_epochs=50 --es --patience=5 --class_weight 
```

<!--
### 2) Evaluate

### 3) Predict

The outcomes are saved into a numpy array.
-->

### Comparison with random selection

```bash
python scripts/GraphGONet.py --save --n_inputs=18427 --n_nodes=10636 --n_nodes_annotated=8288 --n_classes=12 --selection_op="random" --selection_ratio=0.001 --n_epochs=50 --es --patience=5 --class_weight 
```

### Comparison with no selection

```bash
python scripts/GraphGONet.py --save --n_inputs=18427 --n_nodes=10636 --n_nodes_annotated=8288 --n_classes=12 --n_epochs=50 --es --patience=5 --class_weight 
```

### Train the model with a small number of training samples

```bash
python scripts/GraphGONet.py --save --n_samples=50 --n_inputs=18427 --n_nodes=10636 --n_nodes_annotated=8288 --n_classes=12 --selection_op="top" --selection_ratio=0.001 --n_epochs=50 --es --patience=5 --class_weight 
```

### Help

All the details about the command line flags can be provided by the following command:

```bash
python scripts/GraphGONet.py --help
```

For most of the flags, the default values can be employed. *dir_data*, *dir_files*, and *dir_log* can be set to your own repositories. Only the flags in the command lines displayed have to be adjusted to reproduce the results from the paper. If you have enough GPU memory, you can choose to switch to the entire GO graph (argument *type_graph* set to "entire"). The graph can be reconstructed by following the notebooks: Build_GONet_graph_part{1,2,3}.ipynb located in the notebooks directory. Then, you should change the value of the arguments *n_nodes* and *n_nodes_annotated* in the command line. 


###  Interpretation tool

Please see the notebook entitled *Interpretation_tool.ipynb* (located in the notebooks directory) to perform the biological interpretation of the results.