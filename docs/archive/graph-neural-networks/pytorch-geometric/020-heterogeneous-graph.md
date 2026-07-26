---
title: Heterogeneous Graphs
draft: false
---
sources:  
        [https://pytorch-geometric.readthedocs.io/en/latest/tutorial/heterogeneous.html](https://pytorch-geometric.readthedocs.io/en/latest/tutorial/heterogeneous.html)  
        [https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.HeteroData.html](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.HeteroData.html)  

### Heterogeneous Graph Learning

Heterogeneous graphs come with **different types of information attached to nodes and edges**. Thus, **a single node or edge feature tensor *cannot* hold all node or edge features of the whole graph**, due to differences in type and dimensionality. Instead, **a set of types need to be specified for nodes and edges**, respectively, each having its own data tensors. As a consequence of the different data structure, the message passing formulation changes accordingly, allowing the computation of message and update function conditioned on node or edge type.

![hg-example](img/hg_example.svg)

#### The Representation of a Heterogeneous Graph

For example assume a heterogeneous graph that has four **node types**: `node_type_1`, `node_type_2`, `node_type_3`, `node_type_4`  
And it has four **edge types**:

1. **`edge_type_1`**: directed edge from the source `node_type_1` connected to the destination `node_type_2`
2. **`edge_type_2`**: directed edge from the source `node_type_1` connected to the destination `node_type_3`
3. **`edge_type_3`**: directed edge from the source `node_type_2` connected to the destination `node_type_2` (a directed self-loop edge)
4. **`edge_type_4`**: directed edge from the source `node_type_2` connected to the destination `node_type_4`  

We define **node feature** tensors, **edge index** tensors and **edge feature** tensors individually for each type.

```python
from torch_geometric.data import HeteroData

data = HeteroData()

data['node_type_1'].x = ... # [count_of_all_type1_nodes, num_of_the_features_of_a_node_type1]
data['node_type_2'].x = ... # [count_of_all_type2_nodes, num_of_the_features_of_a_node_type2]
data['node_type_3'].x = ... # [count_of_all_type3_nodes, num_of_the_features_of_a_node_type3]
data['node_type_4'].x = ... # [count_of_all_type4_nodes, num_of_the_features_of_a_node_type4]

data['node_type_1', 'edge_type_1', 'node_type_2'].edge_index = ... # [2, num_edges_edge_type1]
data['node_type_1', 'edge_type_2', 'node_type_3'].edge_index = ... # [2, num_edges_edge_type2]
data['node_type_2', 'edge_type_3', 'node_type_2'].edge_index = ... # [2, num_edges_edge_type3]
data['node_type_2', 'edge_type_4', 'node_type_4'].edge_index = ... # [2, num_edges_edge_type4]

data['node_type_1', 'edge_type_1', 'node_type_2'].edge_attr = ... # [num_edges_edge_type1, num_of_the_features_of_a_edge_type1]
data['node_type_1', 'edge_type_2', 'node_type_3'].edge_attr = ... # [num_edges_edge_type2, num_of_the_features_of_a_edge_type2]
data['node_type_2', 'edge_type_3', 'node_type_2'].edge_attr = ... # [num_edges_edge_type3, num_of_the_features_of_a_edge_type3]
data['node_type_2', 'edge_type_4', 'node_type_4'].edge_attr = ... # [num_edges_edge_type4, num_of_the_features_of_a_edge_type4]
```

Node or edge tensors will be automatically created upon first access and indexed by string keys.  
**Node types are identified by a single string** while **edge types are identified by using a triplet (`source_node_type`, `edge_type`, `destination_node_type`)** of strings: the edge type identifier and the two node types between which the edge type can exist. As such, the data object allows different feature dimensionalities for each type.
