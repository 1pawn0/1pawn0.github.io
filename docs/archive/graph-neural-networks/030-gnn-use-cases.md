---
title: "GNN Use Cases"
tags:
    - GNN Use Cases
    - Graph-based learning
    - Graph Neural Networks
    - Graph Neural Network
    - GNN
    - GNNs
    - Graphs
    - Graph theory
draft: false
---

### When to use a GNN?

There are three types of criteria for identifying GNN problems:  

##### I. Implicit relationships and interdependencies  

It’s usually beneficial to explore whether implicit relationships or interdependencies might exist that could be represented explicitly. Implicit relationships are **connections that aren’t immediately documented or obvious** within the data but can still play a significant role in understanding the underlying patterns and behaviors.  
To determine if your problem might benefit from modeling implicit relationships with graphs, **consider whether there are hidden or indirect connections between entities in your dataset**.  
Another indicator is the presence of entities that **share common attributes or activities without a direct or documented relationship**.

##### II. High dimensionality and sparsity  

To determine if your problem involves high-dimensional and sparse data suitable for GNNs, **consider whether your dataset contains numerous entities with limited direct interactions or relationships**.  
Another indicator that your problem may be suitable for graph-based models is when the **data represents entities that are sparsely connected but share significant characteristics**.

##### III. Complex nonlocal interactions

To determine if your problem involves complex, nonlocal interactions suitable for GNNs, **consider whether the outcome or behavior of one entity depends on the attributes or actions of entities that aren’t directly connected to it but may be indirectly connected through other entities**.
Another indicator is whether the **problem involves scenarios where information, influence, or effects propagate through a network over time**.  

- In determining whether your problem is a good candidate for a GNN, ask yourself these questions:
  - Are there implicit relationships or interdependencies in my data that I could model?
  - Do the interactions between entities exhibit complex, nonlocal dependencies that go beyond immediate connections?
  - Is the data high-dimensional and sparse, with a need to capture underlying relational structures?
