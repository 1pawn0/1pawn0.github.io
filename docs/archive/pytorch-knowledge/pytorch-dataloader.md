---
draft: false
title: PyTorch DataLoader
tags:
  - DataLoader
  - PyTorch
  - Datasets
  - Torch
  - Machine Learning
  - Deep Learning
  - Python
  - Data
---

# [PyTorch DataLoader](https://docs.pytorch.org/docs/main/data.html#torch.utils.data.DataLoader)

Source: [`torch.utils.data` module documentation](https://docs.pytorch.org/docs/main/data.html#module-torch.utils.data)

Consider a Python class that is going to be used as a PyTorch [Dataset](https://docs.pytorch.org/docs/main/data.html#torch.utils.data.Dataset) object. The methods [` __init__()`](https://docs.python.org/3/reference/datamodel.html#object.__init__), [`__getitem__()`](https://docs.python.org/3/reference/datamodel.html#object.__getitem__), [`__len__()`](https://docs.python.org/3/reference/datamodel.html#object.__len__), and optionally [`__iter__()`](https://docs.python.org/3/reference/datamodel.html#object.__iter__) must have been already implemented in that class. [Source](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files)

[`collate_fn`](https://docs.pytorch.org/docs/main/data.html#working-with-collate-fn)
[`torch.utils.data.default_collate`](https://docs.pytorch.org/docs/main/data.html#torch.utils.data.default_collate)

the input dataset of `DataLoader` must be an object that already implements the methods `__init__()`, `__getitem__()`, `__len__()`, and `__iter__()`. this `dataset` object must return pytorch tensors, numpy arrays, numbers, dicts or lists when `dataset[i]` and `dataset.__iter__().__next__()` are called.

> In other words, batch must contain pytorch tensors, numpy arrays, numbers, dicts or lists.

```python

torch.utils.data.DataLoader(
    dataset=ds,
    # the input dataset of `DataLoader` must be an object that already implements the methods `__init__()`, `__getitem__()`, `__len__()`, and `__iter__()`. this `dataset` object must return pytorch tensors, numpy arrays, numbers, dicts or lists when `ds[i]` and `ds.__iter__().__next__()` are called.
)
```
