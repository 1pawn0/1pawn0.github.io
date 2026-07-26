---
title: "Message Passing"
draft: true
---

source: [https://www.dgl.ai/dgl_docs/guide](https://www.dgl.ai/dgl_docs/guide/message.html)

# Message Passing

## Message Passing Paradigm

Let $x_v\in\mathbb{R}^{d_1}$ be the feature for node $v$, and
$w_{e}\in\mathbb{R}^{d_2}$ be the feature for edge $({u}, {v})$. The
**message passing paradigm** defines the following node-wise and
edge-wise computation at step $t+1$:

$$\text{Edge-wise: } m_{e}^{(t+1)} = \phi \left( x_v^{(t)}, x_u^{(t)}, w_{e}^{(t)} \right) , ({u}, {v},{e}) \in \mathcal{E}.$$

$$\text{Node-wise: } x_v^{(t+1)} = \psi \left(x_v^{(t)}, \rho\left(\left\lbrace m_{e}^{(t+1)} : ({u}, {v},{e}) \in \mathcal{E} \right\rbrace \right) \right).$$

In the above equations, $\phi$ is a **message function** defined on each
edge to generate a message by combining the edge feature with the
features of its incident nodes; $\psi$ is an **update function** defined
on each node to update the node feature by aggregating its incoming
messages using the **reduce function** $\rho$.
