# JAX Transformations: Vectorization via [`jax.vmap()`](https://docs.jax.dev/en/latest/_autosummary/jax.vmap.html#jax.vmap)

> JAX’s `vmap` transformation automatically “batches” (vectorizes) a Python function over array axes, letting you write clean, NumPy‐style code without explicit loops while still getting XLA‐level performance. You control the batch dimensions via `in_axes`/`out_axes`, can nest `vmap` or combine it with `jit`/`grad`, and avoid common pitfalls by understanding how shapes propagate.  

---

## Overview of `jax.vmap()`

### Understand the Problem `jax.vmap` Solves: The "Batching" Challenge

Applying a function designed for a single data point (e.g., calculating the loss for one example, applying a per-example transformation) within a standard Python loop over a batch of data is very slow in JAX. This is because JAX needs to trace and compile the function for each iteration of the loop. You lose the benefits of JIT compilation and XLA optimization across the entire batch.
If you try to manually reshape your inputs, you should rewrite your function to handle an extra batch dimension, and reshape the outputs. This approach is cumbersome, error-prone, and makes code harder to read and maintain.

### Introducing `jax.vmap`: Automatic Batching

**What it Does**: `jax.vmap` is a function transformation. It takes a function `f` that operates on one or a few inputs and returns a new function `vmapped_f` that automatically applies `f` across a new leading dimension (the "**batch dimension**") of its inputs. It is a transformation that lifts a function `f(x)` to operate on batches `f_batch(xs)`, tracing `f` once and adding a new batch axis rather than executing Python loops.  
It is being used Eliminates Python‐level loops for cleaner code and better performance by leveraging XLA fusion under the hood.  

**The Magic**:
> `vmap` doesn't just loop internally. It rewrites the original function `f`'s computation graph to perform the operation on the entire batch simultaneously. This is where the performance comes from, as it can be efficiently compiled by XLA.

**An Analogy**:
> Think of it like taking a function that processes a single image (width x height) and automatically getting a function that processes a batch of images (batch_size x width x height).

**Not Parallelism**:
> Contrary to its name, `vmap` is vectorization/batching, not explicit parallel mapping—underneath it lowers to efficient batched primitives.

---

## Core Concepts

### `in_axes` and `out_axes`

Input Axes (in_axes): This is the most crucial argument. It tells vmap which axis of each input array corresponds to the dimension you want to batch over.
in_axes=0 (Common): The first dimension of the input is the batch dimension. This is the default if you provide a single axis value.
in_axes=None: This input does not have a batch dimension. It will be "broadcast" to all elements in the batch. Useful for parameters or constants.
in_axes=(0, None, 1): A tuple for multiple inputs. Each element corresponds to an input argument.
in_axes=0: If the function takes multiple arguments and they all have the batch dimension as the first axis. vmap will apply this 0 to all inputs.
Output Axes (out_axes): This tells vmap where to put the new batch dimension in the output arrays.
out_axes=0 (Default): The new batch dimension becomes the first dimension of the output.
out_axes=1: The new batch dimension becomes the second dimension of the output, and so on.
out_axes=None: The output does not have a batch dimension (less common, but possible).
out_axes=(0, 0): A tuple for multiple outputs.
Broadcasting: Explain how in_axes=None enables broadcasting, similar to NumPy's broadcasting rules but explicitly controlled by vmap.

- **`in_axes`**: Specifies which axis of each argument to batch over. Can be an integer (same for all), a tuple (per-arg), or `None` to leave an arg unbatched.  
- **`out_axes`**: Controls where the new batch axis appears in the output, default is `0`, stacking results along the first dimension.  

### Tracing and Compilation

- JAX traces your function once, builds a batched computation graph, and compiles with XLA. This removes Python loop overhead and fuses operations.  

---

## Basic Usage and Examples

```python
import jax.numpy as jnp
from jax import vmap

# A simple scalar function
def square(x):
    return x ** 2

# Vectorize over the first axis of x
batched_square = vmap(square, in_axes=0, out_axes=0)

xs = jnp.arange(5)           # shape (5,)
ys = batched_square(xs)      # shape (5,)
# ys == [0,1,4,9,16]
```

- This mirrors NumPy’s elementwise ops but generalizes to arbitrary functions.

---

## Performance Benefits

- **Loop elimination**: No Python loops—batching happens in compiled code.  
- **XLA fusion**: Adjacent operations fuse into single GPU/TPU kernels, reducing memory traffic.  
- **Memory layout**: Operating on contiguous batches often improves cache locality and throughput.

Benchmarking your own code against naïve loops can show 5×–50× speedups depending on workload.

---

## Composing with `jit` and `grad`

- **`jit(vmap(f))`**: JIT-compile the batched function for maximum speed.  
- **`vmap(grad(f))`**: Compute per-example gradients in parallel—common in ML for per-sample backprop.  
- You can even do `jit(vmap(grad(f)))` for a fully compiled, batched gradient function.

```python
from jax import grad, jit

batched_grad = jit(vmap(grad(square)))
```

---

## Advanced Patterns

### Multiple Arguments

```python
def add(x, y): return x + y
batched_add = vmap(add, in_axes=(0, None))  # batch first arg only
```

- Use tuples in `in_axes` to control batching per parameter.

### Nested `vmap`

- Apply `vmap` twice to batch over two axes, e.g., time × batch or height × width.

### `axis_name` & `lax` batching

- With `pmap` you get `axis_name`, but with plain `vmap` you stick to `in_axes` semantics. For truly distributed multi-device batching use `pmap` instead.

---

## Pitfalls & Debugging

- **Shape mismatches**: Ensure your function handles the extra axis–print intermediate shapes to debug.  
- **Static vs. dynamic**: `in_axes` must be static (known at compile time).  
- **Unsupported ops**: Some Python operations or side effects can’t be lifted; switch to JAX NumPy primitives.

---

## Best Practices

- **Default axes**: Omit `in_axes`/`out_axes` when you batch all args on axis 0 to keep code concise.  
- **Combine transforms**: Experiment with `jit(vmap(...))` and `vmap(jit(...))` to see what gives better compile times vs. runtime speed.  
- **Profile**: Use JAX’s profiling tools to inspect XLA HLO and kernel performance.  
- **Readability**: Rename wrapped functions clearly (e.g. `batched_f = vmap(f)`), so readers know they’re batched.

---
