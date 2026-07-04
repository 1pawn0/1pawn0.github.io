# JAX Transformations: Just-In-Time Compilation via [`jax.jit()`](https://docs.jax.dev/en/latest/_autosummary/jax.jit.html#jax.jit)

JAX allows us to transform Python functions. JAX accomplishes this by reducing each function into a sequence of [primitive](https://docs.jax.dev/en/latest/jax-primitives.html) operations, each representing one fundamental unit of computation.

What is a [primitive](https://docs.jax.dev/en/latest/jax-primitives.html) operation in JAX?
: In JAX, a primitive is a low-level atomic operation (like addition, multiplication, convolution) that forms the basic building blocks of JAX computations.
each primitive is representing one fundamental unit of computation.

[`jax.make_jaxpr()`](https://docs.jax.dev/en/latest/_autosummary/jax.make_jaxpr.html#jax.make_jaxpr) function:
: Can be used to see the sequence of primitives behind a function.

We can jit compile a function using [`jax.jit()`](https://docs.jax.dev/en/latest/_autosummary/jax.jit.html#jax.jit) function:
> The input parameter of  [`jax.jit(fun: Callable)`](https://docs.jax.dev/en/latest/_autosummary/jax.jit.html#jax.jit)(The callable function that we want to be jitted) **must** be a **pure function**.

```py
import jax
import jax.numpy as jnp

def selu(x, alpha=1.67, lambda_=1.05):
  return lambda_ * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)
x = jnp.arange(1000000)
selu(x).block_until_ready()
selu_jit = jax.jit(selu)
selu_jit(x).block_until_ready()
```

___

```py
import jax
@jax.jit
def selu(x, alpha=1.67, lmbda=1.05):
  return lmbda * jax.numpy.where(x > 0, x, alpha * jax.numpy.exp(x) - alpha)
key = jax.random.key(0)
x = jax.random.normal(key, (10,))
print(selu(x))  
```
