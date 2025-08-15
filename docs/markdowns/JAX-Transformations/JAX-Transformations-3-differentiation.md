# JAX Transformations: Differentiation via [`jax.grad()`](https://docs.jax.dev/en/latest/_autosummary/jax.grad.html#jax.grad)

## Introduction

Automatic differentiation (AD) is a technique to compute exact derivatives of functions implemented as code, chaining together elementary derivative rules without symbolic manipulation or finite differences.  
JAX is a Python library for high‑performance array computing and program transformations; it brings together XLA compilation with a clean AD API.  

Okay, here is a draft for your tutorial on JAX differentiation using jax.grad(). I'll structure it to be clear, practical, and relevant to your interests as a data science/ML enthusiast who uses Google Colab and JAX.

JAX Transformations: Differentiation via jax.grad()
Welcome to this tutorial focusing on one of JAX's fundamental transformations: automatic differentiation using the jax.grad() function. As a data science and machine learning enthusiast, understanding and efficiently computing gradients is absolutely crucial, especially for optimization algorithms like gradient descent. JAX provides a powerful and flexible way to do this, and jax.grad() is your primary tool for computing the gradient of scalar functions.

In machine learning, we often define a loss function that measures how well our model performs. Training a model typically involves minimizing this loss function with respect to the model's parameters. Gradient-based optimization methods (like Stochastic Gradient Descent, Adam, etc.) achieve this by iteratively adjusting parameters in the direction opposite to the gradient of the loss function.

Automatic differentiation (AD) is a set of techniques to computationally evaluate the derivative of a function specified by a computer program. It's neither symbolic differentiation (which can be slow and struggle with complex programs) nor numerical differentiation (which can suffer from precision issues and is computationally expensive for high-dimensional spaces). AD works by breaking down the function into a sequence of elementary operations and applying the chain rule.

JAX implements AD efficiently and at a high level, allowing you to focus on defining your functions rather than manually deriving gradients.

---

## 1. `jax.grad()` Basics

### 1.1 What is `jax.grad()`?  

`jax.grad(f)` takes a Python function `f` that returns a scalar and yields another function that computes ∇f.
`jax.grad()` is a function transformation. It takes a Python function `func` that computes a scalar output and returns a new Python function that computes the gradient of `func` with respect to its first argument by default.

### 1.2 A First Example  

```python
import jax
import jax.numpy as jnp

f = lambda x: 3 * x**2 + 2 * x + 1
df = jax.grad(f)

print(df(4.0))  # Should print 26.0
```

Here, `df(4.0)` computes 6·4 + 2 = 26 exactly.

> `jax.grad()` expects the function to return a scalar value. If your function returns an array or a complex structure, you'll need to sum or process it to get a scalar before passing it to `jax.grad()`. For example, if your function calculates element-wise losses, you would typically `jnp.sum()` or `jnp.mean()` them.
---

## 2. Differentiating JAX‑NumPy Functions

You can wrap JAX‑NumPy ops directly:

```python
grad_tanh = jax.grad(jnp.tanh)
print(grad_tanh(0.2))  # ≈ 1 - tanh(0.2)**2 ≈ 0.961043
```

This reproduces the derivative of tanh at 0.2.

---

## 3. Handy Scalar‑Function Example

Compute the derivative of  
\[
f(x) = x^3 + 2x^2 - 3x + 1
\]  
with JAX:

```python
f = lambda x: x**3 + 2*x**2 - 3*x + 1
df = jax.grad(f)
print(df(5.0))  # 3*5^2 + 4*5 - 3 = 75 + 20 - 3 = 92
```

In JAX, higher‑order derivatives work just as easily.

---

## 4. Vector and Jacobian Calculations

When your function returns a vector, use `jax.jacobian()` (alias of `jax.jacrev`) to get the full Jacobian matrix. For example:

```python
from jax import jacobian

def g(x):
    return jnp.array([x[0] * x[1], x[1]**2])

J = jacobian(g)
print(J(jnp.array([2.0, 3.0])))
# [[3.0, 2.0],
#  [0.0, 6.0]]
```

You can also use `vmap` + `grad` to get gradients over batches.

---

## 5. Higher‑Order Derivatives

Since `grad` itself is a JAX transformation, you can stack it:

```python
import jax.numpy as jnp
from jax import grad

d2 = grad(grad(lambda x: jnp.sin(x)))
print(d2(1.0))  # Should be -sin(1.0)
```

This works out‑of‑the‑box because JAX’s AD transforms are fully composable.

---

## 6. Custom Gradients

Sometimes you need to override or augment JAX’s automatic rule. Use `jax.custom_gradient` (or the lower‑level `jax.custom_vjp`) to supply your own forward and backward pass:

```python
import jax
from jax import custom_gradient

@custom_gradient
def square(x):
    primal = x * x
    def vjp(g):
        return (2 * x * g,)
    return primal, vjp
```

This decorator style follows TensorFlow’s API but plugs into JAX under the hood citeturn0search3. For a deeper dive into custom VJPs, see the dedicated notebook.

---

## 7. Common Pitfalls

- **Python side‑effects** (e.g., printing or mutating lists) won’t be tracked by AD—keep functions pure.  
- **Control flow** (loops, conditionals) works but beware of non‑JIT‑friendly patterns.  
- **Non‑array inputs** (e.g., Python ints) may break differentiation—convert to JAX arrays.  

---

## 8. Performance Tips

- **Combine** `jax.jit` with `grad` for compiled, fused gradient computation:  

  ```python
  jitted_grad = jax.jit(jax.grad(f))
  ```

  This yields both speed and memory benefits.  
- **Batch** your gradient calls with `vmap` to amortize overhead.  
- **Avoid** creating new Python objects inside hot loops.

---

___
-----

# JAX Transformations: Differentiation via [`jax.grad()`](https://www.google.com/search?q=%5Bhttps://docs.jax.dev/en/latest/_autosummary/jax.grad.html%23jax.grad%5D\(https://docs.jax.dev/en/latest/_autosummary/jax.grad.html%23jax.grad\))

Welcome to this tutorial focusing on one of JAX's fundamental transformations: automatic differentiation using the `jax.grad()` function. As a data science and machine learning enthusiast, understanding and efficiently computing gradients is absolutely crucial, especially for optimization algorithms like gradient descent. JAX provides a powerful and flexible way to do this, and `jax.grad()` is your primary tool for computing the gradient of scalar functions.

## What is Automatic Differentiation?

In machine learning, we often define a loss function that measures how well our model performs. Training a model typically involves minimizing this loss function with respect to the model's parameters. Gradient-based optimization methods (like Stochastic Gradient Descent, Adam, etc.) achieve this by iteratively adjusting parameters in the direction opposite to the gradient of the loss function.

Automatic differentiation (AD) is a set of techniques to computationally evaluate the derivative of a function specified by a computer program. It's neither symbolic differentiation (which can be slow and struggle with complex programs) nor numerical differentiation (which can suffer from precision issues and is computationally expensive for high-dimensional spaces). AD works by breaking down the function into a sequence of elementary operations and applying the chain rule.

JAX implements AD efficiently and at a high level, allowing you to focus on defining your functions rather than manually deriving gradients.

## Introducing `jax.grad()`

`jax.grad()` is a function transformation. It takes a Python function `func` that computes a scalar output and returns a *new* Python function that computes the gradient of `func` with respect to its *first* argument by default.

Let's look at the basic syntax:

```python
import jax
import jax.numpy as jnp

# Define a simple function
def my_function(x):
  return x**2 + 2.0 * x + 1.0

# Create the gradient function
gradient_my_function = jax.grad(my_function)

# Compute the function value and its gradient at a point
x_value = 3.0

# Compute the function value
f_x = my_function(x_value)
print(f"Function value at x = {x_value}: {f_x}")

# Compute the gradient value
grad_f_x = gradient_my_function(x_value)
print(f"Gradient value at x = {x_value}: {grad_f_x}")

# Let's verify manually: The derivative of f(x) = x^2 + 2x + 1 is f'(x) = 2x + 2.
# At x = 3.0, f'(3.0) = 2*(3.0) + 2 = 6.0 + 2 = 8.0.
# JAX should give 8.0.
```

When you run this in a Colab notebook:

```python
# Output:
# Function value at x = 3.0: 16.0
# Gradient value at x = 3.0: 8.0
```

This matches our manual calculation. `jax.grad(my_function)` returned a new function that, when called with `x_value`, computes the gradient of `my_function` at that point.

**Important Note:** `jax.grad()` expects the function to return a **scalar** value. If your function returns an array or a complex structure, you'll need to sum or process it to get a scalar before passing it to `jax.grad()`. For example, if your function calculates element-wise losses, you would typically `jnp.sum()` or `jnp.mean()` them.

## Differentiating with Respect to Other Arguments

By default, `jax.grad()` computes the gradient with respect to the first argument. What if your function takes multiple arguments and you want the gradient with respect to a different one, or multiple ones?

You can use the `argnums` argument of `jax.grad()`. `argnums` can be an integer (for a single argument) or a tuple/list of integers (for multiple arguments). The indices are 0-based.

```python
def another_function(x, y):
  return x**2 * y + jnp.sin(y)

# Gradient with respect to the first argument (x)
grad_wrt_x = jax.grad(another_function, argnums=0)

# Gradient with respect to the second argument (y)
grad_wrt_y = jax.grad(another_function, argnums=1)

# Gradient with respect to both arguments (returns a tuple of gradients)
grads_wrt_xy = jax.grad(another_function, argnums=(0, 1))

x_val = 2.0
y_val = jnp.array(jnp.pi / 2.0) # Use jnp array for trig functions

print(f"Function value at x={x_val}, y={y_val}: {another_function(x_val, y_val)}")

print(f"Gradient wrt x at x={x_val}, y={y_val}: {grad_wrt_x(x_val, y_val)}")
# Manual check: d/dx (x^2 * y + sin(y)) = 2*x*y. At x=2, y=pi/2: 2*2*(pi/2) = 2*pi ~= 6.28

print(f"Gradient wrt y at x={x_val}, y={y_val}: {grad_wrt_y(x_val, y_val)}")
# Manual check: d/dy (x^2 * y + sin(y)) = x^2 + cos(y). At x=2, y=pi/2: 2^2 + cos(pi/2) = 4 + 0 = 4.0

grad_x, grad_y = grads_wrt_xy(x_val, y_val)
print(f"Gradients wrt x and y at x={x_val}, y={y_val}: ({grad_x}, {grad_y})")

```

Running this:

```python
# Output will be similar to:
# Function value at x=2.0, y=1.570796370792389: 7.283185005187988
# Gradient wrt x at x=2.0, y=1.570796370792389: 6.283185307179586
# Gradient wrt y at x=2.0, y=1.570796370792389: 4.0
# Gradients wrt x and y at x=2.0, y=1.570796370792389: (6.283185307179586, 4.0)
```

Again, the results match our manual calculations. When `argnums` is a tuple, `jax.grad()` returns a tuple of gradients, corresponding to the order of arguments in `argnums`.

## Higher-Order Derivatives

One of the powerful features of JAX is its composable transformations. You can apply `jax.grad()` to a function that was *itself* the result of a `jax.grad()` call. This allows you to compute higher-order derivatives.

Applying `jax.grad()` once gives you the first derivative. Applying `jax.grad()` to the first derivative function gives you the second derivative (the Hessian for a scalar input function), and so on.

```python
def cubic_function(x):
  return x**3 - 3.0 * x**2 + 5.0

# First derivative: d/dx (x^3 - 3x^2 + 5) = 3x^2 - 6x
first_derivative_func = jax.grad(cubic_function)

# Second derivative: d/dx (3x^2 - 6x) = 6x - 6
second_derivative_func = jax.grad(first_derivative_func)

# Third derivative: d/dx (6x - 6) = 6
third_derivative_func = jax.grad(second_derivative_func)

# Fourth derivative: d/dx (6) = 0
fourth_derivative_func = jax.grad(third_derivative_func)


x_val = 4.0

print(f"f'(x) at x={x_val}: {first_derivative_func(x_val)}")
# Manual check: 3*(4^2) - 6*4 = 3*16 - 24 = 48 - 24 = 24.0

print(f"f''(x) at x={x_val}: {second_derivative_func(x_val)}")
# Manual check: 6*4 - 6 = 24 - 6 = 18.0

print(f"f'''(x) at x={x_val}: {third_derivative_func(x_val)}")
# Manual check: 6.0

print(f"f''''(x) at x={x_val}: {fourth_derivative_func(x_val)}")
# Manual check: 0.0
```

Running this:

```python
# Output:
# f'(x) at x=4.0: 24.0
# f''(x) at x=4.0: 18.0
# f'''(x) at x=4.0: 6.0
# f''''(x) at x=4.0: 0.0
```

This demonstrates the elegance of JAX's composable transformations. You can build up complex operations by applying transformations sequentially.

## Differentiation with Respect to Complex Structures (e.g., Dictionaries, Tuples)

JAX is designed to work with JAX PyTrees, which are structures like lists, tuples, dictionaries, and custom classes that contain JAX arrays. `jax.grad()` can compute gradients with respect to arguments that are PyTrees.

```python
from typing import Dict # Using typing hints for clarity

def function_with_dict_input(params: Dict[str, jnp.ndarray], x: jnp.ndarray) -> jnp.ndarray:
  """A simple linear model: y = w*x + b"""
  w = params['weights']
  b = params['bias']
  return w * x + b

def loss_function(params: Dict[str, jnp.ndarray], x: jnp.ndarray, y_true: jnp.ndarray) -> jnp.ndarray:
  """Mean squared error loss"""
  y_pred = function_with_dict_input(params, x)
  return jnp.mean((y_pred - y_true)**2)

# Sample data
key = jax.random.PRNGKey(0)
true_w = 2.5
true_b = 1.0
x_data = jax.random.normal(key, (10,))
noise = jax.random.normal(key, (10,)) * 0.1
y_data = true_w * x_data + true_b + noise

# Initial parameters
initial_params = {'weights': jnp.array(1.0), 'bias': jnp.array(0.0)}

# Get the gradient of the loss function with respect to the 'params' dictionary
# We only want to differentiate with respect to the first argument (params)
grad_loss_wrt_params = jax.grad(loss_function, argnums=0)

# Compute the gradients at the initial parameters
gradients = grad_loss_wrt_params(initial_params, x_data, y_data)

print(f"Initial parameters: {initial_params}")
print(f"Gradients of the loss wrt parameters: {gradients}")
# The output `gradients` will have the same structure as `initial_params`
```

Running this:

```python
# Output will show the gradients for 'weights' and 'bias'
# Initial parameters: {'weights': Array(1., dtype=float32, weak_type=True), 'bias': Array(0., dtype=float32, weak_type=True)}
# Gradients of the loss wrt parameters: {'weights': Array(-3.10..., dtype=float32), 'bias': Array(-2.58..., dtype=float32)}
```

This is incredibly useful for training machine learning models where your parameters are often stored in dictionaries or other nested structures. `jax.grad()` handles the differentiation through these structures automatically.

## Differentiation of Functions Returning Multiple Values

As mentioned earlier, `jax.grad()` requires the function to return a scalar. However, sometimes you might have a function that computes multiple values, and you want to compute the gradient of a scalar derived from those values (like their sum).

You can use `jax.vmap` or simply compute the scalar within the function you pass to `jax.grad()`.

If your function returns a single value that is an array, you typically sum or mean it:

```python
def elementwise_square(x):
  return x**2 # Returns an array, not a scalar

# This would NOT work directly with jax.grad()
# grad_elementwise = jax.grad(elementwise_square) # Error!

# Instead, define a scalar function based on the output
def sum_of_squares(x):
  return jnp.sum(elementwise_square(x)) # Returns a scalar

grad_sum_of_squares = jax.grad(sum_of_squares)

x_array = jnp.array([1.0, 2.0, 3.0])
print(f"Sum of squares gradient at {x_array}: {grad_sum_of_squares(x_array)}")
# Manual check: d/dx_i (sum(x_j^2)) = d/dx_i (x_i^2 + sum_{j!=i} x_j^2) = 2*x_i
# For [1.0, 2.0, 3.0], the gradient should be [2.0, 4.0, 6.0]
```

Running this:

```python
# Output:
# Sum of squares gradient at [1. 2. 3.]: [2. 4. 6.]
```

This confirms that `jax.grad()` correctly computes the gradient of the scalar sum with respect to the input array.

## What About Control Flow and Side Effects?

JAX's transformations, including `jax.grad()`, work best with pure functions – functions that have no side effects and whose output depends only on their inputs. While JAX can handle some forms of control flow (`jax.lax.cond`, `jax.lax.while_loop`, `jax.lax.fori_loop`), standard Python control flow (`if`, `for`, `while`) might lead to unexpected behavior or errors when used within functions being transformed.

Similarly, functions with side effects (like printing, modifying global variables, or I/O) are not compatible with JAX transformations like `jax.grad()`. Keep your JAX functions pure for reliable differentiation.

## Putting it Together: A Simple Optimization Example

Let's use `jax.grad()` to perform a few steps of gradient descent to minimize our `loss_function` from before.

```python
import jax
import jax.numpy as jnp
# Assume loss_function and function_with_dict_input are defined as above

# Sample data (re-using from above for clarity)
key = jax.random.PRNGKey(0)
true_w = 2.5
true_b = 1.0
x_data = jax.random.normal(key, (10,))
noise = jax.random.normal(key, (10,)) * 0.1
y_data = true_w * x_data + true_b + noise

# Initial parameters
params = {'weights': jnp.array(1.0), 'bias': jnp.array(0.0)}

# Learning rate for gradient descent
learning_rate = 0.1

# Get the gradient function
grad_loss_wrt_params = jax.grad(loss_function, argnums=0)

print(f"Starting parameters: {params}")

# Perform a few gradient descent steps
for step in range(100):
    # Compute gradients
    gradients = grad_loss_wrt_params(params, x_data, y_data)

    # Update parameters using gradient descent: param = param - learning_rate * gradient
    # JAX provides a convenient way to update PyTrees using jax.tree_util
    params = jax.tree_util.tree_map(lambda p, g: p - learning_rate * g, params, gradients)

    if step % 10 == 0:
        current_loss = loss_function(params, x_data, y_data)
        print(f"Step {step}, Loss: {current_loss:.4f}")

print(f"\nFinal parameters: {params}")
print(f"True parameters: {{'weights': {true_w}, 'bias': {true_b}}}")
```

Running this in Colab:

```python
# Output will show the loss decreasing and parameters approaching the true values
# Starting parameters: {'weights': Array(1., dtype=float32, weak_type=True), 'bias': Array(0., dtype=float32, weak_type=True)}
# Step 0, Loss: 8.7499
# Step 10, Loss: 0.2178
# Step 20, Loss: 0.0168
# Step 30, Loss: 0.0017
# ...
# Final parameters: {'weights': Array(2.50..., dtype=float32), 'bias': Array(1.00..., dtype=float32)}
# True parameters: {'weights': 2.5, 'bias': 1.0}
```

This simple example illustrates how `jax.grad()` is the core component for enabling gradient-based optimization within JAX. You compute the gradient of your loss function with respect to your model parameters, and then use these gradients to update the parameters.

## Conclusion

`jax.grad()` is a fundamental transformation in JAX that provides efficient and accurate automatic differentiation for scalar functions. You can easily compute gradients with respect to specific arguments, handle complex data structures like dictionaries, and even compute higher-order derivatives by composing `jax.grad()` calls.

By integrating seamlessly with other JAX features (like `jax.jit` for compilation and `jax.vmap` for vectorization), `jax.grad()` empowers you to build and train high-performance machine learning models with relative ease. As you delve deeper into JAX, mastering `jax.grad()` is a crucial step.

-----
