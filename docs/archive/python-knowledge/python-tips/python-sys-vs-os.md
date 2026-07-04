---
title: Python `sys` vs. `os`
tags: 
    - python
    - sys
    - os
---
# Python [`sys`](https://docs.python.org/3/library/sys.html) vs. [`os`](https://docs.python.org/3/library/os.html)

### Python [`os`](https://docs.python.org/3/library/os.html) module

***os*** interacts with the **underlying operating system**.
This includes functionalities related to the file system, environment variables, and process management.

##### [`os`](https://docs.python.org/3/library/os.html) module examples of usage

- Creating, deleting, or renaming files and directories (os.mkdir(), os.remove(), os.rename()).
- Listing directory contents (`os.listdir()`).
- Getting the current working directory (`os.getcwd()`).
- Accessing or modifying environment variables (`os.environ`).
- Executing system commands (`os.system()`).

### Python [`sys`](https://docs.python.org/3/library/sys.html) module

***sys*** interacts with the Python **interpreter** and its **runtime environment**.
This module provides access to interpreter-specific variables and functions.

##### [`sys`](https://docs.python.org/3/library/sys.html) module examples of usage

- Accessing command-line arguments passed to a script (`sys.argv`).
- Controlling the Python search path for modules (`sys.path`).
- Accessing standard input, output, and error streams (`sys.stdin`, `sys.stdout`, `sys.stderr`).
- Exiting the program (`sys.exit()`).
- Retrieving information about the Python interpreter and platform (`sys.version`, `sys.platform`).

> In summary:
> ***os*** is for operating **system-level** interactions.
> ***sys*** is for Python **interpreter-level** interactions.
