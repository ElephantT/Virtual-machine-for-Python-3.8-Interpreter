## Virtual Machine for Python Interpreter (written in Python!!!)

### Intro

When you are using any programming language you usually want to know a little bit more, than just what are standard libraries and how to write code. Sometimes you need a deeper unserstanding what is actually going on when you execute or build your program. Anyway, even if you never questioned yourself how exactly a computer can manage a given program code, it will be usefull for everyone to understand this process. Different languages use different ways, this is especially true regarding either language is being interpreted or compiled.

For Python execution of a programm runs like this:
* you have some code object (a file, or a lot of files with code)
* first everything is tokenized (in Python you can do this with module "tokenize", written in C)
* then your code is parsed with given tokens using the grammar and syntax of Python which helps to build abstract syntax tree (modules "parser" + "token")
* now your computer can run this code using tree, for languages as C, Go ... computer will compile it to machine code
* for Python, it will be complied to byte-code (the difference is, that it will be ran not by your machine, but by virtual machine (in this project we will create one))


As you know Python has exac and eval methods, which can execute the dynamically created programm, which is either a string or a code object. 

### The whole idea of this project

To get deeper understanding of Python's byte-code by creating your own virtual machine which can execute every byte-code operation (for Python 3.8). This virtual machine will use Python language only, but some Python's functions and modules will be blocked (more details below). During the development of this project I understood how exactly Python runs written code, how it breaks code into instructions, and what are these instructions (for creating virtual machine it is needed to implement each one of them)

### Details about task and implementation 

**exec**, **eval**, **inspect**, **FunctionType** are not allowed!!!
Also you can't do anything like that:
```
        def f():
            pass
        f.__code__ = code
        f()
```
Why we have such restrictions? By using any part of it you will use native Python interpreter, which we want to create by ourselves without cheats :) 

### How to run tests and some useful operations:

`vm_runner.py` - helps with debug

#### how to run one test:

```bash
$ pytest test_public.py::test_all_cases[simple] -vvv
```

#### how to run all tests:

```bash
$ pytest test_public.py -vvv --tb=no
```

### Very useful links
* dis module documentation: https://docs.python.org/release/3.8.5/library/dis.html#module-dis. There you can find all the bytcode operations and how are they implemented in Python.
* An academic project for Python 2.7 and Python 3.3: https://github.com/nedbat/byterun. A lot of comments, will help to start with project
Also there is detailed information about it here: http://www.aosabook.org/en/500L/a-python-interpreter-written-in-python.html.
I would suggest to read it before you will do anything, cause it helps to understand what you are actually doing and what you will need to do for whole python language coverage
* Source code of the native interpreter for Python 3.8: https://github.com/python/cpython/blob/3.8/Python/ceval.c.

### How To Use

Put everything in one directory ('vw' for example - cause we are building our own virtual machine for Python interpreter). There are already about 350 cases of codes, which cover whole 3.8 python version, so you don't need to actualy make any new test for yourself. All the code you want to change is in vm.py. Good luck!


### Thanks to Yandex School of Data Analysis and its Python's course team, which created this project for us - students.
All the information and linkes were given by them, but us (students), each one on his own have created a virtual machine for Python byte-code. Mine vm has passed 200+ cases from about 350, that covers a big chunk of whole Python 3.8