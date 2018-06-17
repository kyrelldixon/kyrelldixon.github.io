---
layout: post
title:  "Understanding List Comprehensions in Python"
date:   2018-06-15
description: "A beginner overview of what a list comprehension is and why you should be using them."
categories:
  - Python
tags:
  - Python
---

Since you’re here that probably means you’re trying to get a grasp of this weird, pythonic thing called a list comprehension. It also means you’re getting deeper into a “Pythonic” way of doings things, which is always a good thing.

This article explains *what* a list comprehension is and *how* you go about using them. I also get into how you can take a simple for loop and condense it into a list comprehension and why you might want to do that. With that being said, let’s get into the first question.

## What Are List Comprehensions?

In simple terms, list comprehensions are a way of iterating over a list and doing stuff like modifying the values or filtering out elements. They typically come in the form `[ expression for item in list if conditional ]` . The main parts took take notice of are:

- The Expression: What you do to the item (this might be nothing)
- The Loop: Where you choose what to iterate over and how
- The Conditional: Where you decide what items you keep

Let’s look at an example of the most common way to iterate, the for loop, to get a better understanding of what that really means.

## The For Loop

```python
array = list(range(10))
for i, el in enumerate(array):
  array[i] = el * 2
```

In this example we take the element at the current index, double it, and assign back to the current position so that `array` now contains:

`[0, 2, 4, 6, 8, 10, 12, 14, 16, 18]`

So what’s going on here? Well the first line `array = list(range(10))` makes a list that contains the elements 0-9. Since we’re going for concise but readable I chose this syntax, but you could just do:

```python
array = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]`
```

**Pro-tip:** The `range()` function can do other cool things like count backwards or in multiples of 2.  More examples can be found in the docs ([here](https://docs.python.org/3/library/stdtypes.html#range), but here’s the two I just mentioned:

```python
# Counting backwards
for i in range(10,0,-1):
  print(i)

# Counting by 2's
for i in range(0,11,2):
  print(i)
```

After range we have  `for i, el in enumerate(array)` . The `enumerate()` function takes an array as input and returns and iterator that has two values—the index and the element at the current iteration.

Here’s another example for me clarity:

```python
array = [0, 'thing', 328, (78,23), {'person': 'John Adams'}]
for i, el in enumerate(array):
  print(f'{i}: {el}')

'''
Outputs:
0: 0
1: thing
2: 328
3: (78, 23)
4: {'person': 'Jogn Adams'}
'''
```

As you can see above, `i` is just the index in the array, and `el` is the value at that index. We used a [formatted string literal](https://docs.python.org/3.6/reference/lexical_analysis.html#f-strings)  or an “f-string” (only in Python 3.6+) to make that print out nicely.

**Pro-tip:** Python lists don’t care about data types. They hold whatever you put in them! This is really great for quick scripts since you don’t really need to think about it, but it something you should be careful with.

While iterating through a list is not difficult to do, it could be done in a much nicer format. That’s where list comprehensions come in.

## The List Comprehension Way

By using a list comprehension, we can modify and assign it list in a fast, clean, and simple way. Not only that, but regardless of whether you decide to use a list comprehension yourself, understanding how list comprehensions work will prepare you for the next time you run into it someone *else’s* code. Now that you’re aware of the benefits, let’s get into how to use one.

To double the values of a list with a list comprehension, here’s all you need to do:

```python
array = [2 * x for x in array]
```

One line of code and no `enumerate()` function necessary. No only that, but a list comprehension is typically *faster* than it’s for loop counterpart.

Another thing we can do fairly easily with a list comprehension is add *filtering.* Basically, we can take the values in a list and only keep what matches our criteria. As an example, say we have a list of names and we want to only take names that start with a capital J. We can easily do this with a list comprehension.

```python
names = ['John','Sam','Jude','Ky','James']
J_names = [name for name in names if name[0] == 'J']
# Output: ['John', 'Jude', 'James']
```

There are several use cases for a list comprehension, and I invite anyone who has more ideas to mention some use cases below!

## To Wrap it Up

The important thing to take away is list comprehensions can be intimidating, but they don’t have to be. Even if you never use them, understanding the syntax and what’s possible in Python will only help you—especially when looking at someone else’s code.

If this helped you out, or if you see something I could improve on, let me know! 
