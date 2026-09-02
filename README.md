# Hyperbolic Attention 

A small transformer whose attention opperates on a lorentz hyperboloid , trained on python source code parsed as AST's to encode the graph structure of code onto a hyperboloid.  The exponential space on a hyperbolic geometry is the benefit that is required for deeply nested ASTs . 

## About the Project 
This is a build project and not a research project per se. The objective of this project is to be mathematically correct , explore the beaty of the hyperbolic geometry and make an attempt to explaining it with clear latex and animations in the documentation ; train the model without experiencing Nan or infinity explosions in the bf16 territory of the TPU stack given by Kaggle . First write a naive JAX reference , then attempt to optimize with custom TPU kernels . This project starts with a write up wth the mathematics , references and design constraints and creates a doc site first . 

## Planned Layout 

```
src/
    /geometry 
        lorentz.py - the python library file with geometry operations for the hyperboloid model implemented as classes or functions making use of OOP and functional programming 
    /model 
        layers.py - the linear layer , residual , normalization 
        model.py - embeddings , blocks , attention heads , loss 
        data.py - tokenizer , tree-parser 
    /kernels 
        flash_lorentz.py - custom pallas kernels for later after making a reference implementation in JAX
tests/
    tests for the code written above 
kaggle/
    kaggle.py - this is the combined script that is used in kaggle to train the model on TPUs , combining all the reusable code written above 
proofs/
    lean/
        this is going to contain the lean proofs for the claims that will be made in the documentation to formally verify the properties that are found in the documentation 
    rocq/
        same as lean
results/ 
    this is going to contain the results from the runs 
archive/ (the failed attempt from before for reference)
```

## Backstory 

Now you might be thinking , how did a second year computer engineering student at UWaterloo come up with the idea of using hyperbolic spaces for representing code and simulating attention on these embeddings . 

Here is why , This all started with a youtube video from Quanta Magazine : [Biggest Breakthroughs in Mathematics: 2025](https://www.youtube.com/watch?v=hRpcWpAeWng). 

One of the three sections on this video covered Nalini Anantharaman and Laura Monk's proof
that almost every sufficiently complex hyperbolic surface has a near-optimal
spectral gap, built on techniques Maryam Mirzakhani developed for studying
random hyperbolic surfaces. 

I could not follow along with the theory , but what stick to me was that there is a geometry where parallel lines do not meet , circles could hold much more volume than they are supposed to . 

This and a lot more research into hyperbolic spaces , lead me to a paper by [Nickel and Kiela, "Poincaré Embeddings for Learning Hierarchical Representations" (NeurIPS 2017)](https://arxiv.org/abs/1705.08039). This paper did an experiment where they embed hierarchical graph data (WordNet's hypernym tree) and social network graphs among them, in the Poincaré ball. This blew my mind as this could represent enormous amounts of data in 5 to 10 dimensions, where euclidean spaces take 100s of dimensions to do the same . Then, I thought of data types that represent a lot of data with a heirarchy , this led me to code : Code is inherently heirarchical as there is sub-objects of objects , with sub-functions inside functions that call other sub-functions . Then I ran experiements on the standard library of python and got good initial results with mathematical errors I didn't know I was making , after a long time spent analysing that result , I came across the errors , and restarted the project with an effort to first formally verify the properties that I am using and then rewrite the code for the equations that I discover in the literature review .