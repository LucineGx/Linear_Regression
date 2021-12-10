# Linear Regression
*42 school project*

This project is composed of two program: one is used to train the prediction model, the other to actually predict a result.

## Subject

Impletement a liner regression algorythm on one element.

## Training program

python3 train.py -h

## Prediciton program

python3 predict.py -h

## Installation

If you don't have a python environment configured yet, you can follow the instruction on the pyenv github page https://github.com/pyenv/pyenv#installation until you succesfully run 

<code>pyenv init -</code>

You'll need to install python 3.6 dependencies: https://devguide.python.org/setup/#build-dependencies. Then run

<code>pyenv install 3.6.13</code>

If you need a new virtual environment, make sure to install pyenv-virtualenv, by following the instruction on https://github.com/pyenv/pyenv-virtualenv

Then you can create the project environment

<code>pyenv virtualenv 3.6.13 LinearRegression_3_6_13</code>

and install dependencies

<code>pip install -r requirements.txt</code>