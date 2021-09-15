#!/bin/bash

{
    git clone https://github.com/patrikhuber/eos &&
    apt-get update && 
    apt-get install git-lfs &&
    apt-get install cmake &&
    git-lfs pull && 
    pip install -r requirements.txt 

} || {

    git clone https://github.com/patrikhuber/eos &&
    sudo apt-get update && 
    sudo apt-get install git-lfs &&
    sudo apt-get install cmake &&
    git-lfs pull && 
    pip install -r requirements.txt 
}

