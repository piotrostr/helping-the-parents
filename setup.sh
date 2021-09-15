#!/bin/bash

{
    sudo apt-get update && 
    sudo apt-get install git-lfs &&
    sudo apt-get install cmake &&
    sudo apt-get install python3-opencv &&
    git clone https://github.com/patrikhuber/eos &&
    git-lfs pull && 
    pip install -r requirements.txt 

} || {

    apt-get update && 
    apt-get install git-lfs &&
    apt-get install cmake &&
    apt-get install python3-opencv &&
    git clone https://github.com/patrikhuber/eos &&
    git-lfs pull && 
    pip install -r requirements.txt 
}

