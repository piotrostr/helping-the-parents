#!/bin/bash

git clone https://github.com/piotrostr/helping-the-parents && 
    cd helping-the-parents && 
    git clone https://github.com/swook/EVE &&
    git clone https://github.com/patrikhuber/eos && 
    sudo apt-get update && 
    sudo apt-get install git-lfs &&
    git-lfs pull && 
    pip install -r requirements.txt 

