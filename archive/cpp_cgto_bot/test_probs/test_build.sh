#!/bin/bash

# Build script for the test program
g++ -std=c++17 -I. -I./include -I./libs/skeleton/include \
    test_winprob.cpp -o test_winprob

if [ $? -eq 0 ]; then
    echo "Build successful! Run with: ./test_winprob"
else
    echo "Build failed!"
    exit 1
fi

