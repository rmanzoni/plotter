#!/bin/bash

#-------------------------------#
# 
# This batch script simply calls the limit script to be run on the batch
#
#-------------------------------#


python produce_limits.py --mass ${1} --coupling ${2} --run_blind ${3} --version ${4} 


