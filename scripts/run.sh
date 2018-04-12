#!/bin/bash

basepath=$(cd `dirname $0`; pwd)
echo $basepath
nohup $basepath/../bin/POSTAG_SEG -l -train ../Data/CTB6.0-postag/train.ctb60.nnpos -dev ../Data/CTB6.0-postag/dev.ctb60.nnpos -test ../Data/CTB6.0-postag/test.ctb60.nnpos -model model -option ../option/option.save > log 2>&1 &
tail -f log


