#!/usr/bin/env bash
percentage=$(./run_test.sh pytest --cov=../AutoDiff | grep 'TOTAL' | awk '{print $4; }' |sed 's/[^0-9]*//g')
num=$(($percentage+0))
echo "passing percentage: "$num
if (( num > 90 ))
then 
    echo 'more than 90% coverage'
    exit 0
else 
    echo 'less than 90% coverage'
    exit 1
fi

## ./run_test.sh pytest --cov=../AutoDiff --cov-report term-missing 