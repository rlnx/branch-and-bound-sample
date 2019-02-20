# /bin/bash

cmd='python -u test_branch_and_bound.py'

$cmd --collection=stack --upper=default
$cmd --collection=stack --upper=distance
$cmd --collection=stack --upper=default+distance

$cmd --collection=queue --upper=default
$cmd --collection=queue --upper=distance
$cmd --collection=queue --upper=default+distance

$cmd --collection=uheap --upper=default
$cmd --collection=uheap --upper=distance
$cmd --collection=uheap --upper=default+distance

$cmd --collection=lheap --upper=default
$cmd --collection=lheap --upper=distance
$cmd --collection=lheap --upper=default+distance

$cmd --collection=ulheap --upper=default
$cmd --collection=ulheap --upper=distance
$cmd --collection=ulheap --upper=default+distance

$cmd --collection=ulheap --upper=aneal
