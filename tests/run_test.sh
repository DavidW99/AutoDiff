#!/usr/bin/env bash

## This script runs the test suite for the team41 project.
#pytest test_ElementFunctions.py
#pytest test_DualNumber.py
# ## run a specific test
# pytest test_ElementFunction.py -k test_sin
# ##(run last failed test)
# pytest test_ElementFunction.py --lf 

tests=(
    # test_other_things_on_root_level.py
    #tests/
    test_AutoDiff.py
    test_DualNumber.py
    test_ElementFunction.py
    test_feature.py
    # subpkg_2/test_module_3.py
    # subpkg_2/test_module_4.py
)

# Must add the module source path because we use `import cs107_package` in
# our test suite.  This is necessary if you want to test in your local
# development environment without properly installing the package.
export PYTHONPATH="$(pwd -P)/../src":${PYTHONPATH}

# decide what driver to use (depending on arguments given)
if [[ $# -gt 0 && ${1} == 'coverage' ]]; then
    driver="${@} -m unittest"
elif [[ $# -gt 0 && ${1} == 'pytest' ]]; then
    driver="${@}"
elif [[ $# -gt 0 && ${1} == 'CI' ]]; then
    # Assumes the package has been installed and dependencies resolved.  This
    # would be the situation for a customer.  Uses `pytest` for testing.
    shift
    unset PYTHONPATH
    driver="pytest ${@}"
else
    driver="python ${@} -m unittest"
fi

# run the tests
${driver} ${tests[@]}