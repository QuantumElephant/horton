#!/usr/bin/env bash

source tools/qa/buildkite_common.sh
checkout_merge_commit

echo "--- Basic source tests"
./tools/qa/check_names.py

echo "--- Build refatoms"
rm -rf data/refatoms/*.h5 data/refatoms/*.tar.bz2
make -C data/refatoms/

echo "--- Build Cython files & HORTON"
./cleanfiles.sh
rm -rf installation
./setup.py install --prefix=`pwd`/installation

echo "--- Uploading build artifact"
tar -zcvf horton-install.tar.gz `pwd`/installation
buildkite-agent artifact upload horton-install.tar.gz

if [ "$BUILDKITE_PULL_REQUEST" != "false" ]; then
    #rebuild ancestor
    echo "--- Build ancestor refatoms"
    rm -rf data/refatoms/*.h5 data/refatoms/*.tar.bz2
    make -C data/refatoms

    echo "--- Build ancestor Cython files & HORTON"
    ./cleanfiles.sh
    rm -rf installation
    ./setup.py install --prefix=`pwd`/ancestor_installation

    echo "--- Uploading ancestor build artifact"
    tar -zcvf horton-ancestor-install.tar.gz `pwd`/ancestor_installation
    buildkite-agent artifact upload horton-ancestor-install.tar.gz
fi