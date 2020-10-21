clear
cd src
make
echo " compilation done."
cp timeStepIteration.lnx ../../octave_src_repo/sources/
echo " binary copied."
make clean
cd ..
