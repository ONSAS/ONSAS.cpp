clear
cd src
make
echo " compilation done."
cp timeStepIteration.lnx ../../ONSAS_repo/sources/
echo " binary copied."
make clean
cd ..
