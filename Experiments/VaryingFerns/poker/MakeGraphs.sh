#!/bin/sh

for numTrees in 1 10 50 100
do
  for treeDepth in 1 3 5 7 10
  do
    outDir="${numTrees}_${treeDepth}"

    mkdir -pv "${outDir}"

    outConfigFile="${outDir}/Config.sad"
    
    cat << EOF > "${outConfigFile}"
numTrees=${numTrees};
treeDepth=${treeDepth};
learningRateMultiplier=1.0;
batchSize=53;
numInputs=10;
labelIndex=10;
numClasses=10;
EOF
    
  done
done

