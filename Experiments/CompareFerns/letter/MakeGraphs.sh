#!/bin/sh

for numTrees in 100
do
  for treeDepth in 10
  do
    outDir="${numTrees}_${treeDepth}"

    mkdir -pv "${outDir}"

    outConfigFile="${outDir}/Config.sad"
    
    cat << EOF > "${outConfigFile}"
numTrees=${numTrees};
treeDepth=${treeDepth};
learningRateMultiplier=1.0;
#batchSize=503;
batchSize=53;
numInputs=16;
labelIndex=0;
numClasses=26;
EOF
    
  done
done

