#!/bin/sh

for layers in 2 3
do
  outDir="${layers}"

  mkdir -pv "${outDir}"

  outConfigFile="${outDir}/Config.sad"
  
  cat << EOF > "${outConfigFile}"
numTrees1=50;
treeDepth1=5;
numTrees2=\$numTrees1 / 2;
treeDepth2=\$treeDepth1;
numTrees3=\$numTrees2 / 2;
treeDepth3=\$treeDepth2;
learningRateMultiplier=1.0;
batchSize=50;
numInputs=8;
labelIndex=8;
EOF

done

