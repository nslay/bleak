#!/bin/sh

numShuffles=5
numFolds=3

for numTrees in 1 10 50 100
do
  for treeDepth in 1 3 5 7 10
  do
    outDir="${numTrees}_${treeDepth}"

    mkdir -pv "${outDir}"
    
    for exp in `seq 1 ${numShuffles}`
    do
      for fold in `seq 1 ${numFolds}`
      do
        run="${exp}_${fold}"
        
        mkdir -pv "${outDir}/Run${run}"
        
        outConfigFile="${outDir}/Run${run}/Config.sad"
        
        trainFold=`expr '(' ${fold} % ${numFolds} ')' + 1`
        validationFold=`expr '(' ${trainFold} % ${numFolds} ')' + 1`
        
        cat << EOF > "${outConfigFile}"
numTrees=${numTrees};
treeDepth=${treeDepth};
learningRateMultiplier=1.0;
batchSize=53;
numInputs=4;
labelIndex=4;
numClasses=3;
trainFoldName="iris_${exp}_fold${trainFold}";
testFoldName="iris_${exp}_fold${fold}";
validationFoldName="iris_${exp}_fold${validationFold}";
EOF
      done
    done
  done
done

