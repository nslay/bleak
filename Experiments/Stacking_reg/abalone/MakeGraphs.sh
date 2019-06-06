#!/bin/sh

for numLayers in 2 3
do
  outDir="${numLayers}"

  mkdir -pv "${outDir}"

  outTrainGraph="${outDir}/train.sad"
  outValidationGraph="${outDir}/validation.sad"
  outTestGraph="${outDir}/test.sad"

  trainTemplate="TrainTemplate${numLayers}.sad"
  testTemplate="TestTemplate${numLayers}.sad"

  cat - "${trainTemplate}" << EOF > "${outTrainGraph}"
learningRateMultiplier=1.0;
dataBaseName="abalone_train";
EOF

  cat - "${testTemplate}" << EOF > "${outValidationGraph}"
dataBaseName="abalone_validation";
EOF

  cat - "${testTemplate}" << EOF > "${outTestGraph}"
dataBaseName="abalone_test";
EOF
done

