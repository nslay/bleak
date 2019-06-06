#!/usr/bin/awk -f

BEGIN {

}

/Test res/ {
  runKey=$1 

  delete tmp
  split(runKey, tmp, "/")

  config=tmp[1]
  run=tmp[2]

  if (!(config in testMap)) {
    testMap[config][0] = $NF
  }
  else {
    i = length(testMap[config])
    testMap[config][i] = $NF
  }

}

/Test accuracy/ {
  runKey=$1 

  delete tmp
  split(runKey, tmp, "/")

  config=tmp[1]
  run=tmp[2]

  if (!(config in testMap)) {
    testMap[config][0] = 1.0 - $NF 
  }
  else {
    i = length(testMap[config])
    testMap[config][i] = 1.0 - $NF 
  }
}

END {
  for (config in testMap) {
    accMean = 0
    accStd = 0
    count=0
    for (accKey in testMap[config]) {
      acc = testMap[config][accKey]

      ++count
      delta = acc - accMean
      accMean += delta/count
      accStd += delta*(acc - accMean)
    }

    if (accStd < 0)
      accStd = 0

    if (count < 2)
      accStd = "nan"
    else
      accStd = sqrt(accStd/(count-1))

    
    print config ": " accMean " +/- " accStd
  }
}

