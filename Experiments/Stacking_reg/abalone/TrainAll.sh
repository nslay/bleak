#!/bin/bash

GetExeName() {
  base_="$1"

  for exe_ in "${base_}" "${base_}.exe"
  do
    if which "${exe_}" > /dev/null 2>&1
    then
      echo "${exe_}"
      return 0
    fi
  done

  return 1
}

seedPrefix="abalone"
searchCmd="bleakTool"
toolCmd=`GetExeName "${searchCmd}"`
numRuns=10

if [ -z "${toolCmd}" ]
then
  echo "Error: ${searchCmd} is not in PATH." 1>&2
  exit 1
fi

for dir in *
do
  if [ ! -d "${dir}" ]
  then
    continue
  fi

  if [ ! -f "${dir}/train.sad" ]
  then
    continue
  fi

  numLayers=`basename "${dir}"`

  echo "Info: Processing ${dir} ..."

  pushd "${dir}"

  for run in `seq 1 ${numRuns}`
  do

    echo "Info: Run ${run} ..."

    runDir="Run${run}"

    mkdir -pv "${runDir}"

    pushd "${runDir}"

    rm -f bleak_*

    seed="${seedPrefix}${run}"
    configFile="../../Training${numLayers}.ini"

    "${toolCmd}" train -c "${configFile}" -g ../train.sad -s "${seed}" > TrainingLog.txt 2>&1

    popd
  done

  popd
done

