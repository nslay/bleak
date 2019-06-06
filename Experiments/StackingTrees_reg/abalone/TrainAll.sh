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

#toolFlags="-I /mnt/c/Work/Source/Bleak/Modules/Common/Subgraphs -I /mnt/c/Work/Source/Bleak/Modules/Trees/Subgraphs"
toolFlags="-I C:/Work/Source/Bleak/Modules/Common/Subgraphs -I C:/Work/Source/Bleak/Modules/Trees/Subgraphs"
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

  if [ ! -f "${dir}/Config.sad" ]
  then
    continue
  fi

  echo "Info: Processing ${dir} ..."

  pushd "${dir}"
  
  layers=`basename "${dir}"`
  
  trainGraph="../../Train${layers}.sad"

  for run in `seq 1 ${numRuns}`
  do

    echo "Info: Run ${run} ..."

    runDir="Run${run}"

    mkdir -pv "${runDir}"

    pushd "${runDir}"

    rm -f bleak_*

    seed="${seedPrefix}${run}"

    "${toolCmd}" train -c ../../Training.ini -g "${trainGraph}" -s "${seed}" -I .. ${toolFlags} > TrainingLog.txt 2>&1

    popd
  done

  popd
done

