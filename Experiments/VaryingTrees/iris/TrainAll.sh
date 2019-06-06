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

seedPrefix="iris"
searchCmd="bleakTool"
toolCmd=`GetExeName "${searchCmd}"`
numShuffles=5

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

  if [ ! -f "${dir}/Run1_1/Config.sad" ]
  then
    continue
  fi

  echo "Info: Processing ${dir} ..."

  pushd "${dir}"

  for exp in `seq 1 ${numShuffles}`
  do
    for fold in `seq 1 3`
    do
      run="${exp}_${fold}"

      echo "Info: Run ${run} ..."

      runDir="Run${run}"

      mkdir -pv "${runDir}"

      pushd "${runDir}"

      rm -f bleak_*

      seed="${seedPrefix}${run}"

      "${toolCmd}" train -c ../../Training.ini -g ../../Train.sad -I . ${toolFlags} -s "${seed}" > TrainingLog.txt 2>&1

      popd

      #exit
    done
  done

  popd
done

