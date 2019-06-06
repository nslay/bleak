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
  
  for runDir in Run*
  do
    if [ ! -d "${runDir}" ]
    then
      continue
    fi
    
    pushd "${runDir}"
  
    bestLoss=100000
    bestWeightsFile=""
  
    for weightsFile in bleak_*
    do
      if [ ! -f "${weightsFile}" ] || (echo "${weightsFile}" | grep -q -- "-lock")
      then
        continue
      fi
  
      #echo -n "Evaluating ${weightsFile} ... "
  
      line=`"${toolCmd}" test -n 1044 -g ../../Validation.sad -I .. ${toolFlags} -w "${weightsFile}" | grep "running average" | tail -n 1`

      #echo "${weightsFile}: ${line}"

      loss=`echo "${line}" | awk '{ gsub("\r",""); print $NF }'`

      isBest=`echo "scale=5; ${loss} < ${bestLoss}" | bc`
  
      if [ "${isBest}" -eq 1 ]
      then
        bestLoss="${loss}"
        bestWeightsFile="${weightsFile}"
      fi
  
      #exit
    done
    
    echo "${dir}/${runDir}: Best validation loss = ${bestLoss}, best weightsFile = ${bestWeightsFile}"
    
    line=`"${toolCmd}" test -n 1044 -g ../../Test.sad -I .. ${toolFlags} -w "${bestWeightsFile}" | grep "running accuracy" | tail -n 1`
    testAcc=`echo "${line}" | awk '{ gsub("\r",""); print $NF }'`
  
    echo "${dir}/${runDir}: Test accuracy = ${testAcc}"

    popd
  done
  
  popd
done

