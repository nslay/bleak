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

seedPrefix="usps"
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
  
    #bestLoss=100000
    bestAcc=-1
    bestWeightsFile=""
  
    for weightsFile in bleak_*
    do
      if [ ! -f "${weightsFile}" ] || (echo "${weightsFile}" | grep -q -- "-lock")
      then
        continue
      fi
  
      #echo -n "Evaluating ${weightsFile} ... "
  
      #line=`"${toolCmd}" test -n 2007 -g ../../Validation.sad -I ../../../hackedSubgraphs -I .. ${toolFlags} -w "${weightsFile}" | grep "running average" | tail -n 1`
      #loss=`echo "${line}" | awk '{ gsub("\r",""); print $NF }'`
      line=`"${toolCmd}" test -n 2007 -g ../../Validation.sad -I ../../../hackedSubgraphs -I .. ${toolFlags} -w "${weightsFile}" | grep "running accuracy" | tail -n 1`
      acc=`echo "${line}" | awk '{ gsub("\r",""); print $NF }'`

      #echo "${loss}"
      
      #isBest=`echo "scale=5; ${loss} < ${bestLoss}" | bc`
      isBest=`echo "scale=5; ${acc} > ${bestAcc}" | bc`
  
      if [ "${isBest}" -eq 1 ]
      then
        #bestLoss="${loss}"
        bestAcc="${acc}"
        bestWeightsFile="${weightsFile}"
      fi
  
      #exit
    done
    
    #echo "${dir}/${runDir}: Best validation loss = ${bestLoss}, best weightsFile = ${bestWeightsFile}"
    echo "${dir}/${runDir}: Best accuracy = ${bestAcc}, best weightsFile = ${bestWeightsFile}"
    
    line=`"${toolCmd}" test -n 2007 -g ../../Test.sad -I ../../../hackedSubgraphs -I .. ${toolFlags} -w "${bestWeightsFile}" | grep "running accuracy" | tail -n 1`
    testAcc=`echo "${line}" | awk '{ gsub("\r",""); print $NF }'`
  
    echo "${dir}/${runDir}: Test accuracy = ${testAcc}"

    popd

    #exit
  done
  
  popd
done

