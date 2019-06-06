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

  if [ ! -f "${dir}/validation.sad" ] || [ ! -f "${dir}/test.sad" ]
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
  
    #bestRes=10000
    #bestR2=0
    bestLoss=10000;
    bestWeightsFile=""
  
    for weightsFile in bleak_*
    do
      if [ ! -f "${weightsFile}" ] || (echo "${weightsFile}" | grep -q -- "-lock")
      then
        continue
      fi
  
      #echo -n "Evaluating ${weightsFile} ... "
  
      line=`"${toolCmd}" test -n 1044 -g ../validation.sad -w "${weightsFile}" | grep "running average" | tail -n 1`
      #res=`echo "${line}" | awk '{ gsub("\r",""); gsub(",",""); print $(NF-1) }'`
      #R2=`echo "${line}" | awk '{ gsub("\r",""); gsub(",",""); print $NF }'`
      loss=`echo "${line}" | awk '{ gsub("\r",""); gsub(",",""); print $NF }'`
      loss=`echo "${loss}" | tr 'e' 'E'`
  
      #echo "${weightsFile}, ${res}, ${R2}"
  
      #isBest=`echo "scale=5; ${res} < ${bestRes}" | bc`
      isBest=`echo "scale=5; ${loss} < ${bestLoss}" | bc`
  
      if [ "${isBest}" -eq 1 ]
      then
        #bestRes="${res}"
        #bestR2="${R2}"
        bestLoss="${loss}"
        bestWeightsFile="${weightsFile}"
      fi
  
      #exit
    done
    
    #echo "${dir}/${runDir}: Best validation res = ${bestRes}, best R2 = ${bestR2}, best weightsFile = ${bestWeightsFile}"
    echo "${dir}/${runDir}: Best validation loss = ${bestLoss}, best weightsFile = ${bestWeightsFile}"
    
    line=`"${toolCmd}" test -n 1044 -g ../test.sad -w "${bestWeightsFile}" | grep "mean residual" | tail -n 1`
    testRes=`echo "${line}" | awk '{ gsub("\r",""); gsub(",",""); print $(NF-1) }'`
    testR2=`echo "${line}" | awk '{ gsub("\r",""); gsub(",",""); print $NF }'`
  
    echo "${dir}/${runDir}: Test res = ${testRes}, test R2 = ${testR2}"

    popd
  done
  
  popd
done

