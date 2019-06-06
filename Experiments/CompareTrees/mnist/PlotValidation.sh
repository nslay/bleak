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

maxIterations=40000
stepSize=500
seedPrefix="mnist"
searchCmd="bleakTool"
toolCmd=`GetExeName "${searchCmd}"`

if [ -z "${toolCmd}" ]
then
  echo "Error: ${searchCmd} is not in PATH." 1>&2
  exit 1
fi

if [ $# -ne 1 ]
then
  echo "Usage: $0 expdDir"
  exit 1
fi

dir=$1

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

for runDir in Run1
do
  if [ ! -d "${runDir}" ]
  then
    continue
  fi
  
  pushd "${runDir}"

  for weightsIndex in `seq 0 ${stepSize} ${maxIterations}`
  do
    weightsFile="bleak_${weightsIndex}"

    if [ ! -f "${weightsFile}" ] || (echo "${weightsFile}" | grep -q -- "-lock")
    then
      continue
    fi

    #echo -n "Evaluating ${weightsFile} ... "

    line=`"${toolCmd}" test -n 10000 -g ../../Validation.sad -I ../../../hackedSubgraphs -I .. ${toolFlags} -w "${weightsFile}" | grep "running average" | tail -n 1`
    loss=`echo "${line}" | awk '{ gsub("\r",""); print $NF }'`

    printf "${weightsIndex}\t${loss}\n"

    #line=`"${toolCmd}" test -n 10000 -g ../../Validation.sad -I ../../../hackedSubgraphs -I .. ${toolFlags} -w "${weightsFile}" | grep "running accuracy" | tail -n 1`
    #acc=`echo "${line}" | awk '{ gsub("\r",""); print $NF }'`

    #printf "${weightsIndex}\t${acc}\n"

    
    #exit
  done
  
  popd

  #exit
done

popd

