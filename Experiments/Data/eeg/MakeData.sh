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

GetConditionLabel() {
  eegFile_=$1

  awk 'BEGIN { found=0 }
    /S1/ { found=1; print 0; exit }
    /S2 match/ { found=1; print 1; exit }
    /S2 nomatch/ { found=1; print 2; exit }
    END {
      if (!found)
        print -1
    }' "${eegFile_}"

}

GetAlcoholicLabel() {
  eegFile_=`basename "$1"`
  case "${eegFile_}"
  in 
  *co?c*)
    echo "0"
    ;;
  *co?a*)
    echo "1"
    ;;
  *)
    echo "-1"
    ;;
  esac
}

makeImageCmd=`GetExeName bleakMakeEEGImage`
rootDir="data"

if [ -z "${makeImageCmd}" ]
then
  echo "Error: bleakMakeEEGImage must be in PATH." 1>&2
  exit 1
fi

if [ ! -d "${rootDir}" ]
then
  echo "Error: Root folder '${rootDir}' does not exist." 1>&2
  exit 1
fi

pushd "${rootDir}"

mkdir -pv images

csvFile="imageInfo.csv"
echo "imageFile,conditionLabel,alcoholicLabel" > "${csvFile}"

find . -iname "*.rd*" -a '!' -iname "*.mha" -type f |\
while read eegFile
do
  if grep -q "err" "${eegFile}"
  then
    continue
  fi

  conditionLabel=`GetConditionLabel "${eegFile}"`
  alcoholicLabel=`GetAlcoholicLabel "${eegFile}"`  

  #echo "${eegFile}"

  base=`basename "${eegFile}"`
  outputFile="images/${base}.mha"
  "${makeImageCmd}" "${eegFile}" "${outputFile}"

  echo "${outputFile},${conditionLabel},${alcoholicLabel}" >> "${csvFile}"
done
