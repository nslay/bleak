#!/bin/bash

# You will need to download ProstateX 1 data before hand!

prostateXDir="C:/Work/ProstateX/PROSTATEx"
outT2wRoot="T2W"
outAdcRoot="ADC"
outBValueRoot="B1500"
logRoot="Logs"

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

# Usually the best scan is the last scan
GetLatestSeriesDir() {
  latestSeriesDir_=`find "$1" -mindepth 1 -maxdepth 1 -type d -printf "%f\n" | sort -n -r | head -n 1`
  if [ -n "${latestSeriesDir_}" ]
  then
    echo "${1}/${latestSeriesDir_}"
  fi
}

FindT2WDir() {
  tmpDir_=`find "$1" -mindepth 1 -maxdepth 1 -type d -iname "*t2*tra*" -print -quit`
  if [ -n "${tmpDir_}" ]
  then
    GetLatestSeriesDir "${tmpDir_}"
  fi
}

FindADCDir() {
  tmpDir_=`find "$1" -mindepth 1 -maxdepth 1 -type d '(' -iname "*adc*" -a '!' -iname "*calc*" ')' -print -quit`
  if [ -n "${tmpDir_}" ]
  then
    GetLatestSeriesDir "${tmpDir_}"
  fi
}

FindBValueDir() {
  tmpDir_=`find "$1" -mindepth 1 -maxdepth 1 -type d '(' -iname "*diff*" -a '!' -iname "*calc*" ')' -print -quit`
  if [ -n "${tmpDir_}" ]
  then
    GetLatestSeriesDir "${tmpDir_}"
  fi
}

sortCommand=`GetExeName SortDicomFiles`
bvalueCommand=`GetExeName ComputeBValue`
normCommand=`GetExeName RicianNormalization`
alignCommand=`GetExeName AlignVolumes`

if [ -z "${sortCommand}" ]
then
  echo "Error: Missing SortDicomFiles." 1>&2
  exit 1
fi

if [ -z "${bvalueCommand}" ]
then
  echo "Error: Missing ComputeBValue." 1>&2
  exit 1
fi

if [ -z "${normCommand}" ]
then
  echo "Error: Missing RicianNormalization." 1>&2
  exit 1
fi

if [ -z "${alignCommand}" ]
then
  echo "Error: Missing AlignVolumes." 1>&2
  exit 1
fi

echo "Info: Step 1) Stage the ProstateX cases..."

stageDir="Raw"
mkdir -pv "${stageDir}"

if [ ! -f "${stageDir}/SORTED" ]
then
  "${sortCommand}" -c -r "${prostateXDir}" "${stageDir}/<patient id>/<study date>/<series description>/<series number>/<file>"
  touch "${stageDir}/SORTED"
else
  echo "Info: Already sorted. Skipping..."
fi

echo "Info: Step 2) Preprocess DICOMs..."

mkdir -pv "${outT2wRoot}" "${outAdcRoot}" "${outBValueRoot}" "${logRoot}"

#for patientDir in "${stageDir}"/ProstateX-0002
for patientDir in "${stageDir}"/*
do
  if [ ! -d "${patientDir}" ]
  then
    continue
  fi

  patientId=`basename "${patientDir}"`

  for dateDir in "${patientDir}"/*
  do
    if [ ! -d "${dateDir}" ]
    then
      continue
    fi

    date=`basename "${dateDir}"`

    # Not a date?
    if ! date -d "${date}" > "/dev/null" 2>&1  
    then
      continue
    fi

    t2wDir=`FindT2WDir "${dateDir}"`

    if [ ! -d "${t2wDir}" ]
    then
      echo "Error: Could not find T2W for patient '${patientId}'. Skipping..." 1>&2
      continue
    fi

    adcDir=`FindADCDir "${dateDir}"`

    if [ ! -d "${adcDir}" ]
    then
      echo "Error: Could not find ADC for patient '${patientId}'. Skipping..." 1>&2
      continue
    fi

    bvalueDir=`FindBValueDir "${dateDir}"`

    if [ ! -d "${bvalueDir}" ]
    then
      echo "Error: Could not find b-value images for patient '${patientId}'. Skipping..." 1>&2
      continue
    fi
    
    # Uncomment this to check that everything for every patient is found
    #continue
    
    logDir="${logRoot}/${patientId}/${date}"
    rm -rf "${logDir}"
    
    mkdir -pv "${logDir}"
    
    b1500Dir="${dateDir}/B1500"
    rm -rf "${b1500Dir}"
    
    # Compute b-1500
    "${bvalueCommand}" -b 1500 -o "${b1500Dir}" mono "${bvalueDir}" > "${logDir}/computeBValue.log" 2>&1
    
    normB1500Dir="${dateDir}/NormalizedB1500"
    rm -rf "${normB1500Dir}"

    # Normalize b-1500
    "${normCommand}" "${b1500Dir}" "${normB1500Dir}" > "${logDir}/normBValue.log" 2>&1
    
    normT2wDir="${dateDir}/NormalizedT2W"
    
    rm -rf "${normT2wDir}"
    
    # Normalize the T2W
    "${normCommand}" "${t2wDir}" "${normT2wDir}" > "${logDir}/normT2w.log" 2>&1
    
    # Align everything to the T2W
    alignedRoot="${dateDir}/Aligned"
    alignedT2wDir="${alignedRoot}/`basename "${normT2wDir}"`"
    alignedAdcDir="${alignedRoot}/`basename "${adcDir}"`"
    alignedB1500Dir="${alignedRoot}/`basename "${normB1500Dir}"`"
    
    rm -rf "${alignedRoot}"
    
    "${alignCommand}" -r 0.5x0.5x0 -o "${alignedRoot}" "${normT2wDir}" "${adcDir}" "${normB1500Dir}" > "${logDir}/alignVolumes.log" 2>&1
    
    outT2wDir="${outT2wRoot}/${patientId}/${date}"
    outAdcDir="${outAdcRoot}/${patientId}/${date}"
    outBValueDir="${outBValueRoot}/${patientId}/${date}"
    
    if grep -i -r -E 'nan|error' "${logDir}"
    then
      echo "Error: Failed to process a ${patientId}/${date}." 1>&2
      echo "Log folder: ${logDir}"
      exit 1
    fi
    
    rm -rf "${outT2wDir}" "${outAdcDir}" "${outBValueDir}"
    
    # Copy everything to output folder
    "${sortCommand}" -e "${alignedT2wDir}" "${outT2wDir}/<z coordinate>.dcm"
    "${sortCommand}" -e "${alignedAdcDir}" "${outAdcDir}/<z coordinate>.dcm"
    "${sortCommand}" -e "${alignedB1500Dir}" "${outBValueDir}/<z coordinate>.dcm"
    
    #exit
  done  
done

echo "Done."
