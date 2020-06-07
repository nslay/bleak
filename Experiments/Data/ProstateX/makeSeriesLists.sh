#!/bin/sh

if [ $# -ne 2 ]
then
  echo "Usage: $0 listFile outputFolder" 1>&2
  exit 1
fi

listFile=$1
outDir=$2

mkdir -pv "${outDir}"

cp -v "${listFile}" "${outDir}/maskList.txt"

awk 'BEGIN { FS="/"; OFS="/" } { $1 = "T2W"; print }' "${listFile}" > "${outDir}/t2wList.txt"
awk 'BEGIN { FS="/"; OFS="/" } { $1 = "ADC"; print }' "${listFile}" > "${outDir}/adcList.txt"
awk 'BEGIN { FS="/"; OFS="/" } { $1 = "B1500"; print }' "${listFile}" > "${outDir}/b1500List.txt"
