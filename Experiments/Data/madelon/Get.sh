#!/bin/sh

if ! which "wget" > /dev/null 2>&1
then
  echo "Error: wget needs to be in PATH." 1>&2
  exit 1
fi

baseUrl="https://archive.ics.uci.edu/ml/machine-learning-databases"
dataSet="madelon"
srcUrl="${baseUrl}/${dataSet}"

for file in "MADELON/${dataSet}_train.data" "MADELON/${dataSet}_train.labels" "MADELON/${dataSet}_valid.data" "Dataset.pdf" "${dataSet}_valid.labels"
do
  baseFile=`basename "${file}"`

  wget -O "${baseFile}" "${srcUrl}/${file}"
done

