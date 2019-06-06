#!/bin/sh

if ! which "wget" > /dev/null 2>&1
then
  echo "Error: wget needs to be in PATH." 1>&2
  exit 1
fi

baseUrl="https://archive.ics.uci.edu/ml/machine-learning-databases"
dataSet="poker"
srcUrl="${baseUrl}/${dataSet}"

for file in "${dataSet}-hand-training-true.data" "${dataSet}-hand-testing.data" "${dataSet}-hand.names"
do
  wget -O "${file}" "${srcUrl}/${file}"
done

