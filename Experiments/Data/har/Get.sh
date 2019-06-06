#!/bin/sh

if ! which "wget" > /dev/null 2>&1
then
  echo "Error: wget needs to be in PATH." 1>&2
  exit 1
fi

if ! which "unzip" > /dev/null 2>&1
then
  echo "Error: unzip needs to be in PATH." 1>&2
  exit 1
fi

baseUrl="https://archive.ics.uci.edu/ml/machine-learning-databases"
srcUrl="${baseUrl}/00240"

for file in "UCI HAR Dataset.names" "UCI HAR Dataset.zip"
do
  wget -O "${file}" "${srcUrl}/${file}"
done

unzip "UCI HAR Dataset.zip"

