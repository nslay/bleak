#!/bin/bash

if ! which "wget" > /dev/null 2>&1
then
  echo "Error: wget needs to be in PATH." 1>&2
  exit 1
fi

if [ ! -f "eeg_full.tar" ]
then
  wget -O "eeg_full.tar" "http://kdd.ics.uci.edu/databases/eeg/eeg_full.tar"
fi

rootDir="data"
eegTarFile="${PWD}/eeg_full.tar"

mkdir -pv "${rootDir}"
pushd "${rootDir}"

tar -xvf "${eegTarFile}"

for subjectTar in *.tar.gz
do
  subject=`basename "${subjectTar}" .tar.gz`
  tar -xvzf "${subjectTar}"

  pushd "${subject}"

  for trialGz in *.gz
  do
    gunzip "${trialGz}"
  done

  popd
done

popd

