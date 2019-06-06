#!/bin/sh

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

searchCmd="bleakMakeDatabase"
makeDBCmd=`GetExeName "${searchCmd}"`

if [ -z "${makeDBCmd}" ]
then
  echo "Error: ${searchCmd} must be in PATH" 1>&2
  exit 1
fi

# This is the way it should be done
#zcat zip.train.gz | head -n 6291 > usps_train.data
#zcat zip.train.gz | tail -n 1000 > usps_validation.data

# This is how I remember some papers did this experiment... validating on testing!
zcat zip.train.gz | head -n 7291 > usps_train.data
zcat zip.test.gz > usps_validation.data

zcat zip.test.gz > usps_test.data


"${makeDBCmd}" -f -d ' ' -o usps_train.lmdb usps_train.data
"${makeDBCmd}" -d ' ' -o usps_validation.lmdb usps_validation.data
"${makeDBCmd}" -d ' ' -o usps_test.lmdb usps_test.data

