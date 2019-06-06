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

#head -n 3133 abalone.data > abalone_train.data

head -n 3133 abalone.data > abalone_all_train.data
tail -n 1044 abalone.data > abalone_test.data
head -n 2089 abalone_all_train.data > abalone_train.data
tail -n 1044 abalone_all_train.data > abalone_validation.data

"${makeDBCmd}" -o abalone_train.lmdb -c mappings.ini abalone_train.data
"${makeDBCmd}" -o abalone_validation.lmdb -c mappings.ini abalone_validation.data
"${makeDBCmd}" -o abalone_test.lmdb -c mappings.ini abalone_test.data

