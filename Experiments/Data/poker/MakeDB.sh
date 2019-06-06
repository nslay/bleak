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

head -n 18757 poker-hand-training-true.data > poker_train.data
tail -n 6253 poker-hand-training-true.data > poker_validation.data

"${makeDBCmd}" -o poker_train.lmdb poker_train.data
"${makeDBCmd}" -o poker_validation.lmdb poker_validation.data
"${makeDBCmd}" -o poker_test.lmdb poker-hand-testing.data

