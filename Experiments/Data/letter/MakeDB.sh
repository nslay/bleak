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

# Turn off validation since other works don't seem to do it...
head -n 16000 letter-recognition.data > letter-recognition-training-all.data
tail -n 4000 letter-recognition.data > letter-recognition-testing.data
tail -n 4000 letter-recognition.data > letter-recognition-validation.data

cp letter-recognition-training-all.data letter-recognition-training.data

#shuf letter-recognition-training-all.data > letter-recognition-training-all-shuf.data

#head -n 14000 letter-recognition-training-all-shuf.data > letter-recognition-training.data
#tail -n 2000 letter-recognition-training-all-shuf.data > letter-recognition-validation.data
#head -n 14000 letter-recognition-training-all.data > letter-recognition-training.data
#tail -n 2000 letter-recognition-training-all.data > letter-recognition-validation.data

"${makeDBCmd}" -f -c mappings.ini -o letter_train.lmdb letter-recognition-training.data
"${makeDBCmd}" -c mappings.ini -o letter_validation.lmdb letter-recognition-validation.data
"${makeDBCmd}" -c mappings.ini -o letter_test.lmdb letter-recognition-testing.data

