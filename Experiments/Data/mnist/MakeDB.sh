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

searchCmd="bleakMakeDatabaseMNIST"
makeDBCmd=`GetExeName "${searchCmd}"`

if [ -z "${makeDBCmd}" ]
then
  echo "Error: ${searchCmd} must be in PATH" 1>&2
  exit 1
fi

for file in *.gz
do
  base=`basename "${file}" .gz`

  gunzip -c "${file}" > "${base}.raw"
done

# TODO: Validation?
"${makeDBCmd}" -f -o mnist_train.lmdb "train-images-idx3-ubyte.raw" "train-labels-idx1-ubyte.raw"
"${makeDBCmd}" -f -o mnist_test.lmdb "t10k-images-idx3-ubyte.raw" "t10k-labels-idx1-ubyte.raw"

