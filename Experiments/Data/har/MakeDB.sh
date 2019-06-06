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

MakeCsvFile() {
  dataFile_=$1
  labelFile_=$2
  outCsv_=$3

  awk 'BEGIN { line=0; lastFile="" }
  { gsub("\\r","") }

  FILENAME == ARGV[1] {
    if (lastFile != FILENAME) {
      lastFile=FILENAME
      line=0
    }

    delete values
    split($0, values, " ")

    row=values[1]

    for (i = 2; i <= length(values); ++i)
      row = row "," values[i]

    table[line++] = row
  }

  FILENAME == ARGV[2] {
    if (lastFile != FILENAME) {
      lastFile=FILENAME
      line=0
    }

    table[line] = table[line] "," $0
    ++line
  }
  END {
    for (i = 0; i < length(table); ++i) {
      print table[i]
    }
  }' "${dataFile_}" "${labelFile_}" > "${outCsv_}"
}

searchCmd="bleakMakeDatabase"
makeDBCmd=`GetExeName "${searchCmd}"`

if [ -z "${makeDBCmd}" ]
then
  echo "Error: ${searchCmd} must be in PATH" 1>&2
  exit 1
fi

trainDataFile="UCI HAR Dataset/train/X_train.txt"
trainLabelFile="UCI HAR Dataset/train/y_train.txt"

testDataFile="UCI HAR Dataset/test/X_test.txt"
testLabelFile="UCI HAR Dataset/test/y_test.txt"

MakeCsvFile "${trainDataFile}" "${trainLabelFile}" "train_all.data"
MakeCsvFile "${testDataFile}" "${testLabelFile}" "test.data"

shuf train_all.data > train_shuffled.data

head -n 5514 train_shuffled.data > train.data
tail -n 1838 train_shuffled.data > validation.data

"${makeDBCmd}" -o har_train.lmdb -c mappings.ini train.data
"${makeDBCmd}" -o har_validation.lmdb -c mappings.ini validation.data
"${makeDBCmd}" -o har_test.lmdb -c mappings.ini test.data

