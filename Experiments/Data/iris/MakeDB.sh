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

# 10 folds total
numShuffles=5

shuffleFile="iris_shuffle.data"
fold1fold2File="iris_fold1_fold2.data"

for exp in `seq 1 ${numShuffles}`
do
  # Awk is for removing empty lines
  shuf "iris.data" | awk 'NF > 0' > "${shuffleFile}"

  # Use one for validation
  fold1File="iris_${exp}_fold1.data"
  fold2File="iris_${exp}_fold2.data"
  fold3File="iris_${exp}_fold3.data"

  head -n 100 "${shuffleFile}" > "${fold1fold2File}"
  head -n 50 "${fold1fold2File}" > "${fold1File}"
  tail -n 50 "${fold1fold2File}" > "${fold2File}"

  tail -n 50 "${shuffleFile}" > "${fold3File}"

  fold1DBFile="iris_${exp}_fold1.lmdb"
  fold2DBFile="iris_${exp}_fold2.lmdb"
  fold3DBFile="iris_${exp}_fold3.lmdb"

  "${makeDBCmd}" -o "${fold1DBFile}" -c mappings.ini "${fold1File}"
  "${makeDBCmd}" -o "${fold2DBFile}" -c mappings.ini "${fold2File}"
  "${makeDBCmd}" -o "${fold3DBFile}" -c mappings.ini "${fold2File}"
done

