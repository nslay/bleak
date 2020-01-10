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

searchCmd="bleakMakeDatabaseCIFAR"
makeDBCmd=`GetExeName "${searchCmd}"`

if [ -z "${makeDBCmd}" ]
then
  echo "Error: ${searchCmd} must be in PATH" 1>&2
  exit 1
fi

for file in *.tar.gz
do
  tar -xvzf "${file}"
done

"${makeDBCmd}" -f -o cifar10_train.lmdb -v cifar10_val.lmdb -V 0.2 -k 10 cifar-10-batches-bin/data_batch_1.bin cifar-10-batches-bin/data_batch_2.bin cifar-10-batches-bin/data_batch_3.bin cifar-10-batches-bin/data_batch_4.bin cifar-10-batches-bin/data_batch_5.bin
"${makeDBCmd}" -f -o cifar10_test.lmdb -k 10 cifar-10-batches-bin/test_batch.bin

"${makeDBCmd}" -f -o cifar100_train.lmdb -v cifar100_val.lmdb -V 0.2 -k 100 cifar-100-binary/train.bin
"${makeDBCmd}" -f -o cifar100_test.lmdb -k 100 cifar-100-binary/test.bin
