#!/bin/sh

if ! which "wget" > /dev/null 2>&1
then
  echo "Error: wget needs to be in PATH." 1>&2
  exit 1
fi

baseUrl="http://yann.lecun.com/exdb/mnist"

for file in "train-images-idx3-ubyte.gz" "train-labels-idx1-ubyte.gz" "t10k-images-idx3-ubyte.gz" "t10k-labels-idx1-ubyte.gz"
do
  wget -O "${file}" "${baseUrl}/${file}"
done

