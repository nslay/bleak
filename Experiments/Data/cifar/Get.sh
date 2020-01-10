#!/bin/sh

if ! which "wget" > /dev/null 2>&1
then
  echo "Error: wget needs to be in PATH." 1>&2
  exit 1
fi

baseUrl="https://www.cs.toronto.edu/~kriz"

for file in "cifar-10-binary.tar.gz" "cifar-100-binary.tar.gz"
do
  wget -O "${file}" "${baseUrl}/${file}"
done

