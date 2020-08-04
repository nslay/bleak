#!/usr/bin/awk -f

function min(x_, y_) {
  return x_ < y_ ? x_ : y_
}

function max(x_,y_) {
  return x_ < y_ ? y_ : x_
}

function RandomShuffle(arr_) {
  count_=0
  
  for (lame_ in arr_)
    ++count_
    
  for (q_ = 0; q_ < count_; ++q_) {
    p_ = int(q_ + rand()*(count_ - q_))
    
    p_ = min(p_, count_ - 1)
    
    lame_ = arr_[q_]
    arr_[q_] = arr_[p_]
    arr_[p_] = lame_
  }
}

BEGIN {
  seed=6112;
  
  err=0
  if (ARGC != 3) {
    print "Usage: ShuffleLists.awk list1.txt list2.csv" > "/dev/stderr"
    err=1
    exit 1
  }
  
  list1Count=0
  list2Count=0
}

FILENAME == ARGV[1] {
  list1[list1Count++] = $0
}

FILENAME == ARGV[2] && FNR > 1 {
  list2[list2Count++] = $0
}

END {
  if (list1Count != list2Count) {
    print "Error: Different list sizes: " list1Count " != " list2Count > "/dev/stderr"
    err=1
    exit 1
  }

  if (err == 0) {
    srand(seed)
    RandomShuffle(list1)
    
    file1=ARGV[1]
    printf "" > file1
    
    for (i = 0; i < list1Count; ++i)
      print list1[i] >> file1
    
    srand(seed)
    RandomShuffle(list2)
    
    file2=ARGV[2]
    print "label" > file2
    
    for (i = 0; i < list2Count; ++i)
      print list2[i] >> file2 
  }
}
