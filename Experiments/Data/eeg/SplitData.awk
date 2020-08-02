#!/usr/bin/awk -f

function min(x_, y_) {
  return x_ < y_ ? x_ : y_
}

function max(x_,y_) {
  return x_ < y_ ? y_ : x_
}

function ExtractCaseID(path_) {
  # Include path to case as part of the case ID
  if (match(path_, "^.+/co(2|3)(a|c)[0-9]+") == 0)
    return ""
    
  return substr(path_, RSTART, RLENGTH)
}

function ExtractCaseExtension(path_) {
  if (match(path_, "\\.rd\\.[0-9]+.+$") == 0)
    return ""
    
  return substr(path_, RSTART, RLENGTH)
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

function MakeConditionLists(conditionTable_) {
  srand(seed)
  
  numClasses_=0
  
  for (label_ in conditionTable_)
    ++numClasses_
    
  print numClasses_
    
  for (label_ = 0; label_ < numClasses_; ++label_) {
    for (caseID_ in conditionTable_[label_]) {
      delete extArr_
      
      caseCount_=0
      for (caseExt_ in conditionTable_[label_][caseID_])
        extArr_[caseCount_++] = caseExt_
        
      if (caseCount_ <= 0)
        continue; # Uhh?
        
      RandomShuffle(extArr_)
      
      trainBegin_=0
      trainEnd_=min(trainBegin_ + max(1, int(trainRatio * caseCount_)), caseCount_-1)
      valBegin_=trainEnd_
      valEnd_=min(valBegin_ + max(1, int(validationRatio * caseCount_)), caseCount_-1)

      
      trainSize_=trainEnd_ - trainBegin_
      valSize_=valEnd_ - valBegin_
      
      testBegin_=valEnd_
      testEnd_=testBegin_ + caseCount_ - (trainSize_ + valSize_)
      
      for (i_ = trainBegin_; i_ < trainEnd_; ++i_) {
        print pathPrefix caseID_ extArr_[i_] >> conditionTrainList
        print label_ >> conditionTrainLabelsCsv
      }
      
      for (i_ = valBegin_; i_ < valEnd_; ++i_) {
        print pathPrefix caseID_ extArr_[i_] >> conditionValidationList
        print label_ >> conditionValidationLabelsCsv
      }
      
      for (i_ = testBegin_; i_ < testEnd_; ++i_) {
        print pathPrefix caseID_ extArr_[i_] >> conditionTestList
        print label_ >> conditionTestLabelsCsv
      }
    }
  }
}

function MakeAlcoholicLists(alcoholicTable_) {
  srand(seed)
  
  numClasses_=0
  
  for (label_ in alcoholicTable_)
    ++numClasses_
    
  for (label_ = 0; label_ < numClasses_; ++label_) {
    for (caseID_ in alcoholicTable_[label_]) {
      delete extArr_
      
      caseCount_=0
      for (caseExt_ in alcoholicTable_[label_][caseID_])
        extArr_[caseCount_++] = caseExt_
        
      if (caseCount_ <= 0)
        continue; # Uhh?
        
      RandomShuffle(extArr_)
      
      trainBegin_=0
      trainEnd_=min(trainBegin_ + max(1, int(trainRatio * caseCount_)), caseCount_-1)
      valBegin_=trainEnd_
      valEnd_=min(valBegin_ + max(1, int(validationRatio * caseCount_)), caseCount_-1)

      
      trainSize_=trainEnd_ - trainBegin_
      valSize_=valEnd_ - valBegin_
      
      testBegin_=valEnd_
      testEnd_=testBegin_ + caseCount_ - (trainSize_ + valSize_)
      
      for (i_ = trainBegin_; i_ < trainEnd_; ++i_) {
        print pathPrefix caseID_ extArr_[i_] >> alcoholicTrainList
        print label_ >> alcoholicTrainLabelsCsv
      }
      
      for (i_ = valBegin_; i_ < valEnd_; ++i_) {
        print pathPrefix caseID_ extArr_[i_] >> alcoholicValidationList
        print label_ >> alcoholicValidationLabelsCsv
      }
      
      for (i_ = testBegin_; i_ < testEnd_; ++i_) {
        print pathPrefix caseID_ extArr_[i_] >> alcoholicTestList
        print label_ >> alcoholicTestLabelsCsv
      }
    }
  }
}

BEGIN {
  seed=6112
  trainRatio=0.5
  validationRatio=0.25
  testRatio=1.0 - trainRatio - validationRatio
  pathPrefix="data/"
  
  conditionTrainList="conditionTrainList.txt"
  conditionValidationList="conditionValidationList.txt"
  conditionTestList="conditionTestList.txt"
  conditionTrainLabelsCsv="conditionTrainLabels.csv"
  conditionValidationLabelsCsv="conditionValidationLabels.csv"
  conditionTestLabelsCsv="conditionTestLabels.csv"
  
  alcoholicTrainList="alcoholicTrainList.txt"
  alcoholicValidationList="alcoholicValidationList.txt"
  alcoholicTestList="alcoholicTestList.txt"
  alcoholicTrainLabelsCsv="alcoholicTrainLabels.csv"
  alcoholicValidationLabelsCsv="alcoholicValidationLabels.csv"
  alcoholicTestLabelsCsv="alcoholicTestLabels.csv"
  
  printf "" > conditionTrainList
  printf "" > conditionValidationList
  printf "" > conditionTestList
  
  print "label" > conditionTrainLabelsCsv
  print "label" > conditionValidationLabelsCsv
  print "label" > conditionTestLabelsCsv
 
  printf "" > alcoholicTrainList
  printf "" > alcoholicValidationList
  printf "" > alcoholicTestList
  
  print "label" > alcoholicTrainLabelsCsv
  print "label" > alcoholicValidationLabelsCsv
  print "label" > alcoholicTestLabelsCsv
  
  FS=","
}

# Skip CSV header
FNR == 1 { next }

NF == 3 {
  caseID=ExtractCaseID($1)
  caseExt=ExtractCaseExtension($1)
  condition=$2
  alcoholic=$3
  
  if (length(caseID) == 0 || length(caseExt) == 0) {
    print "Ignoring " $1 " with invalid caseID/Extension"
    next
  }
  
  if (condition < 0 || condition > 2 || !(condition ~ /[0-9]/)) {
    print "Ignoring " $1 " with condition label " condition
    next
  }
  
  if (alcoholic < 0 || alcoholic > 1 || !(alcoholic ~ /[0-9]/)) {
    print "Ignoring " $1 " with alcoholic label " alcoholic
    next
  }
  
  conditionTable[condition][caseID][caseExt] = 1
  alcoholicTable[alcoholic][caseID][caseExt] = 1
  
  #print caseID " " caseExt
}

END {
  MakeConditionLists(conditionTable)
  MakeAlcoholicLists(alcoholicTable)
}

