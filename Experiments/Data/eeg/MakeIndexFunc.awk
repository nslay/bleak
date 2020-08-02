#!/usr/bin/awk -f

BEGIN {
  if (ARGC < 2) {
    print "Usage: MakeIndexFunc.awk eegCaseFile" > "/dev/stderr"
  }
  channelIndex=0
}

/^#/ { next }

NF == 4 { 
  channelName=$2
  if (!(channelName in channelIndexByName))
    channelIndexByName[channelName] = channelIndex++
}

END {
  for (channelName in channelIndexByName) {
    channelIndex = channelIndexByName[channelName]
    channelNameByIndex[channelIndex] = channelName
  }

  print "int GetChannelIndex(const char *p_cChannelName) {"
  printf "  static const char * const a_cChannelNames[] = {"
  
  for (channelIndex = 0; channelIndex in channelNameByIndex; ++channelIndex) {
    if ((channelIndex % 4) == 0)
      printf "\n    "
    
    channelName = channelNameByIndex[channelIndex]
    printf "\"" channelName "\", "
  }
  
  print "\n    nullptr"
  
  print "  };"
  
  print "\n  if (p_cChannelName == nullptr)"
  print "    return -1;"
  print ""

  print "  for (int iChannelIndex = 0; a_cChannelNames[iChannelIndex] != nullptr; ++iChannelIndex) {"
  print "    if (strcmp(p_cChannelName, a_cChannelNames[iChannelIndex]) == 0)"
  print "      return iChannelIndex;"
  print "  }"
  
  print "\n  return -1;"
  print "}"
}
