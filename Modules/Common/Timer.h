/*-
 * Copyright (c) 2018 Nathan Lay (enslay@gmail.com)
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR(S) ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE AUTHOR(S) BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#ifndef BLEAK_TIMER_H
#define BLEAK_TIMER_H

#if defined(_WIN32)

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif // !WIN32_LEAN_AND_MEAN

#include <Windows.h>
#elif defined(__unix__)
#include <sys/time.h>
#else
#error "Unsupported operating system."
#endif

#include <cstring>
#include <string>
#include <iostream>

namespace bleak {

class Timer {
public:
  Timer() {
    m_p_outStream = nullptr; // No automatic printing
    Start();
  }

  Timer(const std::string &strOutputPrefix, std::ostream &outStream = std::cout) {
    m_strOutputPrefix = strOutputPrefix;
    m_p_outStream = &outStream;
    Start();
  }

  Timer(std::ostream &outStream) {
    m_strOutputPrefix = "Elapsed time: ";
    m_p_outStream = &outStream;
    Start();
  }

  ~Timer() {
    if (!IsStopped()) {
      Stop();
      Print();
    }
  }

  void Start() {
    m_dElapsedTimeInSeconds = -1.0;

#ifdef _WIN32
    if (UseQpc()) {
      std::memset(&m_stQpcStart, 0, sizeof(m_stQpcStart));
      QueryPerformanceCounter(&m_stQpcStart);
    }
    else {
      m_ullStart = GetTickCount64();
    }
#endif // _WIN32

#ifdef __unix__
    std::memset(&m_stStart, 0, sizeof(m_stStart));
    gettimeofday(&m_stStart, nullptr);
#endif // __unix__
  }

  void Stop() {
    if (!IsStopped()) {
      const double dElapsed = ElapsedTimeInSeconds();
      m_dElapsedTimeInSeconds = dElapsed;
    }
  }

  bool IsStopped() const {
    return m_dElapsedTimeInSeconds >= 0.0;
  }  

  double ElapsedTimeInSeconds() const {
    if (IsStopped())
      return m_dElapsedTimeInSeconds;

#ifdef _WIN32
    if (UseQpc()) {
      LARGE_INTEGER stQpcStop;
      std::memset(&stQpcStop, 0, sizeof(stQpcStop));

      QueryPerformanceCounter(&stQpcStop);

      return (stQpcStop.QuadPart - m_stQpcStart.QuadPart)/(double)m_s_stQpcFrequency.QuadPart;
    }
    else {
      ULONGLONG ullStop = GetTickCount64();
      return (ullStop - m_ullStart)/1000.0;
    }
#endif // _WIN32

#ifdef __unix__
    struct timeval stStop;
    std::memset(&stStop, 0, sizeof(stStop));

    // Same as converting both to seconds and subtracting the converted results
    return (stStop.tv_sec - m_stStart.tv_sec) + (stStop.tv_usec - m_stStart.tv_usec)/1000000.0;
#endif // __unix__
  }

  void Print(std::ostream &os) const {
    os << m_strOutputPrefix << ElapsedTimeInSeconds() << " seconds";
  }

  void Print() const {
    if (m_p_outStream != nullptr)
      Print(*m_p_outStream);
  }

private:
  std::string m_strOutputPrefix;
  std::ostream *m_p_outStream;
  double m_dElapsedTimeInSeconds;

#ifdef _WIN32
  static bool UseQpc() {
    static const bool s_bUseQpc = (QueryPerformanceFrequency(&m_s_stQpcFrequency) != FALSE);
    return s_bUseQpc;
  }

  static LARGE_INTEGER m_s_stQpcFrequency;
  LARGE_INTEGER m_stQpcStart;
  ULONGLONG m_ullStart;
#endif // _WIN32

#ifdef __unix__
  struct timeval m_stStart;
#endif // __unix__

};

inline std::ostream & operator<<(std::ostream &os, const Timer &clTimer) {
  clTimer.Print(os);
  return os;
}

} // end namespace bleak

#endif // !BLEAK_TIMER_H
