/*-
 * Copyright (c) 2017 Nathan Lay (enslay@gmail.com)
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

#ifndef BLEAK_DATABASE_H
#define BLEAK_DATABASE_H

#include <cstdint>
#include <string>
#include <vector>
#include <memory>

// Interface inspired by Caffe

namespace bleak {

#define bleakNewAbstractDatabase(thisClass, superClass) \
  typedef superClass SuperType; \
  typedef thisClass SelfType

#define bleakNewDatabase(thisClass, superClass, typeName) \
  static constexpr const char * GetTypeName() { return typeName ; } \
  static std::shared_ptr< thisClass > New() { \
    std::shared_ptr< thisClass> p_clDatabase(new ( thisClass )()); \
    return p_clDatabase; \
  } \
  bleakNewAbstractDatabase(thisClass , superClass)

class Cursor {
public:
  Cursor() = default;
  virtual ~Cursor() = default;

  virtual bool Good() const = 0;
  virtual bool Rewind() = 0;
  virtual bool Next() = 0;
  virtual bool Find(const std::string &strKey) = 0;

  virtual std::string Key() const = 0;
  virtual const uint8_t * Value(size_t &valueSize) const = 0;

private:
  Cursor(const Cursor &) = delete;
  Cursor & operator=(const Cursor &) = delete;
};

class Transaction {
public:
  Transaction() = default;
  virtual ~Transaction() = default;

  virtual bool Put(const std::string &strKey, const uint8_t *p_ui8Buffer, size_t bufferSize) = 0;
  virtual bool Commit() = 0;

private:
  Transaction(const Transaction &) = delete;
  Transaction & operator=(const Transaction &) = delete;
};

class Database {
public:
  enum Mode { READ, WRITE };

  Database() = default;
  virtual ~Database() = default;

  virtual bool Open(const std::string &strPath, Mode eMode) = 0;
  virtual void Close() = 0;

  virtual std::unique_ptr<Cursor> NewCursor() = 0;
  virtual std::unique_ptr<Transaction> NewTransaction() = 0;

private:
  Database(const Database &) = delete;
  Database & operator=(const Database &) = delete;
};

} // end namespace bleak

#endif // !BLEAK_DATABASE_H
