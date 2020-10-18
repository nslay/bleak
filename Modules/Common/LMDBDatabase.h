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

#ifndef BLEAK_LMDBDATABASE_H
#define BLEAK_LMDBDATABASE_H

#include <cstring>
#include <vector>
#include <utility>
#include <memory>
#include "Database.h"
#include "lmdb.h"

namespace bleak {

class LMDBCursor : public Cursor {
public:
  LMDBCursor(MDB_txn *p_mdbTxn, MDB_cursor *p_mdbCursor);
  virtual ~LMDBCursor();

  virtual bool Good() const override {
    return m_stKey.mv_data != nullptr && m_stValue.mv_data != nullptr;
  }

  virtual bool Rewind() override;
  virtual bool Next() override;
  virtual bool Find(const std::string &strKey) override;

  virtual std::string Key() const override {
    if (m_stKey.mv_data == nullptr)
      return std::string();

    return std::string((char *)m_stKey.mv_data, m_stKey.mv_size);
  }

  virtual const uint8_t * Value(size_t &valueSize) const override {
    valueSize = m_stValue.mv_size;
    return (uint8_t *)m_stValue.mv_data;
  }

private:
  MDB_txn *m_p_mdbTxn;
  MDB_cursor *m_p_mdbCursor;

  MDB_val m_stKey, m_stValue;

  LMDBCursor(const LMDBCursor &) = delete;
  LMDBCursor & operator=(const LMDBCursor &) = delete;

  void ClearKeyAndValue() {
    std::memset(&m_stKey,0,sizeof(m_stKey));
    std::memset(&m_stValue,0,sizeof(m_stValue));
  }
};

class LMDBTransaction : public Transaction {
public:
  LMDBTransaction(MDB_env *p_mdbEnv) {
    m_p_mdbEnv = p_mdbEnv;
  }
  virtual ~LMDBTransaction() { }

  virtual bool Put(const std::string &strKey, const uint8_t *p_ui8Buffer, size_t bufferSize) override;
  virtual bool Commit() override;

private:
  enum Status { StatusError, StatusGood, StatusMapFull };

  Status TryCommit();

  MDB_env *m_p_mdbEnv;

  std::vector<std::pair<std::string, std::vector<uint8_t>>> m_vKeyDataPairs;

  LMDBTransaction(const LMDBTransaction &) = delete;
  LMDBTransaction & operator=(const LMDBTransaction &) = delete;
};

class LMDBDatabase : public Database {
public:
  bleakNewDatabase(LMDBDatabase, Database, "LMDB");

  LMDBDatabase();
  virtual ~LMDBDatabase();

  virtual bool Open(const std::string &strPath, Mode eMode) override;
  virtual void Close() override;

  virtual std::unique_ptr<Cursor> NewCursor() override;
  virtual std::unique_ptr<Transaction> NewTransaction() override;

private:
  MDB_env *m_p_mdbEnv;

  LMDBDatabase(const LMDBDatabase &) = delete;
  LMDBDatabase & operator=(const LMDBDatabase &) = delete;
};

} // end namespace bleak

#endif // !BLEAK_LMDBDATABASE_H
