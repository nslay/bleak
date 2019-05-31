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

#include "LMDBDatabase.h"

namespace bleak {

//////////////////////////// Cursor ////////////////////////////

LMDBCursor::LMDBCursor(MDB_txn *p_mdbTxn, MDB_cursor *p_mdbCursor) {
  ClearKeyAndValue();

  m_p_mdbTxn = p_mdbTxn;
  m_p_mdbCursor = p_mdbCursor;

  Rewind();
}

LMDBCursor::~LMDBCursor() {
  mdb_cursor_close(m_p_mdbCursor);
  mdb_txn_abort(m_p_mdbTxn);
}

bool LMDBCursor::Rewind() {
  if (mdb_cursor_get(m_p_mdbCursor, &m_stKey, &m_stValue, MDB_FIRST) != 0) {
    ClearKeyAndValue();
    return false;
  }

  return true;
}

bool LMDBCursor::Next() {
  if (mdb_cursor_get(m_p_mdbCursor, &m_stKey, &m_stValue, MDB_NEXT) != 0) {
    ClearKeyAndValue();
    return false;
  }

  return true;
}

bool LMDBCursor::Find(const std::string &strKey) {
  m_stKey.mv_data = (void *)strKey.data();
  m_stKey.mv_size = strKey.size();

  if (mdb_cursor_get(m_p_mdbCursor, &m_stKey, &m_stValue, MDB_SET_KEY) != 0) {
    ClearKeyAndValue();
    return false;
  }

  return true;
}

//////////////////////////// Transaction ////////////////////////////

bool LMDBTransaction::Put(const std::string &strKey, const uint8_t *p_ui8Buffer, size_t bufferSize) {
  m_vKeyDataPairs.emplace_back(strKey, std::vector<uint8_t>(p_ui8Buffer, p_ui8Buffer + bufferSize));
  return true;
}

bool LMDBTransaction::Commit() {
  if (m_vKeyDataPairs.empty())
    return true;

  MDB_txn *p_mdbTxn = nullptr;
  MDB_dbi mdbDbi;

  if (mdb_txn_begin(m_p_mdbEnv, nullptr, 0, &p_mdbTxn) != 0)
    return false;

  if (mdb_dbi_open(p_mdbTxn, nullptr, MDB_CREATE, &mdbDbi) != 0) {
    mdb_txn_abort(p_mdbTxn);
    return false;
  }

  MDB_val stKey, stValue;

  for (const auto &stKeyValuePair : m_vKeyDataPairs) {
    stKey.mv_data = (void *)stKeyValuePair.first.data();
    stKey.mv_size = stKeyValuePair.first.size();

    stValue.mv_data = (void *)stKeyValuePair.second.data();
    stValue.mv_size = stKeyValuePair.second.size();

    if (mdb_put(p_mdbTxn, mdbDbi, &stKey, &stValue, 0) != 0) {
      // XXX: Handle this.
    }
  }

  m_vKeyDataPairs.clear();

  mdb_txn_commit(p_mdbTxn);
  mdb_dbi_close(m_p_mdbEnv, mdbDbi);

  return true;
}

//////////////////////////// Database ////////////////////////////

LMDBDatabase::LMDBDatabase() {
  m_p_mdbEnv = nullptr;
}

LMDBDatabase::~LMDBDatabase() {
  mdb_env_close(m_p_mdbEnv);
}

bool LMDBDatabase::Open(const std::string &strPath, Mode eMode) {
  Close();

  int iRet = 0;
  if (mdb_env_create(&m_p_mdbEnv) != 0) {
    m_p_mdbEnv = nullptr;
    return false;
  }

  mdb_env_set_mapsize(m_p_mdbEnv, (mdb_size_t)1 << 31);

  unsigned int uiFlags = MDB_NOSUBDIR;

  if (eMode == READ)
    uiFlags |= MDB_RDONLY;

  if (mdb_env_open(m_p_mdbEnv, strPath.c_str(), uiFlags, 0664) != 0) {
    Close();
    return false;
  }

  return true;
}

void LMDBDatabase::Close() {
  mdb_env_close(m_p_mdbEnv);
  m_p_mdbEnv = nullptr;
}

std::unique_ptr<Cursor> LMDBDatabase::NewCursor() {
  MDB_txn *p_mdbTxn = nullptr;

  if (mdb_txn_begin(m_p_mdbEnv, nullptr, MDB_RDONLY, &p_mdbTxn) != 0)
    return std::unique_ptr<Cursor>();

  MDB_dbi mdbDbi;
  if (mdb_dbi_open(p_mdbTxn, nullptr, 0, &mdbDbi) != 0) {
    mdb_txn_abort(p_mdbTxn);
    return std::unique_ptr<Cursor>();
  }

  MDB_cursor *p_mdbCursor = nullptr;

  if (mdb_cursor_open(p_mdbTxn, mdbDbi, &p_mdbCursor) != 0) {
    mdb_txn_abort(p_mdbTxn);
    return std::unique_ptr<Cursor>();
  }

  return std::make_unique<LMDBCursor>(p_mdbTxn, p_mdbCursor);
}

std::unique_ptr<Transaction> LMDBDatabase::NewTransaction() {
  return std::make_unique<LMDBTransaction>(m_p_mdbEnv);
}

} // end namespace bleak

