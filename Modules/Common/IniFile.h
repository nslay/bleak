/*-
 * Copyright (c) 2012 Nathan Lay (nslay@users.sourceforge.net)
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

#ifndef INIFILE_H
#define INIFILE_H

#include <cstring>
#include <sstream>
#include <string>
#include <map>

// Implement own INI parser since iniparser considers '#' a comment!

class IniFile {
public:
	struct StringLessThan {
		bool operator()(const std::string &strString1, const std::string &strString2) const {
#ifdef _WIN32
      return _stricmp(strString1.c_str(), strString2.c_str()) < 0;
#else // !_WIN32
      return strcasecmp(strString1.c_str(), strString2.c_str()) < 0;
#endif // !_WIN32
		}
	};

	class Section {
	public:
		typedef std::map<std::string, std::string, StringLessThan> MapType;
		typedef MapType::iterator KeyIterator;
		typedef MapType::const_iterator ConstKeyIterator;

		Section() { }

		Section(const std::string &strName)
		: m_strName(strName) { }

		KeyIterator KeyBegin() {
			return m_mValueMap.begin();
		}

		ConstKeyIterator KeyBegin() const {
			return m_mValueMap.begin();
		}

		KeyIterator KeyEnd() {
			return m_mValueMap.end();
		}

		ConstKeyIterator KeyEnd() const {
			return m_mValueMap.end();
		}

		const std::string & GetName() const {
			return m_strName;
		}

		bool Empty() const {
			return m_mValueMap.empty();
		}

		template<typename T>
		T GetValue(const std::string &strKey, const T &defaultValue) const;

		template<typename T>
		void SetValue(const std::string &strKey, const T &value);

		void Clear() {
			m_mValueMap.clear();
		}

	private:
		std::string m_strName;
		std::map<std::string, std::string, StringLessThan> m_mValueMap;
	};

	typedef std::map<std::string, Section, StringLessThan> MapType;
	typedef MapType::iterator SectionIterator;
	typedef MapType::const_iterator ConstSectionIterator;

	IniFile() { }

	IniFile(const std::string &strFilename) {
		Load(strFilename);
	}

	SectionIterator SectionBegin() {
		return m_mSectionMap.begin();
	}

	ConstSectionIterator SectionBegin() const {
		return m_mSectionMap.begin();
	}

	SectionIterator SectionEnd() {
		return m_mSectionMap.end();
	}

	ConstSectionIterator SectionEnd() const {
		return m_mSectionMap.end();
	}

	bool Load(const std::string &strFilename);

	bool HasSection(const std::string &strSection) const {
		return m_mSectionMap.find(strSection) != SectionEnd();
	}

	Section & GetSection(const std::string &strSection) {
		// Will do insert automatically
		return m_mSectionMap[strSection];
	}

	void Clear() {
		m_mSectionMap.clear();
	}

private:
	MapType m_mSectionMap;

	static void Trim(std::string &strString);
};

template<typename T>
T IniFile::Section::GetValue(const std::string &strKey, const T &defaultValue) const {
	ConstKeyIterator itr = m_mValueMap.find(strKey);

	if (itr != KeyEnd()) {
		std::stringstream valueStream;
		valueStream.str(itr->second);

		T value;
		return (valueStream >> value) ? value : defaultValue;
	}

	return defaultValue;
}

template<>
inline std::string IniFile::Section::GetValue<std::string>(const std::string &strKey, const std::string &strDefault) const {
	ConstKeyIterator itr = m_mValueMap.find(strKey);

	return itr != KeyEnd() ? itr->second : strDefault;
}

template<typename T>
void IniFile::Section::SetValue(const std::string &strKey, const T &value) {
	std::stringstream valueStream;

	valueStream << value;

	m_mValueMap[strKey] = valueStream.str();
}

template<>
inline void IniFile::Section::SetValue<std::string>(const std::string &strKey, const std::string &strValue) {
	m_mValueMap[strKey] = strValue;
}

#endif // !INIFILE_H

