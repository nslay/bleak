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

#include <iostream>
#include <cctype>
#include <fstream>
#include "IniFile.h"

bool IniFile::Load(const std::string &strFilename) {
	std::ifstream iniStream(strFilename.c_str());

	if (!iniStream)
		return false;

	Section clSection;

	std::string strLine;
	while (std::getline(iniStream, strLine)) {
		// Remove carriage return
		size_t p = strLine.find('\r');

		if (p != std::string::npos)
			strLine.resize(p);

		p = strLine.find(';');

		if (p != std::string::npos) 
			strLine.resize(p);

		Trim(strLine);

		if (strLine.empty())
			continue;

		//std::cout << strLine << std::endl;

		if (strLine[0] == '[') {
			// Assign previously loaded section
			if (!clSection.GetName().empty())
				GetSection(clSection.GetName()) = clSection;

			// Section name
			p = strLine.find(']');

			if (p == std::string::npos) {
				Clear();
				return false;
			}

			std::string strName = strLine.substr(1,p-1);

			// Empty []
			if (strName.empty()) {
				Clear();
				return false;
			}

			//std::cout << "Section name = '" << strName << '\'' << std::endl;

			clSection = Section(strName);
		}
		else {
			// No section read before
			if (clSection.GetName().empty()) {
				Clear();
				return false;
			}

			// Key value pair
			p = strLine.find('=');

			// No =
			if (p == std::string::npos) {
				Clear();
				return false;
			}

			std::string strKey, strValue;

			strKey = strLine.substr(0,p);

			if (++p < strLine.size())
				strValue = strLine.substr(p);

			Trim(strKey);
			Trim(strValue);

			if (strKey.empty()) {
				Clear();
				return false;
			}

			//std::cout << '\'' << strKey << "' = '" << strValue << '\'' << std::endl;

			clSection.SetValue(strKey, strValue);
		}

	}

	if (!clSection.GetName().empty())
		GetSection(clSection.GetName()) = clSection;

	return true;
}

void IniFile::Trim(std::string &strString) {
	size_t p;

	p = strString.find_first_not_of(" \t\r\n");

	if (p != std::string::npos)
		strString.erase(0, p);
	
	p = strString.find_last_not_of(" \t\r\n");

	if (p != std::string::npos)
		strString.resize(p+1);
}

