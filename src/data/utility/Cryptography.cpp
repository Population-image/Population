#include <iostream>
#include <cmath>
#include <fstream>
#include <cstring>

#include "data/utility/Cryptography.h"
#include "PopulationConfig.h"

namespace pop {

// Encrypt the text text using the key key of size key_len. Returns an array of char of size len
char* Cryptography::xor_encryptText(const std::string text, size_t* len,
		const char* key, const size_t key_len) {
	*len = text.length() + 1;
	char* cstr = new char[*len]; // final '\0' not taken into account by length()
	std::strcpy(cstr, text.c_str());

	xor_encryptOrDecryptBinary(cstr, *len, key, key_len);
	return cstr;
}

// Decrypt the binary data d of size len using the key key of size key_len and return the decrypted data as a string
std::string Cryptography::xor_decryptText(const char* d, const size_t len,
		const char* key, const size_t key_len) {
	char* copy = new char[len];
	std::memcpy(copy, d, len);

	xor_encryptOrDecryptBinary(copy, len, key, key_len);

	std::string s = std::string(copy);
	delete[] copy;
	return s;
}

// Encrypt or decrypt the binary data d of size len in place using the key key of size key_len
void Cryptography::xor_encryptOrDecryptBinary(char* d, const size_t len,
		const char* key, const size_t key_len) {
	for (unsigned int i = 0; i < len; i++) {
		d[i] = d[i] ^ key[i % key_len];
	}
}

// Open the encrypted file filename, decrypt it, and return the result of the decryption as an array of char of size *len
char* Cryptography::xor_decryptFile(const std::string filename, size_t *len, const char* key,
		const size_t key_len) {
	std::ifstream infile(filename.c_str(), std::ios::binary);
	infile.seekg(0, infile.end);
    *len = infile.tellg();

    char* c = new char[*len];
	infile.seekg(0, std::ios::beg);
    infile.read(c, *len);
	infile.close();

    xor_encryptOrDecryptBinary(c, *len, key, key_len);

	return c;
}

// Encrypt d and write it in filename
void Cryptography::xor_encryptToFile(const char* d, const size_t len, const std::string filename,
		const char* key, const size_t key_len) {
	char* copy = new char[len];
	std::memcpy(copy, d, len);

	xor_encryptOrDecryptBinary(copy, len, key, key_len);

	std::ofstream outfile(filename.c_str(), std::ofstream::binary);
	outfile.write(copy, len);
	delete[] copy;
}

void Cryptography::cryptOrDecryptCharsXORKey(char* input, size_t size,
		pop::UI32 key) {
	pop::UI32 * ptrUI32 = reinterpret_cast<pop::UI32*>(input);
	for (unsigned int i = 0; i < std::floor(size / 4.); i++) {
		*ptrUI32 = (*ptrUI32) ^ (key);
		ptrUI32++;
	}
}

void Cryptography::cryptOrDecryptFileXORKey(std::string inputfile,
		std::string outputfile, pop::UI32 key) {
	std::ifstream infile(inputfile.c_str(), std::ios::binary);
	infile.seekg(0, infile.end);
	long size = infile.tellg();

	char * c = new char[size];
	infile.seekg(0, std::ios::beg);
	infile.read(c, size);

	cryptOrDecryptCharsXORKey(c, size, key);

	std::ofstream outfile(outputfile.c_str(), std::ofstream::binary);
	outfile.write(c, size);

	delete[] c;

	outfile.close();
	infile.close();
}

}
