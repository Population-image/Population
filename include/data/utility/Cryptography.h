/******************************************************************************\
|*                   Population library for C++ X.X.X                         *|
|*----------------------------------------------------------------------------*|
The Population License is similar to the MIT license in adding this clause:
for any writing public or private that has resulted from the use of the
software population, the reference of this book "Population library, 2012,
Vincent Tariel" shall be included in it.

So, the terms of the Population License are:

Copyright © 2012-2015, Tariel Vincent
Copyright © 2015, Aublin Pierre Louis

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to
deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software and for any writing
public or private that has resulted from the use of the software population,
the reference of this book "Population library, 2012, Vincent Tariel" shall
be included in it.

The Software is provided "as is", without warranty of any kind, express or
implied, including but not limited to the warranties of merchantability,
fitness for a particular purpose and noninfringement. In no event shall the
authors or copyright holders be liable for any claim, damages or other
liability, whether in an action of contract, tort or otherwise, arising
from, out of or in connection with the software or the use or other dealings
in the Software.
\***************************************************************************/

#ifndef Cryptography_HPP
#define Cryptography_HPP

#include "PopulationConfig.h"
#include "data/typeF/TypeF.h"

namespace pop
{

class POP_EXPORTS Cryptography
{
public:
	// Encrypt the text text using the key key of size key_len. Returns an array of char of size len
	static char* xor_encryptText(const std::string text, int* len, const char* key, const int key_len);

	// Decrypt the binary data d of size len using the key key of size key_len and return the decrypted data as a string
	static std::string xor_decryptText(const char* d, const int len, const char* key, const int key_len);

	// Encrypt or decrypt the binary data d of size len in place using the key key of size key_len
	static void xor_encryptOrDecryptBinary(char* d, const int len, const char* key, const int key_len);

    // Open the encrypted file filename, decrypt it, and return the result of the decryption as an array of char of size *len
    static char* xor_decryptFile(const std::string filename, int *len, const char* key, const int key_len);

	// Encrypt d and write it in filename
	static void xor_encryptToFile(const char* d, const int len, const std::string filename, const char* key, const int key_len);

    // (de)crypt the array input, of size size, in-place
	// Obsolete, please use the ones above for a better key
    static void cryptOrDecryptCharsXORKey(char *input, int size, pop::UI32 key =0xAAF588BB);
    static void cryptOrDecryptFileXORKey(std::string inputfile,std::string outputfile, pop::UI32 key =0xAAF588BB);
};
}

#endif
