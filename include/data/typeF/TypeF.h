/******************************************************************************\
|*                   Population library for C++ X.X.X                         *|
|*----------------------------------------------------------------------------*|
The Population License is similar to the MIT license in adding this clause:
for any writing public or private that has resulted from the use of the
software population, the reference of this book "Population library, 2012,
Vincent Tariel" shall be included in it.

So, the terms of the Population License are:

Copyright © 2012-2015, Tariel Vincent

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

#ifndef TypeF
#define TypeF

#include<cmath>

namespace pop
{

/*! \ingroup TypeF
* \defgroup BasicType   {U,}{I,F}{8,16,32,64} as UI8
* \brief Basic Type with fixed bit information
*
* Following the idea developped in <inttypes.h> (see http://en.wikipedia.org/wiki/C_data_types#Fixed-width_integer_types), to enhance the portability of population library,
* we define basic type with fixed size.\n
* The nomenclature convention is {U,}{I,F}{8,16,32,64} standing for
* - {U,}: U means unsigned for all  positive values (no negative "sign" allowed)
* - {I,F}: I means integers for all values 0,1,2,3,... (unsigned) ...,-2,-1,0,1,2 (signed) and F means real numbers
* - {8,16,32,64}: 8 means 8 bits information
*
* For instance UI8 is the numbers, 0,1,2,3...254,255.
*
*/

/*! \typedef UI8
 * \brief Unsigned Integers of 8 bits (0,1,2...255)
 * \ingroup BasicType
 *
 *
 * UI8's are mostly used in pixel/voxel type for grey-level
 */
typedef unsigned char UI8;
/*! \typedef I8
 * \brief Signed Integers of 8 bits (-128,-127,...,126,127)
 * \ingroup BasicType
 *
 */
typedef signed char I8;
/*! \typedef UI16
 * \brief Unsigned Integers of 16 bits (0,1,...,65535)
 * \ingroup BasicType
 *
 * * UI16's are mostly used in pixel/voxel type for labelling
 */
typedef unsigned short UI16;
/*! \typedef I16
 * \brief Signed Integers of 16 bits (-32768,-32767,...,32766,32767)
 * \ingroup BasicType
 *
 */
typedef short I16;
/*! \typedef UI32
 * \brief Unsigned Integers of 32 bits (0,1,...,4294967295)
 * \ingroup BasicType
 *
 * * UI32's are mostly used in pixel/voxel type for labelling
 */
typedef unsigned int UI32;
/*! \typedef I32
 * \brief Signed Integers of 32 bits (-2147483648,...,2147483646,2147483647)
 * \ingroup BasicType
 *
 */
typedef int I32;
/*! \typedef F32
 * \brief float type 32 bits
 * \ingroup BasicType
 *
 * * F32's are mostly used when operations must be done on real numbers
 */
typedef float F32;
/*! \typedef F64
 * \brief float type 64 bits
 * \ingroup BasicType
 *
 * * F32's are mostly used when operations must be done on real numbers
 */
typedef double F64;

}


#endif // TypeF.hPP
