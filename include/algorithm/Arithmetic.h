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

#ifndef ARITHMETIC_H
#define ARITHMETIC_H
#include"data/typeF/TypeF.h"
#include "PopulationConfig.h"

/*!
     * \defgroup Arithmetic Arithmetic
     * \ingroup Algorithm
     * \brief basic arithmetic facilities

*/

namespace pop
{
struct POP_EXPORTS Arithmetic
{

    /*!
     * \class pop::Arithmetic
     * \ingroup Arithmetic
     * \brief Arithmetic
     *
    */

    /*!
     * \brief find exponent k for a^k=i with a the base and i the value
     * \param value value
     * \param base base
     * \return exponent
     *
     * \code
    for(unsigned int i=2;i<100;i++){
        std::cout<<"i="<<i<<"and for 2^k=i, k="<<Arithmetic::exposant(i,2)<<std::endl;
    }
     * \endcode
     */
    static inline F32 exposant(F32 value, int base=2){
        return std::log(value)/std::log(base);
    }
    /*!
     * \brief find q and r for a = b*q + r  with 0<=r<b
     * \param a first integer
     * \param b second integer not equal to 0
     * \param q quotient
     * \param q remainder
     *
     *
     * We can prove that 2{3*n}=1 modulo(7) and in this special case, we find the same result:
     * \code
        long a = std::pow(2,3*4);
        long b = 7;
        long q,r;
        Arithmetic::euclideanDivision(a,b,q,r);
        std::cout<<q<<std::endl;
        std::cout<<r<<std::endl;
     * \endcode
     */
    template<typename Integer>
    static inline void euclideanDivision( Integer  a, Integer b, Integer & q,  Integer &r){
        q = a/b;
        r = a-q*b;
    }
    /*!
     * \brief Greatest Common Divisor (GCD)
     * \param a first integer
     * \param b second integer
     * \return GCD
     *
     *
     * We can prove that 2{3*n}=1 modulo(7) and in this special case, we find the same result:
     * \code
        std::cout<<GCD(28,21)<<std::endl;
     * \endcode
     */
    template<typename Integer>
    static inline unsigned long GCD(Integer  a, Integer  b){
        a = std::abs(a);
        b = std::abs(b);
        Integer r_i,r_i_1;
        if(a>b){
            r_i=a;r_i_1=b;
        }else{
            r_i=b;r_i_1=a;
        }
        while(r_i_1!=0){
            Integer q,r;
            euclideanDivision(r_i,r_i_1,q,r);
            r_i=r_i_1;
            r_i_1 = r;
        }
        return r_i;
    }
    /*!
     * \brief primality test
     * \param n  integer
     * \return true if prime number, false otherwise
     *
     * \code
        std::cout<<isPrime(17)<<std::endl;
        std::cout<<isPrime(121)<<std::endl;
     * \endcode
     */
    template<typename Integer>
    static inline bool isPrime(Integer n){
        if (n <= 3) {
            return n > 1;
        }

        if (n % 2 == 0 || n % 3 == 0) {
            return false;
        }

        for (unsigned short i = 5; i * i <= n; i += 6) {
            if (n % i == 0 || n % (i + 2) == 0) {
                return false;
            }
        }

        return true;
    }
};
}
#endif // ARITHMETIC_H
