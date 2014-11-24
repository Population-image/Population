////////////////////////////////////////////////////////////////////////////////
// The Loki Library
// Copyright (c) 2001 by Andrei Alexandrescu
// This code accompanies the book:
// Alexandrescu, Andrei. "Modern C++ Design: Generic Programming and Design 
//     Patterns Applied". Copyright (c) 2001. Addison-Wesley.
// Permission to use, copy, modify, distribute and sell this software for any 
//     purpose is hereby granted without fee, provided that the above copyright 
//     notice appear in all copies and that both that copyright notice and this 
//     permission notice appear in supporting documentation.
// The author or Addison-Wesley Longman make no representations about the 
//     suitability of this software for any purpose. It is provided "as is" 
//     without express or implied warranty.
////////////////////////////////////////////////////////////////////////////////
#ifndef LOKI_NULLTYPE_INC_
#define LOKI_NULLTYPE_INC_
#include"iostream"
// $Id: NullType.h 751 2006-10-17 19:50:37Z syntheticpp $


namespace Loki
{
////////////////////////////////////////////////////////////////////////////////
// class NullType
// Used as a placeholder for "no type here"
// Useful as an end marker in typelists 
////////////////////////////////////////////////////////////////////////////////
    class NullType {};
    inline std::ostream& operator << (std::ostream& out, const Loki::NullType & ){
        return out;
    }
    inline std::istream& operator >> (std::istream& in, Loki::NullType & ){
        return in;
    }

}   // namespace Loki

#endif // end file guardian
