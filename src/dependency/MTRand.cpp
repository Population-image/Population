


// mtrand.cpp, see include file mtrand.h for information

#include"dependency/MTRand.h"
// non-inline function definitions and static member definitions cannot
// reside in header file because of the risk of multiple declarations

// initialization of static private members

#define NMAX  624
#define MMAX 397

unsigned long MTRand_int32::state[624] = {0x0UL};
int MTRand_int32::p(0);
bool MTRand_int32::init = false;
 #if defined(HAVE_THREAD)
tthread::mutex MTRand_int32::_mutex;
#endif
pop::F64 MTRand_int32::_max = 4294967296.;

pop::F64 MTRand_int32::maxValue(){
    return _max;
}
// inline for speed, must therefore reside in header file
 unsigned long MTRand_int32::twiddle(unsigned long u, unsigned long v) {
  return (((u & 0x80000000UL) | (v & 0x7FFFFFFFUL)) >> 1)
    ^ ((v & 1UL) ? 0x9908B0DFUL : 0x0UL);
}

 unsigned long MTRand_int32::rand_int32() { // generate 32 bit random int

 #if defined(HAVE_THREAD)
    _mutex.lock();
#endif
  if (p == NMAX) gen_state(); // new state std::vector needed
// gen_state() is split off to be non-inline, because it is only called once
// in every 624 calls and otherwise irand() would become too big to get inlined
  unsigned long x = state[p];
  p++;
#if defined(HAVE_THREAD)
  _mutex.unlock();
#endif
  x ^= (x >> 11);
  x ^= (x << 7) & 0x9D2C5680UL;
  x ^= (x << 15) & 0xEFC60000UL;
  return x ^ (x >> 18);
}
 MTRand_int32::MTRand_int32() { if (!init) seed(5489UL); init = true; }
MTRand_int32::MTRand_int32(unsigned long s) {if (!init)  seed(s); init = true;p = 0; }
MTRand_int32::MTRand_int32(const unsigned long* array, int size) {if (!init) seed(array, size); init = true;p = 0; }
unsigned long MTRand_int32::operator()() { return rand_int32(); }
unsigned long MTRand_int32::operator()(unsigned int N ) {return rand_int32()%N; }


 MTRand_int32::~MTRand_int32() {}
//ss
void MTRand_int32::gen_state() { // generate new state std::vector
  for (int i = 0; i < (NMAX - MMAX); ++i)
    state[i] = state[i + MMAX] ^ twiddle(state[i], state[i + 1]);
  for (int i = NMAX - MMAX; i < (NMAX - 1); ++i)
    state[i] = state[i + MMAX - NMAX] ^ twiddle(state[i], state[i + 1]);
  state[NMAX - 1] = state[MMAX - 1] ^ twiddle(state[NMAX - 1], state[0]);
  p = 0; // reset position
}
void MTRand_int32::seed(unsigned long s) {  // init by 32 bit seed
  state[0] = s & 0xFFFFFFFFUL; // for > 32 bit machines
  for (int i = 1; i < NMAX; ++i) {
    state[i] = 1812433253UL * (state[i - 1] ^ ( (state[i - 1]) >> 30)) + i;
// see Knuth TAOCP Vol2. 3rd Ed. P.106 for multiplier
// in the previous versions, MSBs of the seed affect only MSBs of the array state
// 2002/01/09 modified by Makoto Matsumoto
    state[i] &= 0xFFFFFFFFUL; // for > 32 bit machines
  }
  p = NMAX; // force gen_state() to be called for next random number
}

void MTRand_int32::seed(const unsigned long* array, int size) { // init by array
  seed(19650218UL);
  int i = 1, j = 0;
  for (int k = ((NMAX > size) ? NMAX : size); k; --k) {
    state[i] = (state[i] ^ ((state[i - 1] ^ (state[i - 1] >> 30)) * 1664525UL))
      + array[j] + j; // non linear
    state[i] &= 0xFFFFFFFFUL; // for > 32 bit machines
    ++j; j %= size;
    if ((++i) == NMAX) { state[0] = state[NMAX - 1]; i = 1; }
  }
  for (int k = NMAX - 1; k; --k) {
    state[i] = (state[i] ^ ((state[i - 1] ^ (state[i - 1] >> 30)) * 1566083941UL)) - i;
    state[i] &= 0xFFFFFFFFUL; // for > 32 bit machines
    if ((++i) == NMAX) { state[0] = state[NMAX - 1]; i = 1; }
  }
  state[0] = 0x80000000UL; // MSB is 1; assuring non-zero initial array
  p = NMAX; // force gen_state() to be called for next random number
}
MTRand::MTRand() : MTRand_int32() {}
MTRand::MTRand(unsigned long seed) : MTRand_int32(seed) {}
MTRand::MTRand(const unsigned long* seed, int size) : MTRand_int32(seed, size) {}
MTRand::~MTRand() {}
pop::F64 MTRand::operator()() {
  return static_cast<pop::F64>(rand_int32()) * (1. / 4294967296.); } // divided by 2^32
