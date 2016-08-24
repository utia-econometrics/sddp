// Copyright (c) 2000-2010, Heiko Bauke
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
// 
//   * Redistributions of source code must retain the above copyright
//     notice, this list of conditions and the following disclaimer.  
// 
//   * Redistributions in binary form must reproduce the above
//     copyright notice, this list of conditions and the following
//     disclaimer in the documentation and/or other materials provided
//     with the distribution.  
// 
//   * Neither the name of the copyright holder nor the names of its
//     contributors may be used to endorse or promote products derived
//     from this software without specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
// FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
// COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
// INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
// OF THE POSSIBILITY OF SUCH DAMAGE.

#if !(defined TRNG_LAGFIB4XOR_HPP)

#define TRNG_LAGFIB4XOR_HPP

#include "limits.hpp"
#include <climits>
#include <stdexcept>
#include <ostream>
#include <istream>
#include <sstream>
#include "utility.hpp"
#include "minstd.hpp"

namespace trng {

  template<typename integer_type,
	   unsigned int A, unsigned int B, 
	   unsigned int C, unsigned int D>
  class lagfib4xor;
  
  template<typename integer_type,
	   unsigned int A, unsigned int B, 
	   unsigned int C, unsigned int D>
  class lagfib4xor {
  public:
    
    // Uniform random number generator concept
    typedef integer_type result_type;
    result_type operator()() const {
      step();  
      return S.r[S.index];
    }
    static const result_type min=0;
    static const result_type max=~result_type(0);
    
    // Parameter and status classes
    class status_type;
    
    class status_type {
      result_type r[utility::ceil2<D>::result];
      unsigned int index;
      static unsigned int size() {
	return utility::ceil2<D>::result;
      }
    public:
      status_type() { 
	for (unsigned int i=0; i<size(); ++i)
	  r[i]=0;
	index=0;
      };
      
      friend class lagfib4xor;
      
      // Equality comparable concept
      friend bool operator==(const status_type &a, const status_type &b) {
	if (a.index!=b.index) 
	  return false;
	for (unsigned int i=0; i<a.size(); ++i)
	  if (a.r[i]!=b.r[i])
	    return false;
	return true;
      }
      friend bool operator!=(const status_type &a, const status_type &b) {
	return not (a==b);
      }

      // Streamable concept
      template<typename char_t, typename traits_t>
      friend std::basic_ostream<char_t, traits_t> & 
      operator<<(std::basic_ostream<char_t, traits_t> &out, 
		 const status_type &S) {
	std::ios_base::fmtflags flags(out.flags());
	out.flags(std::ios_base::dec | std::ios_base::fixed | 
		  std::ios_base::left);
	out << '(' 
	    << S.index;
 	for (unsigned int i=0; i<S.size(); ++i)
	  out << ' ' << S.r[i];
	out << ')';
	out.flags(flags);
	return out;
      }

      template<typename char_t, typename traits_t>
      friend std::basic_istream<char_t, traits_t> & 
      operator>>(std::basic_istream<char_t, traits_t> &in, 
		 status_type &S) {
	status_type S_new;
	std::ios_base::fmtflags flags(in.flags());
	in.flags(std::ios_base::dec | std::ios_base::fixed | 
		 std::ios_base::left);
	in >> utility::delim('(')
	   >> S_new.index;
 	for (unsigned int i=0; i<S.size(); ++i)
 	  in >> utility::delim(' ') >> S_new.r[i];
	in >> utility::delim(')');
	if (in)
	  S=S_new;
	in.flags(flags);
	return in;
      }
      
    };
    
    // Random number engine concept
    lagfib4xor() : S() {
      seed();
    }
    
    explicit lagfib4xor(unsigned long s) : S() {
      seed(s);
    }
    
    template<typename gen>
    explicit lagfib4xor(gen &g) : S() {
      seed(g);
    }
    
    void seed() {
      seed(0);
    }
    
    void seed(unsigned long s) {
      minstd R(s);
      seed(R);
    }
    
    template<typename gen>
    void seed(gen &g) {
      for (unsigned int i=0; i<D; ++i) {
        result_type r=0;
        for (unsigned int j=0; j<std::numeric_limits<result_type>::digits; ++j) {
          r<<=1;
	  if (g()-gen::min>gen::max/2)
            ++r;
        }
        S.r[i]=r;
      }
      S.index=D-1;
    }
    
    // Equality comparable concept
    friend bool operator==(const lagfib4xor &R1, const lagfib4xor &R2) {
      return R1.S==R2.S;
    }
      
    friend bool operator!=(const lagfib4xor &R1, const lagfib4xor &R2) {
      return not (R1==R2);
    }
    
    // Streamable concept
    template<typename char_t, typename traits_t>
    friend std::basic_ostream<char_t, traits_t> & 
    operator<<(std::basic_ostream<char_t, traits_t> &out, const lagfib4xor &R) {
      std::ios_base::fmtflags flags(out.flags());
      out.flags(std::ios_base::dec | std::ios_base::fixed | 
		std::ios_base::left);
      out << '[' << lagfib4xor::name() << ' ' << R.S << ']';
      out.flags(flags);
      return out;
    }
    
    template<typename char_t, typename traits_t>
    friend std::basic_istream<char_t, traits_t> & 
    operator>>(std::basic_istream<char_t, traits_t> &in, lagfib4xor &R) {
      typename lagfib4xor::status_type S_new;
      std::ios_base::fmtflags flags(in.flags());
      in.flags(std::ios_base::dec | std::ios_base::fixed | 
	       std::ios_base::left);
      in >> utility::ignore_spaces();
      in >> utility::delim('[')
	 >> utility::delim(lagfib4xor::name()) >> utility::delim(' ')
	 >> S_new >> utility::delim(']');
      if (in) 
	R.S=S_new;
      in.flags(flags);
      return in;
    }
    
    // Other useful methods
    static const char * name() {
      static char name_c_str[64]={'\0'};
      if (name_c_str[0]=='\0') {
	std::stringstream name_str;
	name_str << "lagfib4xor_" << std::numeric_limits<result_type>::digits << '_'
		 << A << '_' << B << '_' << C << '_' << D;
	int i=0;
	const char *p=name_str.str().c_str();
	while (p[i]!='\0' and i<63) {
	  name_c_str[i]=p[i];
	  ++i;
	}
	name_c_str[i]='\0';
      }
      return name_c_str;
    }
    long operator()(long x) const {
      return static_cast<long>(utility::uniformco<double, lagfib4xor>(*this)*x);
    }
    
  private:
    mutable status_type S;
    
    void step() const {
      S.index++;
      S.index&=utility::mask<D>::result;
      S.r[S.index]=
	S.r[(S.index-A)&utility::mask<D>::result] ^ 
	S.r[(S.index-B)&utility::mask<D>::result] ^ 
	S.r[(S.index-C)&utility::mask<D>::result] ^ 
	S.r[(S.index-D)&utility::mask<D>::result];
    }
  };
  
  typedef lagfib4xor<unsigned long,       471, 1586,  6988,  9689> Ziff_ul;
  typedef lagfib4xor<unsigned long long,  471, 1586,  6988,  9689> Ziff_ull;
  typedef lagfib4xor<unsigned long,       168,  205,   242,   521> lagfib4xor_521_ul;
  typedef lagfib4xor<unsigned long long,  168,  205,   242,   521> lagfib4xor_521_ull;
  typedef lagfib4xor<unsigned long,       147,  239,   515,   607> lagfib4xor_607_ul;
  typedef lagfib4xor<unsigned long long,  147,  239,   515,   607> lagfib4xor_607_ull;
  typedef lagfib4xor<unsigned long,       418,  705,   992,  1279> lagfib4xor_1279_ul;
  typedef lagfib4xor<unsigned long long,  418,  705,   992,  1279> lagfib4xor_1279_ull;
  typedef lagfib4xor<unsigned long,       305,  610,   915,  2281> lagfib4xor_2281_ul;
  typedef lagfib4xor<unsigned long long,  305,  610,   915,  2281> lagfib4xor_2281_ull;
  typedef lagfib4xor<unsigned long,       576,  871,  1461,  3217> lagfib4xor_3217_ul;
  typedef lagfib4xor<unsigned long long,  576,  871,  1461,  3217> lagfib4xor_3217_ull;
  typedef lagfib4xor<unsigned long,      1419, 1736,  2053,  4423> lagfib4xor_4423_ul;
  typedef lagfib4xor<unsigned long long, 1419, 1736,  2053,  4423> lagfib4xor_4423_ull;
  typedef lagfib4xor<unsigned long,       471, 2032,  4064,  9689> lagfib4xor_9689_ul;
  typedef lagfib4xor<unsigned long long,  471, 2032,  4064,  9689> lagfib4xor_9689_ull;
  typedef lagfib4xor<unsigned long,      3860, 7083, 11580, 19937> lagfib4xor_19937_ul;
  typedef lagfib4xor<unsigned long long, 3860, 7083, 11580, 19937> lagfib4xor_19937_ull;

}

#endif
