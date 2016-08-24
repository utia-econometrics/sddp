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

#if !(defined TRNG_YARN5_HPP)

#define TRNG_YARN5_HPP

#include <ostream>
#include <istream>
#include <stdexcept>
#include "cuda.hpp"
#include "utility.hpp"
#include "int_math.hpp"

namespace trng {
  
  class yarn5;
  
  class yarn5 {
  public:
  
    // Uniform random number generator concept
    typedef long result_type;
    TRNG_CUDA_ENABLE
    result_type operator()() const;
  private:
    static const result_type modulus=2147483647l;
    static const result_type gen=123567893l;
  public:
    static const result_type min=0l;
    static const result_type max=modulus-1;

    // Parameter and status classes
    class parameter_type;
    class status_type;

    class parameter_type {
      result_type a1, a2, a3, a4, a5;
      static int_math::power<yarn5::modulus, yarn5::gen> g;
    public:
      parameter_type() :
        a1(0), a2(0), a3(0), a4(0), a5(0) { };
      parameter_type(result_type a1, result_type a2,
                     result_type a3, result_type a4,
		     result_type a5) :
        a1(a1), a2(a2), a3(a3), a4(a4), a5(a5) { };

      friend class yarn5;

      // Equality comparable concept
      friend bool operator==(const parameter_type &, const parameter_type &);
      friend bool operator!=(const parameter_type &, const parameter_type &);

      // Streamable concept
      template<typename char_t, typename traits_t>
      friend std::basic_ostream<char_t, traits_t> &
      operator<<(std::basic_ostream<char_t, traits_t> &out,
                 const parameter_type &P) {
        std::ios_base::fmtflags flags(out.flags());
        out.flags(std::ios_base::dec | std::ios_base::fixed |
                  std::ios_base::left);
        out << '('
            << P.a1 << ' ' << P.a2 << ' ' << P.a3 << ' ' << P.a4 << ' ' << P.a5
            << ')';
        out.flags(flags);
        return out;
      }

      template<typename char_t, typename traits_t>
      friend std::basic_istream<char_t, traits_t> &
      operator>>(std::basic_istream<char_t, traits_t> &in,
                 parameter_type &P) {
        parameter_type P_new;
        std::ios_base::fmtflags flags(in.flags());
        in.flags(std::ios_base::dec | std::ios_base::fixed |
                 std::ios_base::left);
        in >> utility::delim('(')
           >> P_new.a1 >> utility::delim(' ')
           >> P_new.a2 >> utility::delim(' ')
           >> P_new.a3 >> utility::delim(' ')
           >> P_new.a4 >> utility::delim(' ')
           >> P_new.a5 >> utility::delim(')');
        if (in)
          P=P_new;
        in.flags(flags);
        return in;
      }

    };

    class status_type {
      result_type r1, r2, r3, r4, r5;
    public:
      status_type() : r1(0), r2(1), r3(1), r4(1), r5(1) { };
      status_type(result_type r1, result_type r2, 
		  result_type r3, result_type r4, 
		  result_type r5) :
        r1(r1), r2(r2), r3(r3), r4(r4), r5(r5) { };
      
      friend class yarn5;

      // Equality comparable concept
      friend bool operator==(const status_type &, const status_type &);
      friend bool operator!=(const status_type &, const status_type &);

      // Streamable concept
      template<typename char_t, typename traits_t>
      friend std::basic_ostream<char_t, traits_t> &
      operator<<(std::basic_ostream<char_t, traits_t> &out,
                 const status_type &S) {
        std::ios_base::fmtflags flags(out.flags());
        out.flags(std::ios_base::dec | std::ios_base::fixed |
                  std::ios_base::left);
        out << '('
            << S.r1 << ' ' << S.r2 << ' ' << S.r3 << ' ' << S.r4 << ' ' << S.r5
            << ')';
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
           >> S_new.r1 >> utility::delim(' ')
           >> S_new.r2 >> utility::delim(' ')
           >> S_new.r3 >> utility::delim(' ')
           >> S_new.r4 >> utility::delim(' ')
           >> S_new.r5 >> utility::delim(')');
        if (in)
          S=S_new;
        in.flags(flags);
        return in;
      }

    };
     
    static const parameter_type LEcuyer1;
    
    // Random number engine concept
    explicit yarn5(parameter_type=LEcuyer1);
    explicit yarn5(unsigned long, parameter_type=LEcuyer1);
    template<typename gen>
    explicit yarn5(gen &g, parameter_type P=LEcuyer1) : P(P), S() {
      seed(g);
    }

    void seed();
    void seed(unsigned long);
    template<typename gen>
    void seed(gen &g) {
      result_type r1=static_cast<unsigned long>(g())%
        static_cast<unsigned long>(modulus);
      result_type r2=static_cast<unsigned long>(g())%
        static_cast<unsigned long>(modulus);
      result_type r3=static_cast<unsigned long>(g())%
        static_cast<unsigned long>(modulus);
      result_type r4=static_cast<unsigned long>(g())%
        static_cast<unsigned long>(modulus);
      result_type r5=static_cast<unsigned long>(g())%
        static_cast<unsigned long>(modulus);
      S.r1=r1;
      S.r2=r2;
      S.r3=r3;
      S.r4=r4;
      S.r5=r5;
    }
    void seed(result_type, result_type, result_type, result_type, result_type);

    // Equality comparable concept
    friend bool operator==(const yarn5 &, const yarn5 &);
    friend bool operator!=(const yarn5 &, const yarn5 &);

    // Streamable concept
    template<typename char_t, typename traits_t>
    friend std::basic_ostream<char_t, traits_t> &
    operator<<(std::basic_ostream<char_t, traits_t> &out, const yarn5 &R) {
      std::ios_base::fmtflags flags(out.flags());
      out.flags(std::ios_base::dec | std::ios_base::fixed |
                std::ios_base::left);
      out << '[' << yarn5::name() << ' ' << R.P << ' ' << R.S << ']';
      out.flags(flags);
      return out;
    }

    template<typename char_t, typename traits_t>
    friend std::basic_istream<char_t, traits_t> &
    operator>>(std::basic_istream<char_t, traits_t> &in, yarn5 &R) {
      yarn5::parameter_type P_new;
      yarn5::status_type S_new;
      std::ios_base::fmtflags flags(in.flags());
      in.flags(std::ios_base::dec | std::ios_base::fixed |
               std::ios_base::left);
      in >> utility::ignore_spaces();
      in >> utility::delim('[')
         >> utility::delim(yarn5::name()) >> utility::delim(' ')
         >> P_new >> utility::delim(' ')
         >> S_new >> utility::delim(']');
      if (in) {
        R.P=P_new;
        R.S=S_new;
      }
      in.flags(flags);
      return in;
    }

    // Parallel random number generator concept
    TRNG_CUDA_ENABLE
    void split(unsigned int, unsigned int);
    TRNG_CUDA_ENABLE
    void jump2(unsigned int);
    TRNG_CUDA_ENABLE
    void jump(unsigned long long);

    // Other useful methods
    static const char * name();
    TRNG_CUDA_ENABLE
    long operator()(long) const;

  private:
    parameter_type P;
    mutable status_type S;
    static const char * const name_str;
    
    TRNG_CUDA_ENABLE
    void backward();
    TRNG_CUDA_ENABLE
    void step() const;
  };
    
  // Inline and template methods

  TRNG_CUDA_ENABLE
  inline void yarn5::step() const {
    unsigned long long t(static_cast<unsigned long long>(P.a1)*
			 static_cast<unsigned long long>(S.r1)+
			 static_cast<unsigned long long>(P.a2)*
			 static_cast<unsigned long long>(S.r2)+
			 static_cast<unsigned long long>(P.a3)*
			 static_cast<unsigned long long>(S.r3)+
			 static_cast<unsigned long long>(P.a4)*
			 static_cast<unsigned long long>(S.r4));
    if (t>=2ull*modulus*modulus)
      t-=2ull*modulus*modulus;
    t+=static_cast<unsigned long long>(P.a5)*
      static_cast<unsigned long long>(S.r5);
    S.r5=S.r4;  S.r4=S.r3;  S.r3=S.r2;  S.r2=S.r1;  S.r1=int_math::modulo<modulus, 5>(t);

  }

  TRNG_CUDA_ENABLE
  inline yarn5::result_type yarn5::operator()() const {
    step();
#if defined __CUDA_ARCH__
    if (S.r1==0) 
      return 0;
    yarn5::result_type n=S.r1;
    long long p(1ll), t(gen);
    while (n>0) {
      if ((n&0x1)==0x1)
	p=int_math::modulo<modulus, 1>(p*t);
      t=int_math::modulo<modulus, 1>(t*t);
      n/=2;
    }
    return static_cast<yarn5::result_type>(p);
#else
    return (S.r1==0) ? 0 : P.g(S.r1);
#endif
  }
  
  TRNG_CUDA_ENABLE
  inline long yarn5::operator()(long x) const {
    return static_cast<long>(utility::uniformco<double, yarn5>(*this)*x);
  }

  // Parallel random number generator concept
  TRNG_CUDA_ENABLE
  inline void yarn5::split(unsigned int s, unsigned int n) {
#if !(defined __CUDA_ARCH__)
    if (s<1 or n>=s)
      utility::throw_this(std::invalid_argument("invalid argument for trng::yarn5::split"));
#endif
    if (s>1) {
      jump(n+1);  long q0=S.r1;
      jump(s);    long q1=S.r1;
      jump(s);    long q2=S.r1;
      jump(s);    long q3=S.r1;
      jump(s);    long q4=S.r1;
      jump(s);    long q5=S.r1;
      jump(s);    long q6=S.r1;
      jump(s);    long q7=S.r1;
      jump(s);    long q8=S.r1;
      jump(s);    long q9=S.r1;
      long a[5], b[25];
      a[ 0]=q5;  b[ 0]=q4;  b[ 1]=q3;  b[ 2]=q2;  b[ 3]=q1;  b[ 4]=q0;
      a[ 1]=q6;  b[ 5]=q5;  b[ 6]=q4;  b[ 7]=q3;  b[ 8]=q2;  b[ 9]=q1;
      a[ 2]=q7;  b[10]=q6;  b[11]=q5;  b[12]=q4;  b[13]=q3;  b[14]=q2;
      a[ 3]=q8;  b[15]=q7;  b[16]=q6;  b[17]=q5;  b[18]=q4;  b[19]=q3;
      a[ 4]=q9;  b[20]=q8;  b[21]=q7;  b[22]=q6;  b[23]=q5;  b[24]=q4;
      int_math::gauss<5>(b, a, modulus);
      P.a1=a[0];   P.a2=a[1];   P.a3=a[2];   P.a4=a[3];   P.a5=a[4];
      S.r1=q4;     S.r2=q3;     S.r3=q2;     S.r4=q1;     S.r5=q0;
      for (int i=0; i<5; ++i)
	backward();
    }
  }
  
  TRNG_CUDA_ENABLE
  inline void yarn5::jump2(unsigned int s) {
    long b[25], c[25], d[5], r[5];
    long t1(P.a1), t2(P.a2), t3(P.a3), t4(P.a4), t5(P.a5);
    b[ 0]=P.a1;  b[ 1]=P.a2;  b[ 2]=P.a3;  b[ 3]=P.a4;  b[ 4]=P.a5;
    b[ 5]=1l;    b[ 6]=0l;    b[ 7]=0l;    b[ 8]=0l;    b[ 9]=0l;
    b[10]=0l;    b[11]=1l;    b[12]=0l;    b[13]=0l;    b[14]=0l;
    b[15]=0l;    b[16]=0l;    b[17]=1l;    b[18]=0l;    b[19]=0l;
    b[20]=0l;    b[21]=0l;    b[22]=0l;    b[23]=1l;    b[24]=0l;
    for (unsigned int i(0); i<s; ++i)
      if ((i&1)==0)
        int_math::matrix_mult<5>(b, b, c, modulus);
      else
        int_math::matrix_mult<5>(c, c, b, modulus);
    r[0]=S.r1;  r[1]=S.r2;  r[2]=S.r3;  r[3]=S.r4;  r[4]=S.r5;
    if ((s&1)==0)
      int_math::matrix_vec_mult<5>(b, r, d, modulus);
    else
      int_math::matrix_vec_mult<5>(c, r, d, modulus);
    S.r1=d[0];  S.r2=d[1];  S.r3=d[2];  S.r4=d[3];  S.r5=d[4];
    P.a1=t1;    P.a2=t2;    P.a3=t3;    P.a4=t4;    P.a5=t5;
  }

  TRNG_CUDA_ENABLE
  inline void yarn5::jump(unsigned long long s) {
    if (s<16) {
      for (unsigned int i(0); i<s; ++i) 
	step();
    } else {
      unsigned int i(0);
      while (s>0) {
	if (s%2==1)
	  jump2(i);
	++i;
	s>>=1;
      }
    }
  }

  TRNG_CUDA_ENABLE
  inline void yarn5::backward() {
    long t;
    if (P.a5!=0l) {
      t=S.r1;
      t-=static_cast<long>((static_cast<long long>(P.a1)*
                            static_cast<long long>(S.r2))%modulus);
      if (t<0l)
        t+=modulus;
      t-=static_cast<long>((static_cast<long long>(P.a2)*
                            static_cast<long long>(S.r3))%modulus);
      if (t<0l)
        t+=modulus;
      t-=static_cast<long>((static_cast<long long>(P.a3)*
                          static_cast<long long>(S.r4))%modulus);
      if (t<0l)
        t+=modulus;
      t-=static_cast<long>((static_cast<long long>(P.a4)*
                          static_cast<long long>(S.r5))%modulus);
      if (t<0l)
        t+=modulus;
      t=static_cast<long>((static_cast<long long>(t)*
                           static_cast<long long>
                           (int_math::modulo_invers(P.a5, modulus)))%modulus);
    } else if (P.a4!=0l) {
      t=S.r1;
      t-=static_cast<long>((static_cast<long long>(P.a1)*
			    static_cast<long long>(S.r2))%modulus);
      if (t<0l)
	t+=modulus;
      t-=static_cast<long>((static_cast<long long>(P.a2)*
			    static_cast<long long>(S.r3))%modulus);
      if (t<0l)
	t+=modulus;
      t-=static_cast<long>((static_cast<long long>(P.a3)*
                          static_cast<long long>(S.r4))%modulus);
      if (t<0l)
	t+=modulus;
      t=static_cast<long>((static_cast<long long>(t)*
			   static_cast<long long>
			   (int_math::modulo_invers(P.a4, modulus)))%modulus);
    } else if (P.a3!=0l) {
      t=S.r2;
      t-=static_cast<long>((static_cast<long long>(P.a1)*
			    static_cast<long long>(S.r3))%modulus);
      if (t<0l)
	t+=modulus;
      t-=static_cast<long>((static_cast<long long>(P.a2)*
			    static_cast<long long>(S.r4))%modulus);
      if (t<0l)
	t+=modulus;
      t=static_cast<long>((static_cast<long long>(t)*
			   static_cast<long long>
			   (int_math::modulo_invers(P.a3, modulus)))%modulus);
    } else if (P.a2!=0l) {
      t=S.r3;
      t-=static_cast<long>((static_cast<long long>(P.a1)*
			    static_cast<long long>(S.r4))%modulus);
      if (t<0l)
	t+=modulus;
      t=static_cast<long>((static_cast<long long>(t)*
			   static_cast<long long>
			   (int_math::modulo_invers(P.a2, modulus)))%modulus);
    } else if (P.a1!=0l) {
      t=S.r4;
      t=static_cast<long>((static_cast<long long>(t)*
			   static_cast<long long>
			   (int_math::modulo_invers(P.a1, modulus)))%modulus);
    } else
      t=0l;
    S.r1=S.r2;  S.r2=S.r3;  S.r3=S.r4;  S.r4=S.r5;  S.r5=t;
  }

}
  
#endif
