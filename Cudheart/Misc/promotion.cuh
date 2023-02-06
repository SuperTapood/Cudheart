#include <cmath>
#include <complex>
#include <string>

#include <type_traits>
#include <complex>
#include <iostream>

template <template <class...> class TT, class... Args>
std::true_type is_tt_impl(TT<Args...>);
template <template <class...> class TT>
std::false_type is_tt_impl(...);

template <template <class...> class TT, class T>
using is_tt = decltype(is_tt_impl<TT>(std::declval<typename std::decay_t<T>>()));

#define conditional(a, b, c) std::conditional_t<a, b, c>
#define is_same(a, b) std::is_same_v<a, b>
#define is_integral(a) std::is_integral_v<a>
#define is_floating(a) std::is_floating_point_v<a>
#define is_signed(a) std::is_signed_v<a>
#define is_compatible(a) (is_tt<std::complex, a>::value || is_integral(a) || is_floating(a))
#define eitherSigned(a, b) is_signed(A) || is_signed(B)

template <typename A, typename B>
struct promote_args {
    static const bool compA = is_compatible(A);
    static_assert(compA, "the given type is not compatible with type promotion :)");
    static const bool compB = is_compatible(B);
    static_assert(compB, "the given type is not compatible with type promotion :)");


    using type = typename conditional(
        eitherSigned(A, B), 
            conditional(is_integral(A) && is_integral(B), 
                conditional(is_same(long long, A) || is_same(long long, B), long long, 
                conditional(is_same(long, A) || is_same(long, B), long,
                conditional(is_same(int, A) || is_same(int, B), int,
                short       
                ))),
            
                conditional(is_same(long double, A) || is_same(long double, B), long double,
                conditional(is_same(double, A) || is_same(double, B), double,
                float
            ))),
            conditional(is_same(unsigned long long, A) || is_same(unsigned long long, B), unsigned long long,
            conditional(is_same(unsigned long, A) || is_same(unsigned long, B), unsigned long,
            conditional(is_same(unsigned int, A) || is_same(unsigned int, B), unsigned int,
                unsigned short
            )))
    );
};

template <typename T> struct promote_args<T, T> {using type = T; };
template <typename A, typename B> struct promote_args<std::complex<A>, std::complex<B>> { using type = std::complex<promote_args<A, B>::type>; };
template <typename A, typename B> struct promote_args<A, std::complex<B>> { using type = std::complex<promote_args<A, B>::type>; };
template <typename A, typename B> struct promote_args<std::complex<A>, B> { using type = std::complex<promote_args<A, B>::type>; };

#undef conditional(a, b, c)
#undef is_same(a, b)
#undef is_integral(a)
#undef is_floating(a)
#undef is_signed(a)
#undef is_compatible(a)
#undef eitherSigned(a, b)

#define promote(a, b) promote_args<a, b>::type
#define promote3(a, b, c) promote(promote(a, b), c)



void stupidTemplates() {
    promote(std::complex<int>, long) v;
    cout << "v is " << typeid(v).name() << endl;
    cout << (std::complex<float>)5 << endl;
}