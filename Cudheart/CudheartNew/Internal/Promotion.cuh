#pragma once

#include <cmath>
#include <complex>
#include <string>
#include <type_traits>
#include <iostream>

// how does this even work

template <template <class...> class TT, class... Args>
std::true_type is_tt_impl(TT<Args...>);
template <template <class...> class TT>
std::false_type is_tt_impl(...);

template <template <class...> class TT, class T>
using is_tt = decltype(is_tt_impl<TT>(std::declval<typename std::decay_t<T>>()));

// why

#define conditional(a, b, c) std::conditional_t<a, b, c>
#define is_same(a, b) std::is_same_v<a, b>
#define is_integral(a) std::is_integral_v<a>
#define is_floating(a) std::is_floating_point_v<a>
#define is_signed(a) std::is_signed_v<a>
#define is_compatible(a) (is_tt<std::complex, a>::value || is_integral(a) || is_floating(a))
#define eitherSigned is_signed(A) || is_signed(B)
#define isEither(a, b, t) is_same(a, t) || is_same(b, t)
#define bothInts is_integral(A) && is_integral(B)
#define ifEitherSet(t, e) conditional(isEither(A, B, t), t, e)
#define ifEitherSigned(t, f) conditional(eitherSigned, t, f)
#define ifBothInt(t, f) conditional(bothInts, t, f)

template <typename A, typename B>
struct promote_args {
	// assert we actually can go forward with this promotion :)
	// this will raise compilation error when failed
	static_assert(is_compatible(A), "the given type is not compatible with type promotion :(");
	static_assert(is_compatible(B), "the given type is not compatible with type promotion :(");

	// what the hell is wrong with this language
	// why am i allowed to do this
	// send help
	using type = typename ifEitherSigned(
		// signed
		ifBothInt(
			// both are ints, so we can use integer types
			ifEitherSet(long long, ifEitherSet(long, ifEitherSet(int, short))),
			// one of them is a float, we can't use integer types :(
			ifEitherSet(long double, ifEitherSet(double, float)))
		,
		// unsigned, we know fo sho it's an int b.c. there are no unsigned floats
		ifEitherSet(unsigned long long, ifEitherSet(unsigned long, ifEitherSet(unsigned int, unsigned short)))
	);
};

template <typename T> struct promote_args<T, T> { using type = T; };
template <typename A, typename B> struct promote_args<std::complex<A>, std::complex<B>> {
	using type = std::complex<promote_args<A, B>::type>;
};
template <typename A, typename B> struct promote_args<A, std::complex<B>> {
	using type = std::complex<promote_args<A, B>::type>;
};
template <typename A, typename B> struct promote_args<std::complex<A>, B> {
	using type = std::complex<promote_args<A, B>::type>;
};

// why tf are you like this
// why can't we just have normal macro overloads like in rust
#define promote(a, b) promote_args<a, b>::type
#define promote3(a, b, c) promote2(a, promote2(b, c))
#define promote4(a, b, c, d) promote2(a, promote3(b, c, d))