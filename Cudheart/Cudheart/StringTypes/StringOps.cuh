#pragma once

#include "../Util.cuh"
#include "../Arrays/Matrix.cuh"
#include "../Arrays/Vector.cuh"
#include "../Arrays/VectorOps.cuh"
#include "../Exceptions/Exceptions.cuh"
#include "StringType.cuh"

using Cudheart::NDArrays::Matrix;
using Cudheart::NDArrays::Vector;
using Cudheart::NDArrays::NDArray;
using Cudheart::VectorOps::zeros;
using namespace Cudheart::Exceptions;


namespace Cudheart::StringOps {
	inline NDArray<StringType*>* add(NDArray<StringType*>* x1, NDArray<StringType*>* x2) {
		x1->assertMatchShape(x2->getShape());

		NDArray<StringType*>* out = x1->emptyLike();

		for (int i = 0; i < x1->getSize(); i++) {
			out->set(i, new StringType(x1->get(i)->m_str + x2->get(i)->m_str));
		}

		return out;
	}

	inline NDArray<StringType*>* multiply(NDArray<StringType*>* x, NDArray<int>* l) {
		x->assertMatchShape(l->getShape());

		NDArray<StringType*>* arr = x->emptyLike();

		for (int i = 0; arr->getSize(); i++) {
			string s = "";
			for (int j = 0; j < l->get(i); j++) {
				s += x->get(i)->m_str;
			}
			arr->set(i, new StringType(s));
		}

		return arr;
	}

	inline NDArray<StringType*>* capitalize(NDArray<StringType*>* x) {
		NDArray<StringType*>* out = x->emptyLike();

		for (int i = 0; i < out->getSize(); i++) {
			const char* st = x->get(i)->c_str();
			int code = toupper(st[0]);
			std::string str = "" + code;

			for (int j = 1; j < x->get(j)->str().size(); j++) {
				str += x->get(j)->str().at(j);
			}

			out->set(i, new StringType(str));
		}

		return out;
	}

	inline NDArray<StringType*>* replace(NDArray<StringType*>* a, char oldChar, std::string newChar) {
		NDArray<StringType*>* out = a->emptyLike();

		for (int i = 0; i < out->getSize(); i++) {
			std::string str = "";
			std::string old = a->get(i)->str();

			for (int j = 0; j < old.size(); j++) {
				if (old.at(j) == oldChar) {
					str += newChar;
				}
				else {
					str += old.at(j);
				}
			}

			out->set(i, new StringType(str));
		}

		return out;
	}
	
	inline NDArray<StringType*>* expandTabs(NDArray<StringType*>* a, int tabsize = 8) {
		std::string str = "";

		for (; tabsize > 0; tabsize--) {
			str += " ";
		}

		return replace(a, '\t', str);
	}

	inline NDArray<StringType*>* join(NDArray<StringType*>* sep, NDArray<StringType*>* seq) {
		sep->assertMatchShape(seq->getShape());
		NDArray<StringType*>* out = sep->emptyLike();

		for (int i = 0; i < sep->getSize(); i++) {
			std::string str = sep->get(i)->str() + seq->get(i)->str();

			out->set(i, new StringType(str));
		}

		return out;
	}

	inline NDArray<StringType*>* lower(NDArray<StringType*>* x) {
		NDArray<StringType*>* out = x->emptyLike();

		for (int i = 0; i < out->getSize(); i++) {
			std::string str = "";

			for (int j = 0; j < x->get(j)->str().size(); j++) {
				str += tolower(x->get(j)->str().at(j));
			}

			out->set(i, new StringType(str));
		}

		return out;
	}

	inline NDArray<StringType*>* lJust(NDArray<StringType*>* a, int width, char fillChar = ' ') {
		NDArray<StringType*>* out = a->emptyLike();

		for (int i = 0; i < a->getSize(); i++) {
			std::string str = a->get(i)->str();
			int size = a->get(i)->str().size();

			for (int j = 0; j < (size - width); j++) {
				str += fillChar;
			}

			out->set(i, new StringType(str));
		}

		return out;
	}

	inline NDArray<StringType*>* rJust(NDArray<StringType*>* a, int width, char fillChar = ' ') {
		NDArray<StringType*>* out = a->emptyLike();

		for (int i = 0; i < a->getSize(); i++) {
			std::string str = "";
			int size = a->get(i)->str().size();

			for (int j = 0; j < (size - width); j++) {
				str += fillChar;
			}

			str += a->get(i)->str();

			out->set(i, new StringType(str));
		}

		return out;
	}
}