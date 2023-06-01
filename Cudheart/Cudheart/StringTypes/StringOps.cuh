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

		for (int i = 0; i < x1->size(); i++) {
			out->set(i, new StringType(x1->get(i)->str() + x2->get(i)->str()));
		}

		return out;
	}

	inline NDArray<StringType*>* multiply(NDArray<StringType*>* x, NDArray<int>* l) {
		x->assertMatchShape(l->getShape());

		NDArray<StringType*>* arr = x->emptyLike();

		for (int i = 0; arr->size(); i++) {
			string s = "";
			for (int j = 0; j < l->get(i); j++) {
				s += x->get(i)->str();
			}
			arr->set(i, new StringType(s));
		}

		return arr;
	}

	inline NDArray<StringType*>* capitalize(NDArray<StringType*>* x) {
		NDArray<StringType*>* out = x->emptyLike();

		for (int i = 0; i < out->size(); i++) {
			const char* st = x->get(i)->c_str();
			int code = toupper(st[0]);
			std::string str = "" + code;

			for (int j = 1; j < x->get(j)->str().size(); j++) {
				str += st[j];
			}

			out->set(i, new StringType(str));
		}

		return out;
	}

	inline NDArray<StringType*>* replace(NDArray<StringType*>* a, char oldChar, std::string newChar) {
		NDArray<StringType*>* out = a->emptyLike();

		for (int i = 0; i < out->size(); i++) {
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

		for (int i = 0; i < sep->size(); i++) {
			std::string str = sep->get(i)->str() + seq->get(i)->str();

			out->set(i, new StringType(str));
		}

		return out;
	}

	inline NDArray<StringType*>* lower(NDArray<StringType*>* x) {
		NDArray<StringType*>* out = x->emptyLike();

		for (int i = 0; i < out->size(); i++) {
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

		for (int i = 0; i < a->size(); i++) {
			std::string str = a->get(i)->str();
			size_t size = a->get(i)->str().size();

			for (int j = 0; j < (size - width); j++) {
				str += fillChar;
			}

			out->set(i, new StringType(str));
		}

		return out;
	}

	inline NDArray<StringType*>* rJust(NDArray<StringType*>* a, int width, char fillChar = ' ') {
		NDArray<StringType*>* out = a->emptyLike();

		for (int i = 0; i < a->size(); i++) {
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

	inline NDArray<StringType*>* strip(NDArray<StringType*>* a, std::string chars) {
		NDArray<StringType*>* out = a->emptyLike();

		for (int i = 0; i < a->size(); i++) {
			std::string str = "";
			std::string pre = a->get(i)->str();

			for (int j = 0; j < pre.size(); j++) {
				char c = pre.at(j);

				int k = 0;
				for (; k < chars.size(); k++) {
					if (chars.at(k) == c) {
						break;
					}
				}

				if (k == chars.size()) {
					str += c;
				}
			}

			out->set(i, new StringType(str));
		}

		return out;
	}

	inline NDArray<StringType*>* lStrip(NDArray<StringType*>* a, std::string chars) {
		NDArray<StringType*>* out = a->emptyLike();

		for (int i = 0; i < a->size(); i++) {
			std::string pre = a->get(i)->str();

			int j = 0;

			for (; j < pre.size(); j++) {
				char c = pre.at(j);

				int k = 0;
				bool found = false;
				for (; k < chars.size(); k++) {
					if (chars.at(k) == c) {
						found = true;
						break;
					}
				}

				if (!found) {
					break;
				}
			}

			out->set(i, new StringType(pre.substr(j, pre.size() - j)));
		}

		return out;
	}

	inline NDArray<StringType*>* rStrip(NDArray<StringType*>* a, std::string chars) {
		NDArray<StringType*>* out = a->emptyLike();

		for (size_t i = 0; i < a->size(); i++) {
			std::string pre = a->get(i)->str();

			size_t j = pre.size();

			for (; j >= 0; j--) {
				char c = pre.at(j);

				size_t k = 0;
				bool found = false;
				for (; k < chars.size(); k++) {
					if (chars.at(k) == c) {
						found = true;
						break;
					}
				}

				if (!found) {
					break;
				}
			}

			out->set(i, new StringType(pre.substr(0, j)));
		}

		return out;
	}

	inline NDArray<StringType*>* upper(NDArray<StringType*>* x) {
		NDArray<StringType*>* out = x->emptyLike();

		for (int i = 0; i < out->size(); i++) {
			std::string str = "";

			for (int j = 0; j < x->get(j)->str().size(); j++) {
				str += toupper(x->get(j)->str().at(j));
			}

			out->set(i, new StringType(str));
		}

		return out;
	}

	inline NDArray<StringType*>** partition(NDArray<StringType*>* a, std::string sep) {
		NDArray<StringType*>** out = (NDArray<StringType*>**)malloc(sizeof(NDArray<StringType*>*) * a->size() * 2);

		NDArray<StringType*>* before = a->emptyLike();
		NDArray<StringType*>* after = a->emptyLike();

		for (int i = 0; i < a->size(); i++) {
			bool passed = false;
			std::string str = a->get(i)->str();
			std::string bef = "";
			std::string aft = "";
			int index = 0;
			while (!passed) {
				if (str.at(index) == sep.at(0)) {
					int j = 1;
					index++;
					for (; j < sep.size(); j++) {
						if (str.at(index++) != sep.at(j)) {
							break;
						}
					}

					passed = j == sep.size();
				}
				else {
					bef += str.at(index);
					index++;
				}
			}

			before->set(i, new StringType(bef));

			for (; index < str.size(); index++) {
				aft += str.at(index);
			}

			after->set(i, new StringType(aft));
		}

		out[0] = before;
		out[1] = after;

		return out;
	}

	inline Vector<bool>* equal(Vector<StringType*>* a, Vector<StringType*>* b) {
		a->assertMatchShape(b->getShape());
		Vector<bool>* out = new Vector<bool>(a->size());

		for (int i = 0; i < out->size(); i++) {
			out->set(i, a->get(i)->str() == b->get(i)->str());
		}

		return out;
	}

	inline Matrix<bool>* equal(Matrix<StringType*>* a, Matrix<StringType*>* b) {
		a->assertMatchShape(b->getShape());
		Matrix<bool>* out = new Matrix<bool>(a->getHeight(), a->getWidth());

		for (int i = 0; i < out->size(); i++) {
			out->set(i, a->get(i)->str() == b->get(i)->str());
		}

		return out;
	}

	inline Vector<bool>* notEqual(Vector<StringType*>* a, Vector<StringType*>* b) {
		a->assertMatchShape(b->getShape());
		Vector<bool>* out = new Vector<bool>(a->size());

		for (int i = 0; i < out->size(); i++) {
			out->set(i, a->get(i)->str() != b->get(i)->str());
		}

		return out;
	}

	inline Matrix<bool>* notEqual(Matrix<StringType*>* a, Matrix<StringType*>* b) {
		a->assertMatchShape(b->getShape());
		Matrix<bool>* out = new Matrix<bool>(a->getHeight(), a->getWidth());

		for (int i = 0; i < out->size(); i++) {
			out->set(i, a->get(i)->str() != b->get(i)->str());
		}

		return out;
	}

	inline NDArray<int>* count(NDArray<StringType*>* a, string sub, int start = 0, size_t end = -1) {
		NDArray<int>* out = (new Vector<int>(a->size()))->reshape(a->getShape());

		for (int i = 0; i < a->size(); i++) {
			string str = a->get(i)->str();
			if (end == -1) {
				end = str.length();
			}

			for (int j = start; j < end; i++) {
				int idx = 0;
				while (idx < str.length() && j < end && sub[idx] == str[j]) {
					idx++;
					j++;
				}
				if (idx == sub.length()) {
					out->set(i, out->get(i) + 1);
				}
			}
		}

		return out;
	}

	inline NDArray<bool>* startsWith(NDArray<StringType*>* a, string prefix, int start = 0, int end = -1) {
		NDArray<bool>* out = (new Vector<bool>(a->size()))->reshape(a->getShape());
		end = end == -1 ? prefix.length() : end;

		for (int i = 0; i < a->size(); i++) {
			string str = a->get(i)->str();
			if (str.length() >= prefix.length()) {
				if (str.substr(start, end) == prefix) {
					out->set(i, true);
				}
			}
		}

		return out;
	}

	inline NDArray<bool>* endsWith(NDArray<StringType*>* a, string suffix, int start = -1, size_t end = -1) {
		NDArray<bool>* out = (new Vector<bool>(a->size()))->reshape(a->getShape());
		end = end == -1 ? suffix.length() : end;

		for (int i = 0; i < a->size(); i++) {
			string str = a->get(i)->str();
			size_t st = start == -1 ? str.length() - suffix.length() : start;
			if (str.length() >= suffix.length()) {
				if (str.substr(st, end) == suffix) {
					out->set(i, true);
				}
			}
		}

		return out;
	}

	inline NDArray<int>* find(NDArray<StringType*>* a, string sub, int start = 0, int end = -1) {
		NDArray<int>* out = (new Vector<int>(a->size()))->reshape(a->getShape());

		for (int i = 0; i < a->size(); i++) {
			string str = a->get(i)->str();
			if (end == -1) {
				end = str.length();
			}

			for (int j = start; j < end; i++) {
				int idx = 0;
				while (idx < str.length() && j < end && sub[idx] == str[j]) {
					idx++;
					j++;
				}
				if (idx == sub.length()) {
					out->set(i, j - idx);
				}
			}
		}

		return out;
	}

	inline NDArray<bool>* isNumeric(NDArray<StringType*>* a) {
		NDArray<bool>* out = (new Vector<bool>(a->size()))->reshape(a->getShape());

		for (int i = 0; i < a->size(); i++) {
			string str = a->get(i)->str();
			out->set(i, true);
			for (char c : str) {
				if (!isdigit(c)) {
					out->set(i, false);
					break;
				}
			}
		}

		return out;
	}

	inline NDArray<bool>* isUpper(NDArray<StringType*>* a) {
		NDArray<bool>* out = (new Vector<bool>(a->size()))->reshape(a->getShape());

		for (int i = 0; i < a->size(); i++) {
			string str = a->get(i)->str();
			out->set(i, true);
			for (char c : str) {
				if (!isupper(c)) {
					out->set(i, false);
					break;
				}
			}
		}

		return out;
	}

	inline NDArray<bool>* isLower(NDArray<StringType*>* a) {
		NDArray<bool>* out = (new Vector<bool>(a->size()))->reshape(a->getShape());

		for (int i = 0; i < a->size(); i++) {
			string str = a->get(i)->str();
			out->set(i, true);
			for (char c : str) {
				if (!islower(c)) {
					out->set(i, false);
					break;
				}
			}
		}

		return out;
	}
}