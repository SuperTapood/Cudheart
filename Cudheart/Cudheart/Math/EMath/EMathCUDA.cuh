#pragma once

#include "../../Arrays/Arrays.cuh"
#include <cmath>
#include "../Constants.cuh"
#include "EMathKernels.cuh"

using Cudheart::NDArrays::Vector;
using Cudheart::NDArrays::Matrix;
using namespace Cudheart::Exceptions;
using Cudheart::VectorOps::empty;
using Cudheart::VectorOps::emptyLike;
using Cudheart::MatrixOps::fromVector;
using namespace Cudheart::Kernels::Math::EMath;

namespace Cudheart::CUDA::Math::EMath {
	template <typename T>
	Vector<T>* squareRoot(Vector<T>* vec) {

		int len = vec->getSize();
		Vector<T>* output = empty<T>(len);
		ContainerAB<T>* con = vec->getContainerAB(output);

		kernelSqrt << <1, len >> > (con->devB, con->devA);

		delete con;

		return output;
	}

	template <typename T>
	Matrix<T>* squareRoot(Matrix<T>* mat) {
		Vector<T>* flat = mat->flatten();

		Vector<T>* out = squareRoot(flat);

		delete flat;

		return fromVector(out, mat->getWidth(), mat->getHeight(), true);
	}

	template <typename T>
	Vector<T>* loga(Vector<T>* vec) {
		int len = vec->getSize();
		Vector<T>* output = empty<T>(len);

		ContainerAB<T>* con = vec->getContainerAB(output);

		kernelLog << <1, len >> > (con->devB, con->devA);

		delete con;

		return output;
	}

	template <typename T>
	Matrix<T>* loga(Matrix<T>* mat) {
		Vector<T>* flat = mat->flatten();

		Vector<T>* out = loga(flat);

		delete flat;

		return fromVector(out, mat->getWidth(), mat->getHeight(), true);
	}

	template <typename T>
	Vector<T>* loga2(Vector<T>* vec) {
		int len = vec->getSize();
		Vector<T>* output = empty<T>(len);

		ContainerAB<T>* con = vec->getContainerAB(output);

		kernelLog2 << <1, len >> > (con->devB, con->devA);

		delete con;

		return output;
	}

	template <typename T>
	Matrix<T>* loga2(Matrix<T>* mat) {
		Vector<T>* flat = mat->flatten();

		Vector<T>* out = loga2(flat);

		delete flat;

		return fromVector(out, mat->getWidth(), mat->getHeight(), true);
	}

	template <typename T>
	Vector<T>* logan(Vector<T>* vec, T n) {
		int len = vec->getSize();
		Vector<T>* output = empty<T>(len);

		ContainerAB<T>* con = vec->getContainerAB(output);

		kernelLogN << < 1, len >> > (con->devB, con->devA, n);

		delete con;

		return output;
	}

	template <typename T>
	Matrix<T>* logan(Matrix<T>* mat, T n) {
		Vector<T>* flat = mat->flatten();

		Vector<T>* out = logan(flat, n);

		delete flat;

		return fromVector(out, mat->getWidth(), mat->getHeight(), true);
	}

	template <typename T>
	Vector<T>* loga10(Vector<T>* vec) {
		int len = vec->getSize();
		Vector<T>* output = empty<T>(len);

		ContainerAB<T>* con = vec->getContainerAB(output);

		kernelLog10 << <1, len >> > (con->devB, con->devA);

		delete con;

		return output;
	}

	template <typename T>
	Matrix<T>* loga10(Matrix<T>* mat) {
		Vector<T>* flat = mat->flatten();

		Vector<T>* out = loga10(flat);

		delete flat;

		return fromVector(out, mat->getWidth(), mat->getHeight(), true);
	}

	template <typename T>
	Vector<T>* power(Vector<T>* base, T po) {
		int len = base->getSize();
		Vector<T>* output = empty<T>(len);

		ContainerAB<T>* con = base->getContainerAB(output);

		kernelPower << <1, len >> > (con->devB, con->devA, po);

		delete con;

		return output;
	}

	template <typename T>
	Vector<T>* power(Vector<T>* base, Vector<T>* po) {
		base->assertMatchSize(power);
		int len = base->getSize();
		Vector<T>* output = empty<T>(len);

		ContainerAB<T>* con = base->getContainerABC(po, output);

		kernelPower << <1, len >> > (con->devC, con->devA, con->devB);

		delete con;

		return output;
	}

	template <typename T>
	Matrix<T>* power(Matrix<T>* base, T po) {
		Vector<T>* flat = base->flatten();

		Vector<T>* out = power(flat, power);

		delete flat;

		return fromVector(out, base->getWidth(), base->getHeight());
	}

	template <typename T>
	Matrix<T>* power(Matrix<T>* base, Matrix<T>* high) {
		base->assertMatchSize(power);
		Vector<T>* b = base->flatten();
		Vector<T>* p = high->flatten();

		Vector<T>* output = power(b, p);

		delete b;
		delete p;

		return fromVector(output, base->getWidth(), base->getHeight());
	}

	template <typename T>
	Matrix<T>* power(Matrix<T>* base, Vector<T>* po) {
		if (base->getHeight() != po->getSize()) {
			Cudheart::Exceptions::ShapeMismatchException(base->getHeight(), po->getSize()).raise();
		}

		Vector<T>* arr = base->toVectorArray();
		Vector<T>* a = (Vector<T>*)malloc(sizeof(Vector<T>) * base->getHeight());

		for (int i = 0; i < base->getHeight(); i++) {
			Vector<T> v = arr[i];
			a[i] = power(v, po->get(i));
		}

		delete arr;

		return MatrixOps::fromVectorArray(a, base->getHeight());
	}

	template <typename T>
	Vector<T>* arccos(Vector<T>* vec) {
		Vector<T>* out = emptyLike(vec);

		ContainerAB<T>* con = vec->getContainerAB(out);

		kernelArccos << <1, vec->getSize() >> > (con->devB, con->devA);

		delete con;

		return out;
	}

	template <typename T>
	Matrix<T>* arccos(Matrix<T>* mat) {
		Vector<T>* flat = mat->flatten();

		Vector<T>* out = arccos(flat);

		delete flat;

		return fromVector(out, mat->getWidth(), mat->getHeight(), true);
	}

	template <typename T>
	Vector<T>* arcsin(Vector<T>* vec) {
		Vector<T>* out = emptyLike(vec);

		ContainerAB<T>* con = vec->getContainerAB(out);

		kernelArcsin << <1, vec->getSize() >> > (con->devB, con->devA);

		delete con;

		return out;
	}

	template <typename T>
	Matrix<T>* arcsin(Matrix<T>* mat) {
		Vector<T>* flat = mat->flatten();

		Vector<T>* out = arccos(flat);

		delete flat;

		return fromVector(out, mat->getWidth(), mat->getHeight(), true);
	}

	template <typename T>
	Vector<T>* arctan(Vector<T>* vec) {
		Vector<T>* out = emptyLike(vec);

		ContainerAB<T>* con = vec->getContainerAB(out);

		kernelArctan << <1, vec->getSize() >> > (con->devB, con->devA);

		delete con;

		return out;
	}

	template <typename T>
	Matrix<T>* arctan(Matrix<T>* mat) {
		Vector<T>* flat = mat->flatten();

		Vector<T>* out = arctan(flat);

		delete flat;

		return fromVector(out, mat->getWidth(), mat->getHeight(), true);
	}

	template <typename T>
	Vector<T>* arccot(Vector<T>* vec) {
		Vector<T>* out = emptyLike(vec);

		ContainerAB<T>* con = vec->getContainerAB(out);

		kernelArccot << <1, vec->getSize() >> > (con->devB, con->devA);

		delete con;

		return out;
	}

	template <typename T>
	Matrix<T>* arccot(Matrix<T>* mat) {
		Vector<T>* flat = mat->flatten();

		Vector<T>* out = arccot(flat);

		delete flat;

		return fromVector(out, mat->getWidth(), mat->getHeight(), true);
	}
}