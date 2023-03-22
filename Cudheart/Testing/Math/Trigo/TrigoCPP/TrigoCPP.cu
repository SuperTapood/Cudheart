#include "TrigoCPP.cuh"

namespace Cudheart::Testing::Math::CPP::Trigo {
	void testTrigo() {
		sin();
		cos();
		tan();
		cot();
		arcsin();
		arccos();
		arctan();
		arccot();
		hypot();
		deg2rad();
		rad2deg();
		sinc();
		sinh();
		cosh();
		tanh();
		arcsinh();
		arccosh();
		arctanh();
	}

	void sin() {
		string cmd;

		auto x = new Vector<double>({ 0.5, 0.33, 0.69, 0.42, 0.502, 0.8, 0.87, 0.3 });
		auto res = Cudheart::CPP::Math::Trigo::sin(x);

		cmd = Numpy::createArray(x->toString(), "x");
		cmd += Numpy::sin("x", "res");

		Testing::submit("Trigo::sin(Vector<double>)", cmd, res->toString());
	}

	void cos() {
		string cmd;

		auto x = new Vector<double>({ 0.5, 0.33, 0.69, 0.42, 0.502, 0.8, 0.87, 0.3 });
		auto res = Cudheart::CPP::Math::Trigo::cos(x);

		cmd = Numpy::createArray(x->toString(), "x");
		cmd += Numpy::cos("x", "res");

		Testing::submit("Trigo::cos(Vector<double>)", cmd, res->toString());
	}

	void tan() {
		string cmd;

		auto x = new Vector<double>({ 0.5, 0.33, 0.69, 0.42, 0.502, 0.8, 0.87, 0.3 });
		auto res = Cudheart::CPP::Math::Trigo::tan(x);

		cmd = Numpy::createArray(x->toString(), "x");
		cmd += Numpy::tan("x", "res");

		Testing::submit("Trigo::tan(Vector<double>)", cmd, res->toString());
	}

	void cot() {
		string cmd;

		auto x = new Vector<double>({ 0.5, 0.33, 0.69, 0.42, 0.502, 0.8, 0.87, 0.3 });
		auto res = Cudheart::CPP::Math::Trigo::cot(x);

		cmd = Numpy::createArray(x->toString(), "x");
		cmd += Numpy::cot("x", "res");

		Testing::submit("Trigo::cot(Vector<double>)", cmd, res->toString());
	}

	void arcsin() {
		string cmd;

		auto x = new Vector<double>({ 0.5, 0.33, 0.69, 0.42, 0.502, 0.8, 0.87, 0.3 });
		auto res = Cudheart::CPP::Math::Trigo::arcsin(x);

		cmd = Numpy::createArray(x->toString(), "x");
		cmd += Numpy::arcsin("x", "res");

		Testing::submit("Trigo::arcsin(Vector<double>)", cmd, res->toString());
	}

	void arccos() {
		string cmd;

		auto x = new Vector<double>({ 0.5, 0.33, 0.69, 0.42, 0.502, 0.8, 0.87, 0.3 });
		auto res = Cudheart::CPP::Math::Trigo::arccos(x);

		cmd = Numpy::createArray(x->toString(), "x");
		cmd += Numpy::arccos("x", "res");

		Testing::submit("Trigo::arccos(Vector<double>)", cmd, res->toString());
	}

	void arctan() {
		string cmd;

		auto x = new Vector<double>({ 0.5, 0.33, 0.69, 0.42, 0.502, 0.8, 0.87, 0.3 });
		auto res = Cudheart::CPP::Math::Trigo::arctan(x);

		cmd = Numpy::createArray(x->toString(), "x");
		cmd += Numpy::arctan("x", "res");

		Testing::submit("Trigo::arctan(Vector<double>)", cmd, res->toString());
	}

	void arccot() {
		string cmd;

		auto x = new Vector<double>({ 0.5, 0.33, 0.69, 0.42, 0.502, 0.8, 0.87, 0.3 });
		auto res = Cudheart::CPP::Math::Trigo::arccot(x);

		cmd = Numpy::createArray(x->toString(), "x");
		cmd += Numpy::arccot("x", "res");

		Testing::submit("Trigo::arccot(Vector<double>)", cmd, res->toString());
	}

	void hypot() {
		string cmd;

		auto x = new Vector<double>({ 0.5, 0.33, 0.69, 0.42, 0.502, 0.8, 0.87, 0.3 });
		auto y = new Vector<double>({ 0.250, 0.39, 0.420, 0.69, 0.11, 0.77, 0.66, 0.13 });
		auto res = Cudheart::CPP::Math::Trigo::hypot(x, y);

		cmd = Numpy::createArray(x->toString(), "x");
		cmd = Numpy::createArray(y->toString(), "y");
		cmd += Numpy::hypot("x", "y", "res");

		Testing::submit("Trigo::hypot(Vector<double>)", cmd, res->toString());
	}

	void deg2rad() {
		string cmd;

		auto x = new Vector<double>({ 0.5, 0.33, 0.69, 0.42, 0.502, 0.8, 0.87, 0.3 });
		auto res = Cudheart::CPP::Math::Trigo::deg2rad(x);

		cmd = Numpy::createArray(x->toString(), "x");
		cmd += Numpy::deg2rad("x", "res");

		Testing::submit("Trigo::deg2rad(Vector<double>)", cmd, res->toString());
	}

	void rad2deg() {
		string cmd;

		auto x = new Vector<double>({ 0.5, 0.33, 0.69, 0.42, 0.502, 0.8, 0.87, 0.3 });
		auto res = Cudheart::CPP::Math::Trigo::rad2deg(x);

		cmd = Numpy::createArray(x->toString(), "x");
		cmd += Numpy::rad2deg("x", "res");

		Testing::submit("Trigo::rad2deg(Vector<double>)", cmd, res->toString());
	}

	void sinc() {
		string cmd;

		auto x = new Vector<double>({ 0.5, 0.33, 0.69, 0.42, 0.502, 0.8, 0.87, 0.3 });
		auto res = Cudheart::CPP::Math::Trigo::sinc(x);

		cmd = Numpy::createArray(x->toString(), "x");
		cmd += Numpy::sinc("x", "res");

		Testing::submit("Trigo::sinc(Vector<double>)", cmd, res->toString());
	}

	void sinh() {
		string cmd;

		auto x = new Vector<double>({ 0.5, 0.33, 0.69, 0.42, 0.502, 0.8, 0.87, 0.3 });
		auto res = Cudheart::CPP::Math::Trigo::sinh(x);

		cmd = Numpy::createArray(x->toString(), "x");
		cmd += Numpy::sinh("x", "res");

		Testing::submit("Trigo::sinh(Vector<double>)", cmd, res->toString());
	}

	void cosh() {
		string cmd;

		auto x = new Vector<double>({ 0.5, 0.33, 0.69, 0.42, 0.502, 0.8, 0.87, 0.3 });
		auto res = Cudheart::CPP::Math::Trigo::cosh(x);

		cmd = Numpy::createArray(x->toString(), "x");
		cmd += Numpy::cosh("x", "res");

		Testing::submit("Trigo::cosh(Vector<double>)", cmd, res->toString());
	}

	void tanh() {
		string cmd;

		auto x = new Vector<double>({ 0.5, 0.33, 0.69, 0.42, 0.502, 0.8, 0.87, 0.3 });
		auto res = Cudheart::CPP::Math::Trigo::tanh(x);

		cmd = Numpy::createArray(x->toString(), "x");
		cmd += Numpy::tanh("x", "res");

		Testing::submit("Trigo::tanh(Vector<double>)", cmd, res->toString());
	}

	void arcsinh() {
		string cmd;

		auto x = new Vector<double>({ 0.5, 0.33, 0.69, 0.42, 0.502, 0.8, 0.87, 0.3 });
		auto res = Cudheart::CPP::Math::Trigo::arcsinh(x);

		cmd = Numpy::createArray(x->toString(), "x");
		cmd += Numpy::arcsinh("x", "res");

		Testing::submit("Trigo::arcsinh(Vector<double>)", cmd, res->toString());
	}

	void arccosh() {
		string cmd;

		auto x = new Vector<double>({ 5, 7, 6.9, 4.2, 5.02, 8, 8.7, 3 });
		auto res = Cudheart::CPP::Math::Trigo::arccosh(x);

		cmd = Numpy::createArray(x->toString(), "x");
		cmd += Numpy::arccosh("x", "res");

		Testing::submit("Trigo::arccosh(Vector<double>)", cmd, res->toString());
	}

	void arctanh() {
		string cmd;

		auto x = new Vector<double>({ 0.5, 0.33, 0.69, 0.42, 0.502, 0.8, 0.87, 0.3 });
		auto res = Cudheart::CPP::Math::Trigo::arctanh(x);

		cmd = Numpy::createArray(x->toString(), "x");
		cmd += Numpy::arctanh("x", "res");

		Testing::submit("Trigo::arctanh(Vector<double>)", cmd, res->toString());
	}
}