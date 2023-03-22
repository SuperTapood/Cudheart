#include "NumpyAPI.cuh"

namespace Numpy {
	string createArray(string arr, string oName) {
		ostringstream os;
		os << oName << " = np.array(" << arr << ")\n";
		return os.str();
	}

	string createArray(string arr, string shape, string oName) {
		string out = createArray(arr, oName);
		out += oName + " = " + oName + ".reshape(" + shape + ")\n";
		return out;
	}

	string reshape(string name, string newShape, string oName) {
		ostringstream os;
		os << oName << " = " << name << ".reshape(" << newShape << ")\n";
		return os.str();
	}

	string empty(string shape, string oName) {
		ostringstream os;
		os << oName << " = np.empty(" << shape << ")\n";
		return os.str();
	}

	string full(string shape, string value, string oName) {
		return oName + " = np.full(" + shape + ", " + value + ")";
	}

	string T(string name, string oName) {
		ostringstream os;
		os << oName << " = " << name << ".T\n";
		return os.str();
	}

	string rot90(string name, int k, string oName) {
		return oName + " = np.rot90(" + name + ", k = " + to_string(k) + ")\n";
	}

	string augment(string a, string b, int axis, string oName) {
		return oName + " = np.concatenate((" + a + ", " + b + "), axis=" + to_string(axis) + ")\n";
	}

	string fromFile(string file, char sep, string dtype, string oName) {
		return oName + " = np.fromfile('" + file + "', sep='" + sep + "', dtype=" + dtype + ")\n";
	}

	string fromFunction(string funcName, string shape, string dtype, string oName) {
		return oName + " = np.fromfunction(" + funcName + ", " + shape + ", dtype=" + dtype + ")\n";
	}

	string arange(string start, string stop, string step, string dtype, string oName) {
		return oName + " = np.arange(" + start + ", " + stop + ", " + step + ", dtype=" + dtype + ")\n";
	}

	string linspace(string start, string stop, string num, string endpoint, string dtype, string oName) {
		return oName + " = np.linspace(" + start + ", " + stop + ", " + num + ", " + endpoint + ", dtype=" + dtype + ")\n";
	}

	string logspace(string start, string stop, string num, string endpoint, string base, string dtype, string oName) {
		return oName + " = np.logspace(" + start + ", " + stop + ", " + num + ", " + endpoint + ", dtype=" + dtype + ", base=" + base + ")\n";
	}

	string geomspace(string start, string stop, string num, string endpoint, string dtype, string oName) {
		return oName + " = np.geomspace(" + start + ", " + stop + ", " + num + ", " + endpoint + ", dtype=" + dtype + ")\n";
	}

	string eye(string N, string M, string k, string dtype, string oName) {
		return oName + " = np.eye(" + N + ", " + M + ", " + k + ", dtype=" + dtype + ")\n";
	}

	string meshgrid(string a, string b, string oName) {
		return oName + " = np.meshgrid(" + a + ", " + b + ")\n";
	}

	string append(string a, string b, string axis, string oName) {
		return oName + " = np.append(" + a + ", " + b + ", axis = " + axis + ")\n";
	}

	string concatenate(string a, string b, string oName) {
		return oName + " = np.concatenate((" + a + ", " + b + "))\n";
	}

	string diag(string v, string k, string oName) {
		return oName + " = np.diag(" + v + ", " + k + ")\n";
	}

	string diagflat(string v, string k, string oName) {
		return oName + " = np.diagflat(" + v + ", " + k + ")\n";
	}

	string split(string ary, string nums, string oName) {
		return oName + " = np.split(" + ary + ", " + nums + ")\n";
	}

	string tile(string a, string reps, string oName) {
		return oName + " = np.tile(" + a + ", " + reps + ")\n";
	}

	string tile(string a, string hReps, string wReps, string oName) {
		return oName + " = np.tile(" + a + ", (" + hReps + ", " + wReps + "))\n";
	}

	string remove(string a, string index, string oName) {
		return oName + " = np.delete(" + a + ", " + index + ")\n";
	}

	string remove(string a, string index, string axis, string oName) {
		return oName + " = np.delete(" + a + ", " + index + ", axis=" + axis + ")\n";
	}

	string trim_zeros(string filt, string trim, string oName) {
		return oName + " = np.trim_zeros(" + filt + ", " + trim + ")\n";
	}

	string unique(string ar, string oName) {
		return oName + " = np.unique(" + ar + ", True, True, True)\n";
	}

	string tri(string N, string M, string k, string dtype, string oName) {
		return oName + " = np.tri(" + N + ", " + M + ", " + k + ", dtype=" + dtype + ")\n";
	}

	string tril(string M, string k, string oName) {
		return oName + " = np.tril(" + M + ", " + k + ")\n";
	}

	string triu(string M, string k, string oName) {
		return oName + " = np.triu(" + M + ", " + k + ")\n";
	}

	string vander(string x, string N, string increasing, string oName) {
		return oName + " = np.vander(" + x + ", " + N + ", " + increasing + ")\n";
	}

	string argmax(string a, string oName) {
		return oName + " = np.argmax(" + a + ")\n";
	}

	string argmin(string a, string oName) {
		return oName + " = np.argmin(" + a + ")\n";
	}

	string nonzero(string a, string oName) {
		return oName + " = np.nonzero(" + a + ")\n";
	}

	string argwhere(string a, string oName) {
		return oName + " = np.argwhere(" + a + ")\n";
	}

	string flatnonzero(string a, string oName) {
		return oName + " = np.flatnonzero(" + a + ")\n";
	}

	string where(string condition, string x, string y, string oName) {
		return oName + " = np.where(" + condition + ", " + x + ", " + y + ")\n";
	}

	string searchsorted(string a, string v, string side, string sorter, string oName) {
		return oName + " = np.searchsorted(" + a + ", " + v + ", " + side + ", " + sorter + ")\n";
	}

	string extract(string condition, string arr, string oName) {
		return oName + " = np.extract(" + condition + ", " + arr + ")\n";
	}

	string count_nonzero(string a, string oName) {
		return oName + " = np.count_nonzero(" + a + ")\n";
	}

	string sort(string a, string kind, string oName) {
		return oName + " = np.sort(" + a + ", kind=" + kind + ")\n";
	}

	string argsort(string a, string kind, string oName) {
		return oName + " = np.argsort(" + a + ", kind=" + kind + ")\n";
	}

	string partition(string a, string kth, string oName) {
		return oName + " = np.partition(" + a + ", kth=" + kth + ")\n";
	}

	string argpartition(string a, string kth, string oName) {
		return oName + " = np.argpartition(" + a + ", kth=" + kth + ")\n";
	}

	string all(string arr, string oName) {
		return oName + " = np.all(" + arr + ")\n";
	}

	string any(string arr, string oName) {
		return oName + " = np.any(" + arr + ")\n";
	}

	string logicalAnd(string a, string b, string oName) {
		return oName + " = np.logical_and(" + a + ", " + b + ")\n";
	}

	string logicalOr(string a, string b, string oName) {
		return oName + " = np.logical_or(" + a + ", " + b + ")\n";
	}

	string logicalNot(string arr, string oName) {
		return oName + " = np.logical_not(" + arr + ")\n";
	}

	string logicalXor(string a, string b, string oName) {
		return oName + " = np.logical_xor(" + a + ", " + b + ")\n";
	}

	string isclose(string a, string b, string rtol, string atol, string oName) {
		return oName + " = np.isclose(" + a + ", " + b + ")\n";
	}

	string allclose(string a, string b, string rtol, string atol, string oName) {
		return oName + " = np.allclose(" + a + ", " + b + ")\n";
	}

	string equals(string a, string b, string oName) {
		return oName + " = np.equal(" + a + ", " + b + ")\n";
	}

	string greater(string a, string b, string oName) {
		return oName + " = np.greater(" + a + ", " + b + ")\n";
	}

	string greaterEquals(string a, string b, string oName) {
		return oName + " = np.greater_equal(" + a + ", " + b + ")\n";
	}

	string less(string a, string b, string oName) {
		return oName + " = np.less(" + a + ", " + b + ")\n";
	}

	string lessEqual(string a, string b, string oName) {
		return oName + " = np.less_equal(" + a + ", " + b + ")\n";
	}

	string maximum(string a, string b, string oName) {
		return oName + " = np.maximum(" + a + ", " + b + ")\n";
	}

	string amax(string x, string oName) {
		return oName + " = np.amax(" + x + ")\n";
	}

	string minimum(string a, string b, string oName) {
		return oName + " = np.minimum(" + a + ", " + b + ")\n";
	}

	string amin(string x, string oName) {
		return oName + " = np.amin(" + x + ")\n";
	}

	string cubeRoot(string arr, string oName) {
		return oName + " = np.cbrt(" + arr + ")\n";
	}

	string square(string base, string oName) {
		return oName + " = np.square(" + base + ")\n";
	}

	string squareRoot(string arr, string oName) {
		return oName + " = np.sqrt(" + arr + ")\n";
	}

	string power(string base, string po, string oName) {
		return oName + " = np.power(" + base + ", " + po + ")\n";
	}

	string around(string arr, string decimals, string oName) {
		return oName + " = np.around(" + arr + ", decimals=" + decimals + ")\n";
	}

	string rint(string arr, string oName) {
		return oName + " = np.rint(" + arr + ")\n";
	}

	string fix(string x, string oName) {
		return oName + " = np.fix(" + x + ")\n";
	}

	string floor(string x, string oName) {
		return oName + " = np.floor(" + x + ")\n";
	}

	string trunc(string x, string oName) {
		return oName + " = np.trunc(" + x + ")\n";
	}

	string ceil(string x, string oName) {
		return oName + " = np.ceil(" + x + ")\n";
	}

	string prod(string x, string oName) {
		return oName + " = np.prod(" + x + ")\n";
	}

	string sum(string x, string oName) {
		return oName + " = np.sum(" + x + ")\n";
	}

	string cumProd(string x, string oName) {
		return oName + " = np.cumprod(" + x + ")\n";
	}

	string cumSum(string x, string oName) {
		return oName + " = np.cumsum(" + x + ")\n";
	}

	string signBit(string x, string oName) {
		return oName + " = np.signbit(" + x + ")\n";
	}

	string copySign(string a, string b, string oName) {
		return oName + " = np.copysign(" + a + ", " + b + ")\n";
	}

	string abs(string x, string oName) {
		return oName + " = np.absolute(" + x + ")\n";
	}

	string lcm(string a, string b, string oName) {
		return oName + " = np.lcm(" + a + ", " + b + ")\n";
	}

	string gcd(string a, string b, string oName) {
		return oName + " = np.gcd(" + a + ", " + b + ")\n";
	}

	string add(string a, string b, string oName) {
		return oName + " = np.add(" + a + ", " + b + ")\n";
	}

	string subtract(string a, string b, string oName) {
		return oName + " = np.subtract(" + a + ", " + b + ")\n";
	}

	string multiply(string a, string b, string oName) {
		return oName + " = np.multiply(" + a + ", " + b + ")\n";
	}

	string divide(string a, string b, string oName) {
		return oName + " = np.divide(" + a + ", " + b + ")\n";
	}

	string floorDivide(string a, string b, string oName) {
		return oName + " = np.floor_divide(" + a + ", " + b + ")\n";
	}

	string mod(string a, string b, string oName) {
		return oName + " = np.mod(" + a + ", " + b + ")\n";
	}

	string divMod(string a, string b, string oName) {
		return oName + " = np.divmod(" + a + ", " + b + ")\n";
	}

	string reciprocal(string x, string oName) {
		return oName + " = np.reciprocal(" + x + ")\n";
	}

	string positive(string x, string oName) {
		return oName + " = np.positive(" + x + ")\n";
	}

	string negative(string x, string oName) {
		return oName + " = np.negative(" + x + ")\n";
	}

	string sign(string x, string oName) {
		return oName + " = np.sign(" + x + ")\n";
	}

	string heaviside(string a, string b, string oName) {
		return oName + " = np.heaviside(" + a + ", " + b + ")\n";
	}

	string bitwiseAnd(string a, string b, string oName) {
		return oName + " = np.bitwise_and(" + a + ", " + b + ")\n";
	}

	string bitwiseOr(string a, string b, string oName) {
		return oName + " = np.bitwise_or(" + a + ", " + b + ")\n";
	}

	string bitwiseXor(string a, string b, string oName) {
		return oName + " = np.bitwise_xor(" + a + ", " + b + ")\n";
	}

	string bitwiseLeftShift(string a, string b, string oName) {
		return oName + " = np.left_shift(" + a + ", " + b + ")\n";
	}

	string bitwiseRightShift(string a, string b, string oName) {
		return oName + " = np.right_shift(" + a + ", " + b + ")\n";
	}

	string bitwiseNot(string x, string oName) {
		return oName + " = np.bitwise_not(" + x + ")\n";
	}

	string angle(string z, string deg, string oName) {
		return oName + " = np.angle(" + z + ", " + deg + ")\n";
	}

	string real(string val, string oName) {
		return oName + " = np.real(" + val + ")\n";
	}

	string imag(string val, string oName) {
		return oName + " = np.imag(" + val + ")\n";
	}

	string conj(string x, string oName) {
		return oName + " = np.conj(" + x + ")\n";
	}

	string ln(string x, string oName) {
		return oName + " = np.log(" + x + ")\n";
	}

	string loga2(string x, string oName) {
		return oName + " = np.log2(" + x + ")\n";
	}

	string logan(string x, string n, string oName) {
		// why
		return oName + " = np.log(" + x + ") / np.log(" + n + ")\n";
	}

	string loga10(string x, string oName) {
		return oName + " = np.log10(" + x + ")\n";
	}

	string expo(string x, string oName) {
		return oName + " = np.exp(" + x + ")\n";
	}

	string expom1(string x, string oName) {
		return oName + " = np.expm1(" + x + ")\n";
	}

	string expo2(string x, string oName) {
		return oName + " = np.exp2(" + x + ")\n";
	}

	string logaddexp(string a, string b, string oName) {
		return oName + " = np.logaddexp(" + a + ", " + b + ")\n";
	}

	string logaddexp2(string a, string b, string oName) {
		return oName + " = np.logaddexp2(" + a + ", " + b + ")\n";
	}

	string dot(string a, string b, string oName) {
		return oName + " = np.dot(" + a + ", " + b + ")\n";
	}

	string inner(string a, string b, string oName) {
		return oName + " = np.inner(" + a + ", " + b + ")\n";
	}

	string outer(string a, string b, string oName) {
		return oName + " = np.outer(" + a + ", " + b + ")\n";
	}

	string det(string a, string oName) {
		return oName + " = np.linalg.det(" + a + ")\n";
	}

	string trace(string a, string offset, string oName) {
		return oName + " = np.trace(" + a + ", " + offset + ")\n";
	}

	string solve(string a, string b, string oName) {
		return oName + " = np.linalg.solve(" + a + ", " + b + ")\n";
	}

	string norm(string a, string oName) {
		return oName + " = np.norm(" + a + ")\n";
	}

	string eig(string a, string oName) {
		return oName + " = np.linalg.eig(" + a + ")\n";
	}

	string eigvals(string a, string oName) {
		return oName + " = np.linalg.eigvals(" + a + ")\n";
	}

	string roots(string p, string oName) {
		return oName + " = np.roots(" + p + ")\n";
	}

	string inv(string a, string oName) {
		return oName + " = np.linalg.inv(" + a + ")\n";
	}

	string convolve(string a, string b, string oName) {
		return oName + " = np.convolve(" + a + ", " + b + ")\n";
	}

	string clip(string arr, string min, string max, string oName) {
		return oName + " = np.clip(" + arr + ", " + min + ", " + max + ")\n";
	}

	string ptp(string a, string oName) {
		return oName + " = np.ptp(" + a + ")\n";
	}

	string percentile(string a, string q, string oName) {
		return oName + " = np.percentile(" + a + ", " + q + ")\n";
	}

	string quantile(string a, string q, string oName) {
		return oName + " = np.quantile(" + a + ", " + q + ")\n";
	}

	string median(string a, string oName) {
		return oName + " = np.median(" + a + ")\n";
	}

	string average(string a, string weights, string oName) {
		return oName + " = np.average(" + a + ", weights=" + weights + ")\n";
	}

	string mean(string a, string oName) {
		return oName + " = np.mean(" + a + ")\n";
	}

	string std(string a, string oName) {
		return oName + " = np.std(" + a + ")\n";
	}

	string var(string a, string oName) {
		return oName + " = np.var(" + a + ")\n";
	}

	string cov(string m, string rowvar, string oName) {
		return oName + " = np.cov(" + m + ", rowvar=" + rowvar + ")\n";
	}

	string corrcoef(string x, string rowvar, string oName) {
		return oName + " = np.corrcoef(" + x + ", rowvar=" + rowvar + ")\n";
	}

	string histogram(string a, string bins, string low, string high, string oName) {
		return oName + " = np.histogram(" + a + ", " + bins + ", (" + low + ", " + high + ")\n";
	}

	string histogram2d(string x, string y, string binX, string binY, string lowX, string highX, string lowY, string highY, string oName) {
		return oName + " = np.histogram2d(" + x + ", " + y + ", bins=(" + binX + ", " + binY + "), " + ", range=([" + lowX + ", " + highX + "], [" + lowY + ", " + highY + "])\n";
	}

	string bincount(string x, string oName) {
		return oName + " = np.bincount(" + x + ")\n";
	}

	string digitize(string x, string bins, string right, string oName) {
		return oName + " = np.digitize(" + x + ", " + bins + ", " + right + ")\n";
	}
}