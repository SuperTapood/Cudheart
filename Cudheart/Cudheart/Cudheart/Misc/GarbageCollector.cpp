#include "GarbageCollector.h"

void GarbageCollector::create()
{
	if (GarbageCollector::collector == nullptr) {
		GarbageCollector::collector = new GarbageCollector();
	}
}

void GarbageCollector::deletePtrs(Array obj)
{
	tryDelete(obj.arr, deletedVoids);
	tryDelete(obj.dtype, deletedDtypes);
	tryDelete(obj.shape, deletedShapes);
}

void GarbageCollector::destroy()
{
	delete collector;
}

void GarbageCollector::tryDelete(void* ptr, vector<string> vec)
{
	ostringstream os;
	os << ptr;
	string id = os.str();
	for (string elem : vec) {
		if (id == elem) {
			return;
		}
	}
	vec.push_back(id);
	delete ptr;
}

GarbageCollector* GarbageCollector::get() {
	create();
	return collector;
}
