#pragma once

#include <vector>
#include <string>
#include "../Arrays/Array.h"

using std::vector;
using std::string;


class GarbageCollector {
private:
	static vector<string> deletedVoids;
	static vector<string> deletedDtypes;
	static vector<string> deletedShapes;
	static GarbageCollector* collector;
public:
	GarbageCollector() {
		deletedVoids = vector<string>();
		deletedDtypes = vector<string>();
		deletedShapes = vector<string>();
	}
	static void create();
	static GarbageCollector* get();
	void deletePtrs(Array obj);
	static void destroy();
	static void tryDelete(void* ptr, vector<string> vec);
};