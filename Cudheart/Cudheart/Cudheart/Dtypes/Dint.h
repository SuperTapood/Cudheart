#pragma once


#include "Dtype.h"

class DInt32 : public Dtype {
public:
	void* get(void* arr, size_t i) override;
	string toString(void* arr, size_t i) override;
	string getName() override;
};