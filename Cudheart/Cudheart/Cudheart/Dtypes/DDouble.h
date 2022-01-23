#pragma once

#include "Dtype.h"

class DDouble : public Dtype {
public:
	void* get(void* arr, size_t i) override;
	string toString(void* arr, size_t i) override;
	string getName() override;
	void* copy(void* arr, int size) override;
	int getSize() override;
	void set(void* arr, size_t i, void* value) override;
	void* empty(int size) override;
};