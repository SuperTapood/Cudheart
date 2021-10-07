#pragma once



// maybe change to a struct?

class Shape {
public:
	Shape(int shape[]);
	~Shape();
	const int at(int idx);
public:
	__int64 length;
	int size = 0;
private:
	int shape[4];
};