#pragma once


namespace Cudheart::Test {
	void test();
	void exceptionTest();

	class ObjTest {
	public:
		virtual void creationTest() = 0;
	};

	class VectorTest : public ObjTest {
	public:
		void creationTest();
	};
}

