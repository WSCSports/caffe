// ClassLibrary1.h

#pragma once
#include <string>

#include <caffe\forward.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <msclr\marshal_cppstd.h>

using namespace System;

namespace CaffeInterop
{
	public ref class OcrWrapper
	{

	public:
		OcrWrapper(System::String^ modelFile, System::String^ weightsFile);
		~OcrWrapper();

		System::String^ GetValue(IntPtr img, int rows, int cols, int type);
	private:
		caffeWrapper* _inner;
	};
}
