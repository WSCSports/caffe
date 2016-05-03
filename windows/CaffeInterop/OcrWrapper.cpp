// This is the main DLL file.

#include "stdafx.h"

#include "OcrWrapper.h"


using namespace System;
using namespace msclr::interop;

namespace CaffeInterop {
	OcrWrapper::OcrWrapper(System::String^ modelFile, System::String^ weightsFile)
	{
		_inner = new caffeWrapper();
		std::string strModelFile = marshal_as<std::string>(modelFile);
		std::string strWeightsFile = marshal_as<std::string>(weightsFile);

		_inner->Init(strModelFile, strWeightsFile);

	}
	OcrWrapper::~OcrWrapper()
	{
		_inner->Kill();
		delete _inner;
	}
	static bool PairCompare(const std::pair<float, int>& lhs,
		const std::pair<float, int>& rhs) {
		return lhs.first > rhs.first;
	}
	static std::vector<int> Argmax(const std::vector<float>& v, int N) {
		std::vector<std::pair<float, int> > pairs;
		for (size_t i = 0; i < v.size(); ++i)
			pairs.push_back(std::make_pair(v[i], static_cast<int>(i)));
		std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

		std::vector<int> result;
		for (int i = 0; i < N; ++i)
			result.push_back(pairs[i].second);
		return result;
	}
	System::String^ OcrWrapper::GetValue(IntPtr img, int rows, int cols, int type)
	{

		cv::Mat y = cv::Mat(rows, cols, type, img.ToPointer());
		cv::imwrite("c:\\temp\\miracle.png", y);
		vector<float> predictions = _inner->Forward(y, 1, 2, 3);
		System::String^ result = "" + Argmax(predictions, 1)[0];
		return result;
	}

}