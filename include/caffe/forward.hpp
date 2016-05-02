//#include <iostream>
#include <string>
//#include <vector>
//#include <algorithm>
////
//#define USE_EIGEN
////#define CPU_ONLY
////
#include "caffe/caffe.hpp"
//#include "caffe/blob.hpp"
//#include "caffe/util/io.hpp"
//#include "proto/caffe.pb.h"

#include <opencv2/core/core.hpp>

using namespace caffe;

using namespace std;
using namespace cv;

class __declspec(dllexport) caffeWrapper {
	Net<float> *net;
	bool isInitialized;

public:
	caffeWrapper();
	~caffeWrapper();
	
	bool __stdcall Init(const string &inDefinition, const string &inWeights);
	vector<float> Forward(const cv::Mat& cv_img, const unsigned &inWidth, const unsigned &inHeight, const unsigned &inChannels);
	void __stdcall Kill();
};

extern "C" __declspec(dllexport) caffeWrapper* __cdecl newCaffe();