#include "caffe/forward.hpp"
//#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
caffeWrapper::caffeWrapper() {
	isInitialized = false;
	cout << "Caffe constructor, init = " << isInitialized << endl;
}

caffeWrapper::~caffeWrapper() {
	cout << "Caffe destructor" << endl;
	Kill();
}

bool  caffeWrapper::Init(const string &inDefinition, const string &inWeights) {
	if (isInitialized) {
		cout << "~Re-initializing net(" << isInitialized << ") " << inDefinition.c_str() << endl;
	}

	Kill();

	Caffe::set_mode(Caffe::CPU);

	NetParameter net_definition;
	ReadProtoFromTextFile(inDefinition, &net_definition);

	net = new Net<float>(inDefinition, caffe::Phase::TEST);

	cout << "Num Layers Net:" << net->layers().size() << endl;

	net->CopyTrainedLayersFrom(inWeights);


	isInitialized = true;
	return net->layers().size();
}
void printMat(const std::string name, const cv::Mat& img)
{
	std::ofstream file;
	file.open("C:\\Projects\\SportsOCRAll\\model\\vector." + name);
	for (int y = 0; y < img.rows; y++)
	{
		for (int x = 0; x < img.cols; x++)
		{
			if (img.type() == CV_8UC3)
			{
				cv::Vec3b color = img.at<cv::Vec3b>(y, x);
				file << color << '\n';
			}
			else
			{
				uint8_t color = img.at<uint8_t>(y, x);
				file << std::to_string(color) << '\n';
			}


		}
	}
}

static cv::Mat transposeImg(const cv::Mat& img)
{
	int size[2];
	size[0] = img.channels();
	size[1] = img.rows;
	cv::Mat newMat = cv::Mat(img.cols, size, CV_8U);
	for (int channel = 0; channel < img.channels(); channel++)
	{
		for (int y = 0; y < img.rows; y++)
		{
			for (int x = 0; x < img.cols; x++)
			{
				newMat.at<uchar>(channel, y, x) = img.at<cv::Vec3b>(x, y)[channel];
			}
		}
	}
	return newMat;

}
static cv::Mat convertImageToVector(const cv::Mat& img)
{
	int num_channels = 1;
	int height = 1;
	int width = 3 * 48 * 32;

	cv::Mat bigImg;
	cv::resize(img, bigImg, cv::Size(48, 32), 0, 0, CV_INTER_LINEAR);

#if DEBUG
	printMat("original", img);
	printMat("big", bigImg);
#endif

	cv::Mat flatMat = cv::Mat(num_channels, width, CV_8UC1);
	int idx = 0;
	//for (int channel = bigImg.channels() - 1; channel >= 0; channel--)
	for (int channel = 0; channel < bigImg.channels(); channel++)
	{
		for (int y = 0; y < bigImg.rows; y++)
		{
			for (int x = 0; x < bigImg.cols; x++)
			{
				cv::Vec3b color = bigImg.at<cv::Vec3b>(y, x);
				//flatMat.at<uchar>(y*bigImg.cols*bigImg.channels() + x*bigImg.channels() + channel) = color[channel];
				flatMat.at<uchar>(idx) = color[channel];
				idx++;
			}
		}
	}
#if DEBUG
	printMat("flat", flatMat);
#endif
	return flatMat;
}

void CVMatToDatumTranspose(const cv::Mat& cv_img, Datum* datum) {
	CHECK(cv_img.depth() == CV_8U) << "Image data type must be unsigned byte";



	datum->set_channels(cv_img.cols);
	datum->set_height(cv_img.rows);
	datum->set_width(cv_img.channels());
	datum->clear_data();
	datum->clear_float_data();
	datum->set_encoded(false);
	int datum_channels = datum->channels();
	int datum_height = datum->height();
	int datum_width = datum->width();
	int datum_size = datum_channels * datum_height * datum_width;
	std::string buffer(datum_size, ' ');

	for (int w = 0; w < cv_img.channels(); w++)
	{
		for (int h = 0; h < cv_img.rows; h++)
		{
			for (int c = 0; c < cv_img.cols; c++)
			{
				int datum_index = (c * datum_height + h) * datum_width + w;
				buffer[datum_index] = static_cast<char>(cv_img.at<cv::Vec3b>(h, c)[w]);
			}
		}
	}
	datum->set_data(buffer);
}
vector<float> caffeWrapper::Forward(const cv::Mat& cv_img, const unsigned &inWidth, const unsigned &inHeight, const unsigned &inChannels) {

	cv::Mat bigImg;
	cv::resize(cv_img, bigImg, cv::Size(48, 32), 0, 0, CV_INTER_LINEAR);
	//cv::Mat img = transposeImg(bigImg);
	int num_channels = bigImg.channels();
	int width = bigImg.cols;
	int height = bigImg.rows;

	Blob<float>* blob = new Blob<float>(1, bigImg.channels(), bigImg.rows, bigImg.cols);
	BlobProto blob_proto;

	Datum datum;

	/*if (num_channels == 1 || num_channels == 3) {*/
		CVMatToDatum(bigImg, &datum);
	//}
	//else {
	//	cout << "Caffe input image: #invalid num channels!" << endl;

	//	return vector<float>();
	//}

	blob_proto.set_num(1);
	blob_proto.set_channels(datum.channels());
	blob_proto.set_height(datum.height());
	blob_proto.set_width(datum.width());
	const int data_size = datum.channels() * datum.height() * datum.width();
	int size_in_datum = std::max<int>(datum.data().size(),
		datum.float_data_size());
	for (int i = 0; i < size_in_datum; i++) {
		blob_proto.add_data(0.);
	}
	const string& data = datum.data();
	if (data.size() != 0) {
		for (int i = 0; i < size_in_datum; i++) {
			blob_proto.set_data(i, blob_proto.data(i) + ((float)((uint8_t)data[i]) / 255.0));
		}
	}

	//set data into blob
	blob->FromProto(blob_proto);

	//fill the vector
	vector<Blob<float>*> bottom;
	bottom.push_back(blob);

	float type = 0.0f;
	const vector<Blob<float>*>& result = net->Forward(bottom, &type);

	vector<float> output;

	for (unsigned f = 0; f < result[0]->count(); f++) {
		output.push_back(result[0]->cpu_data()[f]);
	}

	delete blob;

	return output;
}

void caffeWrapper::Kill() {
	if (isInitialized) {
		isInitialized = false;
		delete net;
	}
}

extern "C" __declspec(dllexport) caffeWrapper* __cdecl newCaffe()
{
	return new caffeWrapper;
}