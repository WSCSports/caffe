using OpenCvSharp.CPlusPlus;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace UsageExample
{
    class Program
    {
        static void Main(string[] args)
        {
            var x = Cv2.ImRead(@"C:\Users\WSC-User-6\odrive\OneDrive - Work\OldOCR\OCR\fc08c029-671d-428c-8053-89ad43e182c3.0014400-0021659-0028918.0014518.mp4\4.part.7.U.png");
            x.ImWrite(@"c:\temp\simple.png");
            var w = new Mat(x.Rows, x.Cols, x.Type(), x.Ptr());
            w.ImWrite(@"c:\temp\realsimple.png");
            var res = new CaffeInterop.OcrWrapper(@"C:\Projects\WSC\alternate\ZoomInCloud\SportsOCR\SportOCR3_lib\SportOCR\models\cnn_bn.caffe.proto", @"C:\Projects\WSC\alternate\ZoomInCloud\SportsOCR\SportOCR3_lib\SportOCR\models\cnn_bn.caffe.model" ).GetValue(x.Ptr(), x.Rows, x.Cols, x.Type().Value);
        }
    }
}
