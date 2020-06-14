using OpenCvSharp;
using OpenCvSharp.Dnn;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace VideoFrameAnalyzer
{
    public class Yolo3DnnDetector
    {
        string[] _outNames;

        private const string DataRoot = @"C:\Users\raimo\OneDrive\Repos\Cognitive-Samples-VideoFrameAnalysis\YoloData";

        //YOLOv3
        //https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg
        private const string Cfg = DataRoot + @"\yolov3.cfg";

        //https://pjreddie.com/media/files/yolov3.weights
        private const string Weight = DataRoot + @"\yolov3.weights";

        //https://github.com/pjreddie/darknet/blob/master/data/coco.names
        private const string Names = DataRoot + @"\coco.names";

        //random assign color to each label
        private static readonly Scalar[] Colors = Enumerable.Repeat(false, 80).Select(x => Scalar.RandomColor()).ToArray();

        //get labels from coco.names
        private static readonly string[] Labels = File.ReadAllLines(Names).ToArray();

        private OpenCvSharp.Dnn.Net nnet;
        private Mat[] outs;

        public Yolo3DnnDetector()
        {
            nnet = OpenCvSharp.Dnn.CvDnn.ReadNetFromDarknet(Cfg, Weight);
            //nnet.SetPreferableBackend(Net.Backend.INFERENCE_ENGINE);
            //nnet.SetPreferableTarget(Net.Target.CPU);
            _outNames = nnet.GetUnconnectedOutLayersNames();
            outs = Enumerable.Repeat(_, _outNames.Length).Select(_ => new Mat()).ToArray();
        }

        public DnnDetectedObject[] ClassifyObjects(Mat image, Rect boxToAnalyze)
        {
            if (image == null)
            {
                throw new ArgumentNullException($"{nameof(image)}");
            }

            using var blob = CvDnn.BlobFromImage(image, 1.0 / 255, new Size(320, 320), crop: false);
            nnet.SetInput(blob);

            //forward model
            nnet.Forward(outs, _outNames);

            const float threshold = 0.5f;       //for confidence 
            const float nmsThreshold = 0.3f;    //threshold for nms

            return ExtractYolo3Results(outs, image, threshold, nmsThreshold, 1.0f);
        }

        private DnnDetectedObject[] ExtractYolo3Results(IEnumerable<Mat> output, Mat image, float threshold, float nmsThreshold, float scaleFactor, bool nms = true)
        {
            //for nms
            var classIds = new List<int>();
            var confidences = new List<float>();
            var probabilities = new List<float>();
            var boxes = new List<Rect2d>();

            var w = image.Width;
            var h = image.Height;
            /*
             YOLO3 COCO trainval output
             0 1 : center                    2 3 : w/h
             4 : confidence                  5 ~ 84 : class probability 
            */
            const int prefix = 5;   //skip 0~4

            foreach (var prob in output)
            {
                for (var i = 0; i < prob.Rows; i++)
                {
                    var confidence = prob.At<float>(i, 4);

                    //Filter out bogus results of > 100% confidence
                    if (confidence > threshold && confidence <= 1.0)
                    {
                        //get classes probability
                        Cv2.MinMaxLoc(prob.Row(i).ColRange(prefix, prob.Cols), out _, out OpenCvSharp.Point max);
                        var classes = max.X;
                        var probability = prob.At<float>(i, classes + prefix);

                        if (probability > threshold) //more accuracy, you can cancel it
                        {
                            //get center and width/height
                            var centerX = prob.At<float>(i, 0) * w * scaleFactor;
                            var centerY = prob.At<float>(i, 1) * h * scaleFactor;
                            var width = prob.At<float>(i, 2) * w * scaleFactor;
                            var height = prob.At<float>(i, 3) * h * scaleFactor;

                            float X = Math.Max(0, centerX - (width / 2.0f));
                            float Y = Math.Max(0, centerY - (height / 2.0f));

                            //put data to list for NMSBoxes
                            classIds.Add(classes);
                            confidences.Add(confidence);
                            probabilities.Add(probability);
                            boxes.Add(new Rect2d(X, Y, width, height));
                        }
                    }
                }
            }

            int[] indices;

            if (!nms)
            {
                //using non-maximum suppression to reduce overlapping low confidence box
                indices = Enumerable.Range(0, boxes.Count).ToArray();
            }
            else
            {
                CvDnn.NMSBoxes(boxes, confidences, threshold, nmsThreshold, out indices);
                ConcurrentLogger.WriteLine($"NMSBoxes drop {confidences.Count - indices.Length} overlapping result.");
            }

            var result = new List<DnnDetectedObject>();

            foreach (var i in indices)
            {
                var box = boxes[i];

                var detection = new DnnDetectedObject()
                {
                    Index = classIds[i],
                    Label = Labels[classIds[i]],
                    Color = Colors[classIds[i]],
                    Probability = probabilities[i],
                    BoundingBox = box
                };
                result.Add(detection);
            }

            return result.ToArray();
        }

    }
}
