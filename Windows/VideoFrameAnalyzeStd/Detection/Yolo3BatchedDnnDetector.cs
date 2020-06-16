﻿using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using OpenCvSharp;
using OpenCvSharp.Dnn;
using System;
using System.Collections.Generic;
using System.Configuration;
using System.IO;
using System.Linq;
using VideoFrameAnalyzeStd.Detection;

namespace VideoFrameAnalyzer
{
    public class Yolo3BatchedDnnDetector : IBatchedDnnDetector
    {
        private readonly ILogger _logger;
        private readonly IConfiguration _configuration;

        //YOLOv3
        //https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg
        private string Cfg;

        //https://pjreddie.com/media/files/yolov3.weights
        private string Weight;

        //https://github.com/pjreddie/darknet/blob/master/data/coco.names
        private string Names;

        private readonly Scalar[] Colors;
        private readonly string[] Labels;
        private readonly OpenCvSharp.Dnn.Net nnet;
        private Mat[] outs;
        private readonly string[] _outNames;

        public Yolo3BatchedDnnDetector(ILogger<IDnnDetector> logger, IConfiguration configuration)
        {
            _logger = logger;
            _configuration = configuration;
            ReadConfig(configuration);

            //random assign color to each label
            Labels = File.ReadAllLines(Names).ToArray();

            //get labels from coco.names
            Colors = Enumerable.Repeat(false, Labels.Length).Select(x => Scalar.RandomColor()).ToArray();

            nnet = OpenCvSharp.Dnn.CvDnn.ReadNetFromDarknet(Cfg, Weight);
            //nnet.SetPreferableBackend(Net.Backend.INFERENCE_ENGINE);
            //nnet.SetPreferableTarget(Net.Target.CPU);
            _outNames = nnet.GetUnconnectedOutLayersNames();
            outs = Enumerable.Repeat(false, _outNames.Length).Select(_ => new Mat()).ToArray();
        }

        private void ReadConfig(IConfiguration configuration)
        {
            var options = new Yolo3Options();
            configuration.GetSection(Yolo3Options.Yolo3).Bind(options);

            Cfg = Path.Combine(options.RootPath, options.ConfigFile);
            Weight = Path.Combine(options.RootPath, options.WeightsFile);
            Names = Path.Combine(options.RootPath, options.NamesFile);
        }

        public DnnDetectedObject[][] ClassifyObjects(IEnumerable<Mat> images)
        {
            if (images == null)
            {
                throw new ArgumentNullException($"{nameof(images)}");
            }

            using var blob = CvDnn.BlobFromImages(images, 1.0 / 255, new Size(320, 320), crop: false);
            nnet.SetInput(blob);

            //forward model
            nnet.Forward(outs, _outNames);

            const float threshold = 0.5f;       //for confidence 
            const float nmsThreshold = 0.3f;    //threshold for nms

            return ExtractYolo3BatchedResults(outs, images, threshold, nmsThreshold);
        }


        private DnnDetectedObject[][] ExtractYolo3BatchedResults(IEnumerable<Mat> output, IEnumerable<Mat> image, float threshold, float nmsThreshold, bool nms = true)
        {
            var inputImages = image.ToList();

            DnnDetectedObject[][] results = new DnnDetectedObject[inputImages.Count][];

            var classIds = new List<int>();
            var confidences = new List<float>();
            var probabilities = new List<float>();
            var boxes = new List<Rect2d>();

            for (int inputIndex = 0; inputIndex < inputImages.Count; inputIndex++)
            {
                //for nms
                classIds.Clear();
                confidences.Clear();
                probabilities.Clear();
                boxes.Clear();

                var w = inputImages[inputIndex].Width;
                var h = inputImages[inputIndex].Height;
                /*
                 YOLO3 COCO trainval output
                 0 1 : center                    2 3 : w/h
                 4 : confidence                  5 ~ 84 : class probability 
                */
                const int prefix = 5;   //skip 0~4

                foreach (var prob in output)
                {
                    for (var i = 0; i < prob.Size(1); i++)
                    {
                        var confidence = prob.At<float>(inputIndex, i, 4);

                        //Filter out bogus results of > 100% confidence
                        if (confidence > threshold && confidence <= 1.0)
                        {
                            var colRange = new Range(prefix, prob.Size(2) - 1);

                            ////get classes probability
                            //Cv2.MinMaxLoc((classProbsArray, out _, out OpenCvSharp.Point max);

                            var classes = GetMaxProbabilityClassIndex(prob, inputIndex, i, colRange);
                            if (classes == -1)
                            {
                                continue;
                            }
                            //var classes = max.X;
                            var probability = prob.At<float>(inputIndex, i, classes);

                            if (probability > threshold) //more accuracy, you can cancel it
                            {
                                //get center and width/height
                                var centerX = prob.At<float>(inputIndex, i, 0) * w;
                                var centerY = prob.At<float>(inputIndex, i, 1) * h;
                                var width = prob.At<float>(inputIndex, i, 2) * w;
                                var height = prob.At<float>(inputIndex, i, 3) * h;

                                float X = Math.Max(0, centerX - (width / 2.0f));
                                float Y = Math.Max(0, centerY - (height / 2.0f));

                                //put data to list for NMSBoxes
                                classIds.Add(classes - prefix);
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
                    _logger.LogInformation($"NMSBoxes drop {confidences.Count - indices.Length} overlapping result.");
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
                results[inputIndex] = result.ToArray();
            }

            return results;
        }

        private int GetMaxProbabilityClassIndex(Mat inputMatrix, int dim0Index, int dim1Index, Range dim2Range)
        {
            int numItems = dim2Range.End - dim2Range.Start + 1;
            int dim2MaxIndex = -1;
            float dim2MaxValue = 0;

            for(int dim2Index = dim2Range.Start; dim2Index <= dim2Range.End; dim2Index++)
            {
                var curValue = inputMatrix.At<float>(dim0Index, dim1Index, dim2Index);
                if(curValue > dim2MaxValue)
                {
                    dim2MaxIndex = dim2Index;
                    dim2MaxValue = curValue;
                }
            }
            return dim2MaxIndex;
        }
    }
}
