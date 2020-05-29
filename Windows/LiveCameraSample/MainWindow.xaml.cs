// 
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license.
// 
// Microsoft Cognitive Services: http://www.microsoft.com/cognitive
// 
// Microsoft Cognitive Services Github:
// https://github.com/Microsoft/Cognitive
// 
// Copyright (c) Microsoft Corporation
// All rights reserved.
// 
// MIT License:
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
// 
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED ""AS IS"", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
// 

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Net;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using Microsoft.Azure.CognitiveServices.Vision.ComputerVision;
using Microsoft.Azure.CognitiveServices.Vision.Face;
using Microsoft.Azure.CognitiveServices.Vision.Face.Models;
using Newtonsoft.Json.Linq;
using OpenCvSharp;
using OpenCvSharp.Extensions;
using OpenCvSharp.Dnn;

using VideoFrameAnalyzer;
using FaceAPI = Microsoft.Azure.CognitiveServices.Vision.Face;
using Rect = OpenCvSharp.Rect;
using Size = OpenCvSharp.Size;
using VisionAPI = Microsoft.Azure.CognitiveServices.Vision.ComputerVision;
using System.IO;
using Microsoft.Azure.CognitiveServices.Vision.ComputerVision.Models;

namespace LiveCameraSample
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : System.Windows.Window, IDisposable
    {
        private FaceAPI.FaceClient _faceClient = null;
        private VisionAPI.ComputerVisionClient _visionClient = null;
        private readonly FrameGrabber<LiveCameraResult> _grabber;
        private static readonly ImageEncodingParam[] s_jpegParams = {
            new ImageEncodingParam(ImwriteFlags.JpegQuality, 60)
        };
        private readonly CascadeClassifier _localFaceDetector = new CascadeClassifier();
        private bool _fuseClientRemoteResults;
        private LiveCameraResult _latestResultsToDisplay = null;
        private AppMode _mode;
        private DateTime _startTime;

        public enum AppMode
        {
            Faces,
            Emotions,
            EmotionsWithClientFaceDetect,
            Tags,
            Celebrities
        }

        public MainWindow()
        {
            InitializeComponent();
            ServicePointManager.SecurityProtocol = SecurityProtocolType.Tls12;

            //nnet.SetPreferableBackend(Net.Backend.OPENCV);
            //nnet.SetPreferableTarget(Net.Target.CPU);

            // Create grabber. 
            _grabber = new FrameGrabber<LiveCameraResult>();

            // Set up a listener for when the client receives a new frame.
            _grabber.NewFrameProvided += (s, e) =>
            {
                if (_mode == AppMode.EmotionsWithClientFaceDetect)
                {
                    // Local face detection. 
                    var rects = _localFaceDetector.DetectMultiScale(e.Frame.Image);
                    // Attach faces to frame. 
                    e.Frame.UserData = rects;
                }

                // The callback may occur on a different thread, so we must use the
                // MainWindow.Dispatcher when manipulating the UI. 
                this.Dispatcher.BeginInvoke((Action)(() =>
                {
                    // Display the image in the left pane.
                    LeftImage.Source = e.Frame.Image.ToBitmapSource();

                    // If we're fusing client-side face detection with remote analysis, show the
                    // new frame now with the most recent analysis available. 
                    if (_fuseClientRemoteResults)
                    {
                        RightImage.Source = VisualizeResult(e.Frame);
                    }
                }));

                // See if auto-stop should be triggered. 
                if (Properties.Settings.Default.AutoStopEnabled && (DateTime.Now - _startTime) > Properties.Settings.Default.AutoStopTime)
                {
                    _grabber.StopProcessingAsync().GetAwaiter().GetResult();
                }
            };

            // Set up a listener for when the client receives a new result from an API call. 
            _grabber.NewResultAvailable += (s, e) =>
            {
                this.Dispatcher.BeginInvoke((Action)(() =>
                {
                    if (e.TimedOut)
                    {
                        MessageArea.Text = "API call timed out.";
                    }
                    else if (e.Exception != null)
                    {
                        string apiName = "";
                        string message = e.Exception.Message;
                        var faceEx = e.Exception as FaceAPI.Models.APIErrorException;
                        var visionEx = e.Exception as VisionAPI.Models.ComputerVisionErrorException;
                        if (faceEx != null)
                        {
                            apiName = "Face";
                            message = faceEx.Message;
                        }
                        else if (visionEx != null)
                        {
                            apiName = "Computer Vision";
                            message = visionEx.Message;
                        }
                        MessageArea.Text = string.Format("{0} API call failed on frame {1}. Exception: {2}", apiName, e.Frame.Metadata.Index, message);
                    }
                    else
                    {
                        _latestResultsToDisplay = e.Analysis;
                        var tempProvider = (e.Analysis as IProvideTempImage);

                        if (tempProvider != null)
                        {
                            var tempImage1 = tempProvider.TempImage1;
                            if (tempImage1 != null)
                            {
                                TempImage1.Source = tempImage1.ToBitmapSource();
                            }
                            var tempImage2 = tempProvider.TempImage2;
                            if (tempImage2 != null)
                            {
                                TempImage2.Source = tempImage2.ToBitmapSource();
                            }
                        }

                        // Display the image and visualization in the right pane. 
                        if (!_fuseClientRemoteResults)
                        {
                            RightImage.Source = VisualizeResult(e.Frame);
                        }
                    }
                }));
            };

            // Create local face detector. 
            _localFaceDetector.Load("Data/haarcascade_frontalface_alt2.xml");

            InitNeuralNet();
        }

        string[] _outNames;
        //private static string prott1 = @"C:\Users\Raimo\Downloads\MobileNetSSD_deploy.prototxt";
        //private static string prott2 = @"C:\Users\Raimo\Downloads\mobilenet_iter_73000.caffemodel";

        //private static string prott1 = @"C:\Users\Raimo\Downloads\mobilenet_yolov3_lite_deploy.prototxt";
        //private static string prott2 = @"C:\Users\Raimo\Downloads\mobilenet_yolov3_lite_deploy.caffemodel";
        //private OpenCvSharp.Dnn.Net nnet = OpenCvSharp.Dnn.CvDnn.ReadNetFromCaffe(prott1, prott2);


        //YOLOv3
        //https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg
        private const string Cfg = @"C:\Users\Raimo\Downloads\yolov3.cfg";

        //https://pjreddie.com/media/files/yolov3.weights
        private const string Weight = @"C:\Users\Raimo\Downloads\yolov3.weights";

        //https://github.com/pjreddie/darknet/blob/master/data/coco.names
        private const string Names = @"C:\Users\Raimo\Downloads\coco.names";

        //random assign color to each label
        private static readonly Scalar[] Colors = Enumerable.Repeat(false, 80).Select(x => Scalar.RandomColor()).ToArray();

        //get labels from coco.names
        private static readonly string[] Labels = File.ReadAllLines(Names).ToArray();

        private OpenCvSharp.Dnn.Net nnet;

        private void InitNeuralNet()
        {
            nnet = OpenCvSharp.Dnn.CvDnn.ReadNetFromDarknet(Cfg, Weight);
            _outNames = nnet.GetUnconnectedOutLayersNames();
        }

        /// <summary> Function which submits a frame to the Face API. </summary>
        /// <param name="frame"> The video frame to submit. </param>
        /// <returns> A <see cref="Task{LiveCameraResult}"/> representing the asynchronous API call,
        ///     and containing the faces returned by the API. </returns>
        private async Task<LiveCameraResult> FacesAnalysisFunction(VideoFrame frame)
        {
            // Encode image. 
            var jpg = frame.Image.ToMemoryStream(".jpg", s_jpegParams);
            // Submit image to API. 
            var attrs = new List<FaceAPI.Models.FaceAttributeType> {
                FaceAPI.Models.FaceAttributeType.Age,
                FaceAPI.Models.FaceAttributeType.Gender,
                FaceAPI.Models.FaceAttributeType.HeadPose
            };
            var faces = await _faceClient.Face.DetectWithStreamAsync(jpg, returnFaceAttributes: attrs);
            // Count the API call. 
            Properties.Settings.Default.FaceAPICallCount++;
            // Output. 
            return new LiveCameraResult { Faces = faces.ToArray() };
        }

        private Task<LiveCameraResult> OpenCVPeopleDetect(VideoFrame frame)
        {
            // Encode image. 
            var jpg = frame.Image; //.ToMemoryStream(".jpg", s_jpegParams);

            var img = jpg; //Cv2.ImDecode(ImRead(FilePath.Image.Asahiyama, ImreadModes.Color);

           // var detectImg = img.Resize(Size.Zero, 0.5, 0.5);

            var hog = new HOGDescriptor();
            hog.SetSVMDetector(HOGDescriptor.GetDefaultPeopleDetector());

            bool b = hog.CheckDetectorSize();
            //            Console.WriteLine("CheckDetectorSize: {0}", b);

            //            var watch = Stopwatch.StartNew();

            // run the detector with default parameters. to get a higher hit-rate
            // (and more false alarms, respectively), decrease the hitThreshold and
            // groupThreshold (set groupThreshold to 0 to turn off the grouping completely).
            //Rect[] found = hog.DetectMultiScale(img, 0, new Size(8, 8), new Size(24, 16), 1.05, 2);
            Rect[] found = hog.DetectMultiScale(img, 
                                                winStride: new Size(8, 8), 
                                                padding: new Size(8, 8), 
                                                scale: 1.05,
                                                groupThreshold:1);


            //watch.Stop();
            //Console.WriteLine("Detection time = {0}ms", watch.ElapsedMilliseconds);
            //Console.WriteLine("{0} region(s) found", found.Length);

            var faces = new List<DetectedFace>();

            foreach (Rect rect in found)
            {
                // the HOG detector returns slightly larger rectangles than the real objects.
                // so we slightly shrink the rectangles to get a nicer output.
                var r = new Rect
                {
                    X = rect.X + (int)Math.Round(rect.Width * 0.1),
                    Y = rect.Y + (int)Math.Round(rect.Height * 0.1),
                    Width = (int)Math.Round(rect.Width * 0.8),
                    Height = (int)Math.Round(rect.Height * 0.8)
                };
                //img.Rectangle(r.TopLeft, r.BottomRight, Scalar.Red, 3);

                var face = new DetectedFace()
                {
                    FaceAttributes = new FaceAttributes(),
                    FaceLandmarks = new FaceLandmarks(),
                    FaceRectangle = new FaceAPI.Models.FaceRectangle(r.Width, r.Height, r.X, r.Y),
                    RecognitionModel = "custom"
                };
                faces.Add(face);
            }

            //using (var window = new Window("people detector", WindowMode.Normal, img))
            //{
            //    window.SetProperty(WindowProperty.Fullscreen, 1);
            //    Cv2.WaitKey(0);
            //}

            return Task.FromResult(new LiveCameraResult { Faces = faces.ToArray() });
        }

        //        def non_max_suppression_fast(boxes, overlapThresh):
        //    #if there are no boxes, return an empty list
        //	if len(boxes) == 0:
        //		return []
        //#if the bounding boxes integers, convert them to floats --
        //# this is important since we'll be doing a bunch of divisions
        //	if boxes.dtype.kind == "i":
        //		boxes = boxes.astype("float")
        //# initialize the list of picked indexes	
        //	pick = []
        //# grab the coordinates of the bounding boxes
        //	x1 = boxes[:,0]
        //	y1 = boxes[:,1]
        //	x2 = boxes[:,2]
        //	y2 = boxes[:,3]
        //# compute the area of the bounding boxes and sort the bounding
        //# boxes by the bottom-right y-coordinate of the bounding box
        //	area = (x2 - x1 + 1) * (y2 - y1 + 1)
        //	idxs = np.argsort(y2)
        //# keep looping while some indexes still remain in the indexes
        //# list
        //	while len(idxs) > 0:
        //# grab the last index in the indexes list and add the
        //# index value to the list of picked indexes
        //		last = len(idxs) - 1
        //		i = idxs[last]
        //		pick.append(i)
        //# find the largest (x, y) coordinates for the start of
        //# the bounding box and the smallest (x, y) coordinates
        //# for the end of the bounding box
        //		xx1 = np.maximum(x1[i], x1[idxs[:last]])
        //		yy1 = np.maximum(y1[i], y1[idxs[:last]])
        //		xx2 = np.minimum(x2[i], x2[idxs[:last]])
        //		yy2 = np.minimum(y2[i], y2[idxs[:last]])
        //# compute the width and height of the bounding box
        //		w = np.maximum(0, xx2 - xx1 + 1)
        //		h = np.maximum(0, yy2 - yy1 + 1)
        //# compute the ratio of overlap
        //		overlap = (w * h) / area[idxs[:last]]
        //# delete all indexes from the index list that have
        //		idxs = np.delete(idxs, np.concatenate(([last],
        //			np.where(overlap > overlapThresh)[0])))
        //# return only the bounding boxes that were picked using the
        //# integer data type
        //	return boxes[pick].astype("int")

        private Mat _diffBaseFrame = null;
        private int _frameCounter = 0;
        private int _frameDiffGap = 100;
        private Mat _dilateElement = Cv2.GetStructuringElement(MorphShapes.Ellipse, new Size(10, 10));



        private Task<LiveCameraResult> OpenCVDiffContourPeopleDetect(VideoFrame frame)
        {
            var faces = new List<DetectedFace>();
            var result = new LiveCameraResult();

            try
            {             
                var image = frame.Image; //.ToMemoryStream(".jpg", s_jpegParams);
                if (image == null)
                {
                    return Task.FromResult(new LiveCameraResult { Faces = new DetectedFace[0] });
                }
                var frame2 = new Mat();

                var curWidth = image.Width;
                var factor = 1.0; // 500.0 / curWidth;

                //Cv2.Resize(image, frame2, Size.Zero, fx: factor);
                Cv2.CopyTo(image, frame2);

                _frameCounter++;
                _frameCounter %= _frameDiffGap;
                //if (_frameCounter > 1)
                {
                    //(H, W) = frame.shape[:2]
                    var gray = new Mat();
                    Cv2.CvtColor(frame2, gray, ColorConversionCodes.BGR2GRAY);
                    Cv2.GaussianBlur(gray, gray, new Size(21, 21), 0);

                    //if the first frame is None, initialize it
                    if (_diffBaseFrame == null || _frameCounter == 0)
                    {
                        _diffBaseFrame = gray;

                        result.Faces = faces.ToArray();
                        return Task.FromResult(result);
                    }

                    var frameDelta = new Mat();
                    // compute the absolute difference between the current frame and first frame
                    Cv2.Absdiff(_diffBaseFrame, gray, frameDelta);
                    var tresh = new Mat();
                    //Cv2.Threshold(frameDelta, tresh, 25, 255, ThresholdTypes.Binary);
                    Cv2.Threshold(frameDelta, tresh, 30, 255, ThresholdTypes.Binary|ThresholdTypes.Otsu);

                    
                    // dilate the thresholded image to fill in holes, then find contours on thresholded image
                    Cv2.Dilate(tresh, tresh, _dilateElement, iterations: 2);

                    result.TempImage1 = frameDelta;
                    result.TempImage2 = tresh;

                    Mat[] cnts;
                    Mat newTresh = new Mat();
                    Mat hierarchy = new Mat();
                    tresh.CopyTo(newTresh);
                    //cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                    Cv2.FindContours(newTresh, out cnts, hierarchy, RetrievalModes.External, 
                        ContourApproximationModes.ApproxTC89KCOS);


                    List<Rect> boxes = new List<Rect>();
                    List<double> areas = new List<double>();

                    //ClassifyObjects(image, r);


                    //loop over the contours identified
                    //contourcount = 0
                    foreach (var c in cnts)
                    {
                        //contourcount =  contourcount + 1
                        var ca = Cv2.ContourArea(c);

                        if(ca < 5000) //ignore small areas.
                        {
                            continue;
                        }
    
                        //compute the bounding box for the contour, draw it on the frame,
                        var r = Cv2.BoundingRect(c);
                        r.X = (int)(r.X * factor);
                        r.Y = (int)(r.Y * factor);
                        r.Width = (int)(r.Width * factor);
                        r.Height = (int)(r.Height * factor);
                        //(x, y, w, h) = cv2.boundingRect(c)
                        //initBB2 =(x,y,w,h)
                        boxes.Add(r);
                        areas.Add(ca);
                    }

                    if (boxes.Count > 0)
                    {
                        double maxArea = areas.Max();
                        var scores = areas.Select(a => (float)(a / maxArea));

                        int[] indices;
                        //OpenCvSharp.Dnn.CvDnn.NMSBoxes(boxes, scores, 0.3f, 0.4f, out indices);
                        indices = Enumerable.Range(0, boxes.Count()).ToArray();

                        foreach (int index in indices)
                        {
                            Rect r = boxes[index];

                            var face = new DetectedFace()
                            {
                                FaceAttributes = new FaceAttributes() { Age = areas[index] },
                                FaceLandmarks = new FaceLandmarks(),
                                FaceRectangle = new FaceAPI.Models.FaceRectangle(r.Width, r.Height, r.X, r.Y),
                                RecognitionModel = "custom"
                            };
                            faces.Add(face);
                        }
                    }
                }
            }
            catch(Exception ex)
            {
                int a = 1;
            }
            result.Faces = faces.ToArray();

            return Task.FromResult(result);
        }

        private Task<LiveCameraResult> OpenCVDNNYoloPeopleDetect(VideoFrame frame)
        {
            LiveCameraResult result;

            try
            {
                var image = frame.Image;
                if (image == null)
                {
                    return Task.FromResult(new LiveCameraResult { Faces = new DetectedFace[0] });
                }

                var watch = new Stopwatch();
                watch.Start();

                result = ClassifyObjects(image, Rect.Empty);

                watch.Stop();
                ConcurrentLogger.WriteLine($"Classifiy-objects ms:{watch.ElapsedMilliseconds}");
            }
            catch (Exception ex)
            {
                result = new LiveCameraResult();
                ConcurrentLogger.WriteLine($"Exception in analysis:{ex.Message}");
            }
            

            return Task.FromResult(result);
        }

        private LiveCameraResult ClassifyObjects(Mat image, Rect boxToAnalyze)
        {

            var blob = CvDnn.BlobFromImage(image, 1.0/255, new Size(320, 320), new Scalar(), crop: false);
            nnet.SetInput(blob);

            //create mats for output layer
            Mat[] outs = Enumerable.Repeat(false, _outNames.Length).Select( _ => new Mat()).ToArray();

            //forward model
            nnet.Forward(outs, _outNames);

            const float threshold = 0.5f;       //for confidence 
            const float nmsThreshold = 0.3f;    //threshold for nms

            return GetResult(outs, image, threshold, nmsThreshold, 1.0f);
        }

        private LiveCameraResult GetResult(IEnumerable<Mat> output, Mat image, float threshold, float nmsThreshold, float scaleFactor, bool nms = true)
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
                    if (confidence > threshold)
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

                            float X = centerX - (width / 2.0f);
                            float Y = centerY - (height / 2.0f);
                            //if (!nms)
                            //{
                            //    // draw result (if don't use NMSBoxes)
                            //    Draw(image, classes, confidence, probability, centerX, centerY, width, height);
                            //    continue;
                            //}

                            //put data to list for NMSBoxes
                            classIds.Add(classes);
                            confidences.Add(confidence);
                            probabilities.Add(probability);
                            boxes.Add(new Rect2d(X, Y, width, height));
                        }
                    }
                }
            }

            //if (!nms) return;

            int[] indices;
            //using non-maximum suppression to reduce overlapping low confidence box
            indices = Enumerable.Range(0, boxes.Count()).ToArray();
            //CvDnn.NMSBoxes(boxes, confidences, threshold, nmsThreshold, out indices);

            //ConcurrentLogger.WriteLine($"NMSBoxes drop {confidences.Count - indices.Length} overlapping result.");

            LiveCameraResult result = new LiveCameraResult();
            List<DetectedFace> faces = new List<DetectedFace>();
            List<ImageTag> tags = new List<ImageTag>(); 

            foreach (var i in indices)
            {
                var box = boxes[i];
                //Draw(image, classIds[i], confidences[i], probabilities[i], box.X, box.Y, box.Width, box.Height);

                var face = new DetectedFace()
                {
                    FaceAttributes = new FaceAttributes() { Age = classIds[i], Smile = probabilities[i] },
                    FaceLandmarks = new FaceLandmarks(),
                    FaceRectangle = new FaceAPI.Models.FaceRectangle((int)box.Width, (int)box.Height, (int)box.X, (int)box.Y),
                    RecognitionModel = "custom"
                };
                faces.Add(face);
                tags.Add(new ImageTag(Labels[classIds[i]], confidences[i]));
            }

            result.Faces = faces.ToArray();
            result.Tags = tags.ToArray();

            return result;
        }

        /// <summary> Function which submits a frame to the Emotion API. </summary>
        /// <param name="frame"> The video frame to submit. </param>
        /// <returns> A <see cref="Task{LiveCameraResult}"/> representing the asynchronous API call,
        ///     and containing the emotions returned by the API. </returns>
        private async Task<LiveCameraResult> EmotionAnalysisFunction(VideoFrame frame)
        {
            // Encode image. 
            var jpg = frame.Image.ToMemoryStream(".jpg", s_jpegParams);
            // Submit image to API. 
            FaceAPI.Models.DetectedFace[] faces = null;

            // See if we have local face detections for this image.
            var localFaces = (OpenCvSharp.Rect[])frame.UserData;
            if (localFaces == null || localFaces.Count() > 0)
            {
                // If localFaces is null, we're not performing local face detection.
                // Use Cognigitve Services to do the face detection.
                Properties.Settings.Default.FaceAPICallCount++;
                faces = (await _faceClient.Face.DetectWithStreamAsync(
                    jpg,
                    returnFaceId: false,
                    returnFaceLandmarks: false,
                    returnFaceAttributes: new FaceAPI.Models.FaceAttributeType[1] { FaceAPI.Models.FaceAttributeType.Emotion })).ToArray();
            }
            else
            {
                // Local face detection found no faces; don't call Cognitive Services.
                faces = new FaceAPI.Models.DetectedFace[0];
            }

            // Output. 
            return new LiveCameraResult
            {
                Faces = faces
            };
        }

        /// <summary> Function which submits a frame to the Computer Vision API for tagging. </summary>
        /// <param name="frame"> The video frame to submit. </param>
        /// <returns> A <see cref="Task{LiveCameraResult}"/> representing the asynchronous API call,
        ///     and containing the tags returned by the API. </returns>
        private async Task<LiveCameraResult> TaggingAnalysisFunction(VideoFrame frame)
        {
            // Encode image. 
            var jpg = frame.Image.ToMemoryStream(".jpg", s_jpegParams);
            // Submit image to API. 
            var tagResult = await _visionClient.TagImageInStreamAsync(jpg);
            // Count the API call. 
            Properties.Settings.Default.VisionAPICallCount++;
            // Output. 
            return new LiveCameraResult { Tags = tagResult.Tags.ToArray() };
        }

        /// <summary> Function which submits a frame to the Computer Vision API for celebrity
        ///     detection. </summary>
        /// <param name="frame"> The video frame to submit. </param>
        /// <returns> A <see cref="Task{LiveCameraResult}"/> representing the asynchronous API call,
        ///     and containing the celebrities returned by the API. </returns>
        private async Task<LiveCameraResult> CelebrityAnalysisFunction(VideoFrame frame)
        {
            // Encode image. 
            var jpg = frame.Image.ToMemoryStream(".jpg", s_jpegParams);
            // Submit image to API. 
            var domainModelResults = await _visionClient.AnalyzeImageByDomainInStreamAsync("celebrities", jpg);
            // Count the API call. 
            Properties.Settings.Default.VisionAPICallCount++;
            // Output. 
            var jobject = domainModelResults.Result as JObject;
            var celebs = jobject.ToObject<VisionAPI.Models.CelebrityResults>().Celebrities;
            return new LiveCameraResult
            {
                // Extract face rectangles from results. 
                Faces = celebs.Select(c => CreateFace(c.FaceRectangle)).ToArray(),
                // Extract celebrity names from results. 
                CelebrityNames = celebs.Select(c => c.Name).ToArray()
            };
        }

        private BitmapSource VisualizeResult(VideoFrame frame)
        {
            // Draw any results on top of the image. 
            BitmapSource visImage = frame.Image.ToBitmapSource();

            var result = _latestResultsToDisplay;

            if (result != null)
            {
                // See if we have local face detections for this image.
                var clientFaces = (OpenCvSharp.Rect[])frame.UserData;
                if (clientFaces != null && result.Faces != null)
                {
                    // If so, then the analysis results might be from an older frame. We need to match
                    // the client-side face detections (computed on this frame) with the analysis
                    // results (computed on the older frame) that we want to display. 
                    MatchAndReplaceFaceRectangles(result.Faces, clientFaces);
                }

                visImage = Visualization.DrawFaces(visImage, result.Faces, result.CelebrityNames);
                visImage = Visualization.DrawTags(visImage, result.Tags);
            }

            return visImage;
        }

        /// <summary> Populate CameraList in the UI, once it is loaded. </summary>
        /// <param name="sender"> Source of the event. </param>
        /// <param name="e">      Routed event information. </param>
        private void CameraList_Loaded(object sender, RoutedEventArgs e)
        {
            int numCameras = _grabber.GetNumCameras();

            if (numCameras == 0)
            {
                MessageArea.Text = "No cameras found!";
            }

            var comboBox = sender as ComboBox;
            comboBox.ItemsSource = Enumerable.Range(0, numCameras).Select(i => string.Format("Camera {0}", i + 1));
            comboBox.SelectedIndex = 0;
        }

        /// <summary> Populate ModeList in the UI, once it is loaded. </summary>
        /// <param name="sender"> Source of the event. </param>
        /// <param name="e">      Routed event information. </param>
        private void ModeList_Loaded(object sender, RoutedEventArgs e)
        {
            var modes = (AppMode[])Enum.GetValues(typeof(AppMode));

            var comboBox = sender as ComboBox;
            comboBox.ItemsSource = modes.Select(m => m.ToString());
            comboBox.SelectedIndex = 0;
        }

        private void ModeList_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            // Disable "most-recent" results display. 
            _fuseClientRemoteResults = false;

            var comboBox = sender as ComboBox;
            var modes = (AppMode[])Enum.GetValues(typeof(AppMode));
            _mode = modes[comboBox.SelectedIndex];
            switch (_mode)
            {
                case AppMode.Faces:
                    _grabber.AnalysisFunction = OpenCVDiffContourPeopleDetect;
                    break;
                case AppMode.Emotions:
                    //_grabber.AnalysisFunction = EmotionAnalysisFunction;
                    _grabber.AnalysisFunction = OpenCVPeopleDetect;
                    break;
                case AppMode.EmotionsWithClientFaceDetect:
                    // Same as Emotions, except we will display the most recent faces combined with
                    // the most recent API results. 
                    _grabber.AnalysisFunction = EmotionAnalysisFunction;
                    _fuseClientRemoteResults = true;
                    break;
                case AppMode.Tags:
                    _grabber.AnalysisFunction = OpenCVDNNYoloPeopleDetect;
                    break;
                case AppMode.Celebrities:
                    _grabber.AnalysisFunction = CelebrityAnalysisFunction;
                    break;
                default:
                    _grabber.AnalysisFunction = null;
                    break;
            }
        }

        private async void StartButton_Click(object sender, RoutedEventArgs e)
        {
            LeftImage.Source = null;
            RightImage.Source = null;
            TempImage1.Source = null;
            TempImage2.Source = null;

            if (!CameraList.HasItems)
            {
                MessageArea.Text = "No cameras found; cannot start processing";
                return;
            }

            // Clean leading/trailing spaces in API keys. 
            Properties.Settings.Default.FaceAPIKey = Properties.Settings.Default.FaceAPIKey.Trim();
            Properties.Settings.Default.VisionAPIKey = Properties.Settings.Default.VisionAPIKey.Trim();

            // Create API clients.
            _faceClient = new FaceAPI.FaceClient(new FaceAPI.ApiKeyServiceClientCredentials(Properties.Settings.Default.FaceAPIKey))
            {
                Endpoint = Properties.Settings.Default.FaceAPIHost
            };
            _visionClient = new VisionAPI.ComputerVisionClient(new VisionAPI.ApiKeyServiceClientCredentials(Properties.Settings.Default.VisionAPIKey))
            {
                Endpoint = Properties.Settings.Default.VisionAPIHost
            };

            // How often to analyze. 
            _grabber.TriggerAnalysisOnInterval(Properties.Settings.Default.AnalysisInterval);

            // Reset message. 
            MessageArea.Text = "hallo";

            // Record start time, for auto-stop
            _startTime = DateTime.Now;

            _diffBaseFrame = null;
            _frameCounter = 0;

            //await _grabber.StartProcessingCameraAsync(CameraList.SelectedIndex);
            //await _grabber.StartProcessingFileAsync(
            //    @"C:\Users\raimo\Downloads\Side Door - 20200518 - 164300_Trim.mp4", 30, RotateFlags.Rotate90Clockwise);

            //            await _grabber.StartProcessingFileAsync(
            //              @"C:\Users\raimo\Downloads\HIKVISION - DS-2CD2143G0-I - 20200518 - 194212-264.mp4", 15, null);
            //    //@"C:\Users\raimo\Downloads\HIKVISION - DS - 2CD2143G0 - I - 20200518 - 194212.mp4", 15);

            //await _grabber.StartProcessingFileAsync(
            //    @"rtsp://admin:M3s%21Ew9JEH%2A%23@foscam.home:88/videoSub", 30, RotateFlags.Rotate90Clockwise);


        //rtsp://admin:examplepass123!@192.168.1.100:554/Streaming/Channels/101/
            await _grabber.StartProcessingFileAsync(
                @"rtsp://admin:nCmDZx8U@192.168.2.125:554/Streaming/Channels/102");
        //rtsp://cam-admin:M3s%21Ew9JEH%2A%23@foscam.home:88/videoMain
            //M3s!Ew9JEH*#
        }

        private async void StopButton_Click(object sender, RoutedEventArgs e)
        {
            await _grabber.StopProcessingAsync();
        }

        private void SettingsButton_Click(object sender, RoutedEventArgs e)
        {
            SettingsPanel.Visibility = 1 - SettingsPanel.Visibility;
        }

        private void SaveSettingsButton_Click(object sender, RoutedEventArgs e)
        {
            SettingsPanel.Visibility = Visibility.Hidden;
            Properties.Settings.Default.Save();
        }

        private void Hyperlink_RequestNavigate(object sender, RequestNavigateEventArgs e)
        {
            Process.Start(new ProcessStartInfo(e.Uri.AbsoluteUri));
            e.Handled = true;
        }

        private FaceAPI.Models.DetectedFace CreateFace(VisionAPI.Models.FaceRectangle rect)
        {
            return new FaceAPI.Models.DetectedFace
            {
                FaceRectangle = new FaceAPI.Models.FaceRectangle
                {
                    Left = rect.Left,
                    Top = rect.Top,
                    Width = rect.Width,
                    Height = rect.Height
                }
            };
        }

        private void MatchAndReplaceFaceRectangles(FaceAPI.Models.DetectedFace[] faces, OpenCvSharp.Rect[] clientRects)
        {
            // Use a simple heuristic for matching the client-side faces to the faces in the
            // results. Just sort both lists left-to-right, and assume a 1:1 correspondence. 

            // Sort the faces left-to-right. 
            var sortedResultFaces = faces
                .OrderBy(f => f.FaceRectangle.Left + 0.5 * f.FaceRectangle.Width)
                .ToArray();

            // Sort the clientRects left-to-right.
            var sortedClientRects = clientRects
                .OrderBy(r => r.Left + 0.5 * r.Width)
                .ToArray();

            // Assume that the sorted lists now corrrespond directly. We can simply update the
            // FaceRectangles in sortedResultFaces, because they refer to the same underlying
            // objects as the input "faces" array. 
            for (int i = 0; i < Math.Min(faces.Length, clientRects.Length); i++)
            {
                // convert from OpenCvSharp rectangles
                OpenCvSharp.Rect r = sortedClientRects[i];
                sortedResultFaces[i].FaceRectangle = new FaceAPI.Models.FaceRectangle { Left = r.Left, Top = r.Top, Width = r.Width, Height = r.Height };
            }
        }

        private bool disposedValue = false; // To detect redundant calls

        protected virtual void Dispose(bool disposing)
        {
            if (!disposedValue)
            {
                if (disposing)
                {
                    _grabber?.Dispose();
                    _visionClient?.Dispose();
                    _faceClient?.Dispose();
                    _localFaceDetector?.Dispose();
                }

                disposedValue = true;
            }
        }

        public void Dispose()
        {
            Dispose(true);
        }
    }
}
