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
using OpenCvSharp;
using VideoFrameAnalyzer;

namespace BasicConsoleSample
{
    internal class Program
    {
        private static void Main(string[] args)
        {
            ServicePointManager.SecurityProtocol = SecurityProtocolType.Tls12;
            // Create grabber. 
            var grabber = new FrameGrabber<DnnDetectedObject[]>();

            // Set up a listener for when we acquire a new frame.
//            grabber.NewFrameProvided += (s, e) =>
//            {
////                Console.WriteLine($"New frame acquired at {e.Frame.Metadata.Timestamp}");
//            };

            // Set up Face API call.
            grabber.AnalysisFunction = OpenCVDNNYoloPeopleDetect;

            // Set up a listener for when we receive a new result from an API call. 
            grabber.NewResultAvailable += (s, e) =>
            {
                if (e.TimedOut)
                    Console.WriteLine("API call timed out.");
                else if (e.Exception != null)
                    Console.WriteLine($"API call threw an exception: {e.Exception}");
                else
                {
                    Console.WriteLine($"New result received for frame acquired at {e.Frame.Metadata.Timestamp}. {e.Analysis.Length} objects detected");
                    foreach (var dObj in e.Analysis)
                    {
                        Console.WriteLine($"Detected: {dObj.Label} ; prob: {dObj.Probability}");
                    }
                    
                    if (e.Analysis.Length > 0 && e.Analysis.Any(o => o.Label != "car" && o.Label != "truck"))
                    {
                        var result = Visualizer.AnnotateImage(e.Frame.Image, e.Analysis);
                        var filename = $"obj-{e.Frame.Metadata.Index}.jpg";
                        Cv2.ImWrite(filename, result);
                        Console.WriteLine($"Interesting Detection Saved: {filename}");
                    }
                }
            };

            // Tell grabber when to call API.
            // See also TriggerAnalysisOnPredicate
            grabber.TriggerAnalysisOnInterval(TimeSpan.FromMilliseconds(3000));

            // Start running in the background.
            //grabber.StartProcessingCameraAsync().Wait();

            //grabber.StartProcessingFileAsync(
            //    @"C:\Users\raimo\Downloads\Side Door - 20200518 - 164300_Trim.mp4",
            //    isContinuousStream: false, rotateFlags: RotateFlags.Rotate90Clockwise).Wait();


            grabber.StartProcessingFileAsync(
                @"rtsp://cam-admin:M3s%21Ew9JEH%2A%23@foscam.home:88/videoSub",
                rotateFlags: RotateFlags.Rotate90Clockwise
                , overrideFPS: 15
            ).Wait();



            // Wait for keypress to stop
            Console.WriteLine("Press any key to stop...");
            Console.ReadKey();

            // Stop, blocking until done.
            grabber.StopProcessingAsync().Wait();
        }

        private static Yolo3DnnDetector _dnnDetector = new Yolo3DnnDetector();

        private static Task<DnnDetectedObject[]> OpenCVDNNYoloPeopleDetect(VideoFrame frame)
        {
            var image = frame.Image;
            if (image == null || image.Width <= 0 || image.Height <= 0)
            {
                return Task.FromResult(new DnnDetectedObject[0]);
            }

            Func<DnnDetectedObject[]> detector = () =>
            {
                DnnDetectedObject[] result;

                try
                {
                    var watch = new Stopwatch();
                    watch.Start();

                    result = _dnnDetector.ClassifyObjects(image, Rect.Empty);

                    watch.Stop();
                    ConcurrentLogger.WriteLine($"Classifiy-objects ms:{watch.ElapsedMilliseconds}");
                }
                catch (Exception ex)
                {
                    result = new DnnDetectedObject[0];
                    ConcurrentLogger.WriteLine($"Exception in analysis:{ex.Message}");
                }

                return result;
            };

            //var result2 = detector();
            //return Task.FromResult(result2);

            return Task.Run(() => detector());
        }

    }
}
