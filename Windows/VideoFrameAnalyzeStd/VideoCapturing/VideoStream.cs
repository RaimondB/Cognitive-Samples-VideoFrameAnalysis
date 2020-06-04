//#define TRACE_GRABBER

using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO.IsolatedStorage;
using System.Text;
using System.Threading.Channels;
using System.Threading.Tasks;
using VideoFrameAnalyzer;

namespace VideoFrameAnalyzeStd.VideoCapturing
{

    public class VideoStream : IDisposable
    {
        public string Path  { get; }

        private double _fps;
        public double Fps => _fps;

        private VideoCapture _videoCapture;
        public VideoCapture VideoCapture => _videoCapture;

        public string StreamName { get; }

        public bool IsContinuous { get; }

        public RotateFlags? RotateFlags { get; }

        private bool _stopping;

        public VideoStream(string streamName, string path, double fps = 0, bool isContinuous = true, RotateFlags? rotateFlags = null)
        {
            Path = path;
            _fps = fps;
            StreamName = streamName;
            IsContinuous = isContinuous;
            RotateFlags = rotateFlags;
            _stopping = false;
        }

        [Conditional("TRACE_GRABBER")]
        protected void LogMessage(string format, params object[] args)
        {
            ConcurrentLogger.WriteLine(String.Format(format, args));
        }

        public VideoCapture StartCapturing()
        {
            _videoCapture = new VideoCapture(Path);

            if (Fps == 0)
            {
                var rFpds = _videoCapture.Fps;

                _fps = (rFpds == 0) ? 30 : rFpds;
            }

            return _videoCapture;
        }

        public void StopProcessing()
        {
            _stopping = true;
        }

        public async Task StartProcessingAsync(Channel<VideoFrame> outputChannel)
        {
            using (var reader = this.StartCapturing())
            {

                var width = reader.FrameWidth;
                var height = reader.FrameHeight;
                int frameCount = 0;
                int delayMs = (int)(500.0 / this.Fps);

                var writer = outputChannel.Writer;

                while (!_stopping)
                {
                    var startTime = DateTime.Now;

                    // Grab single frame.
                    var timestamp = DateTime.Now;

                    Mat image = new Mat();
                    bool success = reader.Read(image);
                    frameCount++;

                    LogMessage("Producer: frame-grab took {0} ms", (DateTime.Now - startTime).Milliseconds);

                    if (!success)
                    {
                        // If we've reached the end of the video, stop here.
                        if (!IsContinuous)
                        {
                            LogMessage("Producer: null frame from video file, stop!");
                            // This will call StopProcessing on a new thread.
                            _stopping = true;
                            writer.Complete();
                            // Break out of the loop to make sure we don't try grabbing more
                            // frames.
                            break;
                        }
                        else
                        {
                            // If failed on live camera, try again.
                            LogMessage("Producer: null frame from live camera, continue!");
                            continue;
                        }
                    }

                    if (RotateFlags.HasValue)
                    {
                        Mat rotImage = new Mat();
                        Cv2.Rotate(image, rotImage, RotateFlags.Value);

                        image = rotImage;
                    }

                    // Package the image for submission.
                    VideoFrameMetadata meta;
                    meta.Index = frameCount;
                    meta.Timestamp = timestamp;
                    VideoFrame vframe = new VideoFrame(image, meta);

                    writer.TryWrite(vframe);

                    await Task.Delay(delayMs);
                }
            }
            // We reach this point by breaking out of the while loop. So we must be stopping.
        }


        public void Dispose()
        {
            ((IDisposable)_videoCapture).Dispose();
        }
    }
}
