// Uncomment this to enable the LogMessage function, which can with debugging timing issues.
//#define TRACE_GRABBER

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;
using System.Threading.Channels;

using OpenCvSharp;
using VideoFrameAnalyzeStd.VideoCapturing;
using System.Linq;

namespace VideoFrameAnalyzer
{
    /// <summary> A frame grabber. </summary>
    /// <typeparam name="TAnalysisResultType"> Type of the analysis result. This is the type that
    ///     the AnalysisFunction will return, when it calls some API on a video frame. </typeparam>
    public class MultiFrameGrabber<TAnalysisResultType> : IDisposable
    {
        #region Types

        /// <summary> Additional information for new frame events. </summary>
        /// <seealso cref="T:System.EventArgs"/>
        public class NewFrameEventArgs : EventArgs
        {
            public NewFrameEventArgs(VideoFrame frame)
            {
                Frame = frame;
            }
            public VideoFrame Frame { get; }
        }

        /// <summary> Additional information for new result events, which occur when an API call
        ///     returns. </summary>
        /// <seealso cref="T:System.EventArgs"/>
        public class NewResultEventArgs : EventArgs
        {
            public NewResultEventArgs(VideoFrame frame)
            {
                Frame = frame;
            }
            public VideoFrame Frame { get; }
            public TAnalysisResultType Analysis { get; set; } = default(TAnalysisResultType);
            public bool TimedOut { get; set; } = false;
            public Exception Exception { get; set; } = null;
        }

        #endregion Types

        #region Properties

        /// <summary> Gets or sets the analysis function. The function can be any asynchronous
        ///     operation that accepts a <see cref="VideoFrame"/> and returns a
        ///     <see cref="Task{AnalysisResultType}"/>. </summary>
        /// <value> The analysis function. </value>
        /// <example> This example shows how to provide an analysis function using a lambda expression.
        ///     <code>
        ///     var client = new FaceServiceClient("subscription key", "api root");
        ///     var grabber = new FrameGrabber();
        ///     grabber.AnalysisFunction = async (frame) =&gt; { return await client.RecognizeAsync(frame.Image.ToMemoryStream(".jpg")); };
        ///     </code></example>
        public Func<VideoFrame, Task<TAnalysisResultType>> AnalysisFunction { get; set; } = null;

        /// <summary> Gets or sets the analysis timeout. When executing the
        ///     <see cref="AnalysisFunction"/> on a video frame, if the call doesn't return a
        ///     result within this time, it is abandoned and no result is returned for that
        ///     frame. </summary>
        /// <value> The analysis timeout. </value>
        public TimeSpan AnalysisTimeout { get; set; } = TimeSpan.FromMilliseconds(5000);

        public bool IsRunning { get { return _analysisTaskQueue != null; } }

        public double FrameRate
        {
            get { return _fps; }
            set
            {
                _fps = value;
                if (_timer != null)
                {
                    _timer.Change(TimeSpan.Zero, TimeSpan.FromSeconds(1.0 / _fps));
                }
            }
        }

        public int Width { get; protected set; }
        public int Height { get; protected set; }

        #endregion Properties

        #region Fields

        protected Predicate<VideoFrame> _analysisPredicate = null;
        protected List<VideoStream> _streams = null;
        protected bool _readerIsContinuous = false;

        protected Timer _timer = null;
        protected SemaphoreSlim _timerMutex = new SemaphoreSlim(1);
        protected AutoResetEvent _frameGrabTimer = new AutoResetEvent(false);
        protected bool _stopping = false;
        protected Task _producerTask = null;
        protected Task _consumerTask = null;
        protected BlockingCollection<Task<NewResultEventArgs>> _analysisTaskQueue = null;
        protected bool _resetTrigger = true;
        protected int _numCameras = -1;
        protected int _currCameraIdx = -1;
        protected double _fps = 0;
        private bool disposedValue = false;

        #endregion Fields

        #region Methods

        public MultiFrameGrabber()
        {
        }

        /// <summary> (Only available in TRACE_GRABBER builds) logs a message. </summary>
        /// <param name="format"> Describes the format to use. </param>
        /// <param name="args">   Event information. </param>
        [Conditional("TRACE_GRABBER")]
        protected void LogMessage(string format, params object[] args)
        {
            ConcurrentLogger.WriteLine(String.Format(format, args));
        }

        protected async Task<NewResultEventArgs> DoAnalyzeFrame(VideoFrame frame)
        {
            using (CancellationTokenSource source = new CancellationTokenSource())
            {
                // Make a local reference to the function, just in case someone sets
                // AnalysisFunction = null before we can call it.
                var fcn = AnalysisFunction;
                if (fcn != null)
                {
                    NewResultEventArgs output = new NewResultEventArgs(frame);
                    var task = fcn(frame);
                    LogMessage("DoAnalysis: started task {0}", task.Id);
                    try
                    {
                        if (task == await Task.WhenAny(task, Task.Delay(AnalysisTimeout, source.Token)))
                        {
                            output.Analysis = await task;
                            source.Cancel();
                        }
                        else
                        {
                            LogMessage("DoAnalysis: Timeout from task {0}", task.Id);
                            output.TimedOut = true;
                        }
                    }
                    catch (Exception ae)
                    {
                        output.Exception = ae;
                        LogMessage("DoAnalysis: Exception from task {0}:{1}", task.Id, ae.Message);
                    }

                    LogMessage("DoAnalysis: returned from task {0}", task.Id);

                    return output;
                }
                else
                {
                    return null;
                }
            }
        }

        private TimeSpan _analysisInterval;
        public void TriggerAnalysisOnInterval(TimeSpan interval)
        {
            _analysisInterval = interval;
        }

        private readonly Channel<VideoFrame> _capturingChannel =
            Channel.CreateBounded<VideoFrame>(
                new BoundedChannelOptions(1)
                {
                    AllowSynchronousContinuations = true,
                    FullMode = BoundedChannelFullMode.DropOldest,
                    SingleReader = true,
                    SingleWriter = false
                });

        public async Task StartProcessingFileAsync(string fileName, double overrideFPS = 0, bool isContinuousStream = true, RotateFlags? rotateFlags = null)
        {
            VideoStream vs = new VideoStream("first", fileName, overrideFPS, isContinuousStream, rotateFlags);

            //_streams.Add(vs);
            StartProcessing(vs);
        }

        /// <summary> Starts capturing and processing video frames. </summary>
        /// <param name="frameGrabDelay"> The frame grab delay. </param>
        /// <param name="timestampFn">    Function to generate the timestamp for each frame. This
        ///     function will get called once per frame. </param>
        protected void StartProcessing(VideoStream videoStream)
        {
            _producerTask = Task.Run(() => videoStream.StartProcessingAsync(_capturingChannel));

            _consumerTask = Task.Run(async () =>
            {
                var reader = _capturingChannel.Reader;

                while (!_stopping)
                {
                    LogMessage("Consumer: waiting for next result to arrive");

                    var vframe = await reader.ReadAsync();

                    var startTime = DateTime.Now;

                    var result = await DoAnalyzeFrame(vframe);

                    LogMessage("Consumer: analysis took {0} ms", (DateTime.Now - startTime).Milliseconds);

                    var nextAnalysisDue = Task.Delay(_analysisInterval);

                    OnNewResultAvailable(result);

                    await nextAnalysisDue;
                }

                LogMessage("Producer: stopping, destroy reader and timer");

                videoStream.StopProcessing();

                LogMessage("Consumer: stopping");
            });
        }

        /// <summary> Stops capturing and processing video frames. </summary>
        /// <returns> A Task. </returns>
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Reliability", "CA2007:Consider calling ConfigureAwait on the awaited task", Justification = "Sync needed because of eventhandlers")]
        public async Task StopProcessingAsync()
        {
            OnProcessingStopping();

            _stopping = true;
            if (_producerTask != null)
            {
                await _producerTask;
                _producerTask = null;
            }
            if (_consumerTask != null)
            {
                await _consumerTask;
                _consumerTask = null;
            }
            _stopping = false;

            OnProcessingStopped();
        }

        /// <summary> Raises the processing starting event. </summary>
        protected void OnProcessingStarting()
        {
            ProcessingStarting?.Invoke(this, null);
        }

        /// <summary> Raises the processing started event. </summary>
        protected void OnProcessingStarted()
        {
            ProcessingStarted?.Invoke(this, null);
        }

        /// <summary> Raises the processing stopping event. </summary>
        protected void OnProcessingStopping()
        {
            ProcessingStopping?.Invoke(this, null);
        }

        /// <summary> Raises the processing stopped event. </summary>
        protected void OnProcessingStopped()
        {
            ProcessingStopped?.Invoke(this, null);
        }

        /// <summary> Raises the new frame provided event. </summary>
        /// <param name="frame"> The frame. </param>
        protected void OnNewFrameProvided(VideoFrame frame)
        {
            NewFrameProvided?.Invoke(this, new NewFrameEventArgs(frame));
        }

        /// <summary> Raises the new result event. </summary>
        /// <param name="args"> Event information to send to registered event handlers. </param>
        protected void OnNewResultAvailable(NewResultEventArgs args)
        {
            NewResultAvailable?.Invoke(this, args);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!disposedValue)
            {
                if (disposing)
                {
                    _frameGrabTimer?.Dispose();
                    _timer?.Dispose();
                    _timerMutex?.Dispose();
                    _analysisTaskQueue?.Dispose();
                }

                disposedValue = true;
            }
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        #endregion Methods

        #region Events

        public event EventHandler ProcessingStarting;
        public event EventHandler ProcessingStarted;
        public event EventHandler ProcessingStopping;
        public event EventHandler ProcessingStopped;
        public event EventHandler<NewFrameEventArgs> NewFrameProvided;
        public event EventHandler<NewResultEventArgs> NewResultAvailable;

        #endregion Events
    }
}
