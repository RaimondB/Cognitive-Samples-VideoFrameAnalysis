// Uncomment this to enable the LogMessage function, which can with debugging timing issues.
#define TRACE_GRABBER

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

        public bool IsRunning { get { return _mergeTask != null; } }

        //public double FrameRate
        //{
        //    get { return _fps; }
        //    set
        //    {
        //        _fps = value;
        //        if (_timer != null)
        //        {
        //            _timer.Change(TimeSpan.Zero, TimeSpan.FromSeconds(1.0 / _fps));
        //        }
        //    }
        //}

        public int Width { get; protected set; }
        public int Height { get; protected set; }

        #endregion Properties

        #region Fields

        protected Predicate<VideoFrame> _analysisPredicate = null;
        protected List<VideoStream> _streams = new List<VideoStream>();

        protected bool _stopping = false;
        protected Task _producerTask = null;
        protected Task _consumerTask = null;
        protected Task _mergeTask = null;

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

        protected async Task<(bool,TResult)> DoWithTimeout<TResult>(Func<Task<TResult>> func, TimeSpan timeout)
        {
            using (CancellationTokenSource source = new CancellationTokenSource())
            {
                source.CancelAfter(timeout);

                try 
                {
                    var result = await Task.Run(func, source.Token);
                    return (true, result);
                }
                catch(OperationCanceledException ex)
                {
                    LogMessage($"Exception in dowithtimeout:{ex.Message}");
                    return (false, default(TResult));
                }
            }
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
                            output.Analysis = await task.ConfigureAwait(false);
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

        private static Channel<VideoFrame> CreateCapturingChannel() =>
            Channel.CreateBounded<VideoFrame>(
                new BoundedChannelOptions(2)
                {
                    AllowSynchronousContinuations = true,
                    FullMode = BoundedChannelFullMode.DropNewest,
                    SingleReader = true,
                    SingleWriter = true
                });

        private static Channel<VideoFrame> CreateMergeChannel(int inputs) =>
    Channel.CreateBounded<VideoFrame>(
        new BoundedChannelOptions(inputs)
        {
            AllowSynchronousContinuations = false,
            FullMode = BoundedChannelFullMode.DropOldest,
            SingleReader = true,
            SingleWriter = true
        });

        private List<Channel<VideoFrame>> _capturingChannels = new List<Channel<VideoFrame>>();
        private List<Task> _producerTasks = new List<Task>();


        public Task StartProcessingFileAsync(string fileName, double overrideFPS = 0, bool isContinuousStream = true, RotateFlags? rotateFlags = null)
        {
            VideoStream vs = new VideoStream("first", fileName, overrideFPS, isContinuousStream, rotateFlags);

            _streams.Add(vs);
            
            var newChannel = CreateCapturingChannel();
            _capturingChannels.Add(newChannel);
            
            vs.StartProcessingAsync(newChannel, TimeSpan.FromSeconds(3));

            return Task.CompletedTask;
        }

        public Task MergeChannels<T>(
            List<Channel<T>> inputChannels, Channel<T> outputChannel, TimeSpan mergeDelay)
        {
            var timeout = TimeSpan.FromMilliseconds(50);
//            using var cts = new CancellationTokenSource();
//            using var timeoutCts = new CancellationTokenSource();
            //timeoutCts.CancelAfter(150);

            return Task.Run(async () =>
            {
                var readers = inputChannels.Select(c => c.Reader);
                var writer = outputChannel.Writer;

                while(!_stopping)
                {
                    foreach(var reader in readers )
                    {
                        try
                        {
                            var result = await reader.ReadAsync().ConfigureAwait(false);
                            await writer.WriteAsync(result).ConfigureAwait(false);
                        }
                        catch(Exception ex)
                        {
                            LogMessage($"Exception during channel merge:{ex.ToString()}");
                            //Just continue to next on errors or timeouts
                        }
                    }
                    await Task.Delay(mergeDelay).ConfigureAwait(false);
                }
            });
        }

        /// <summary> Starts capturing and processing video frames. </summary>
        /// <param name="frameGrabDelay"> The frame grab delay. </param>
        /// <param name="timestampFn">    Function to generate the timestamp for each frame. This
        ///     function will get called once per frame. </param>
        public void StartProcessingAll()
        {
            var analysisChannel = CreateMergeChannel(_capturingChannels.Count);
            _mergeTask = MergeChannels(_capturingChannels, analysisChannel, _analysisInterval);

            _consumerTask = Task.Run(async () =>
            {
                var reader = analysisChannel.Reader;

                while (!_stopping)
                {
                    LogMessage("Consumer: waiting for next result to arrive");

                    var vframe = await reader.ReadAsync();

                    var startTime = DateTime.Now;

                    try
                    {
                        var result = await DoAnalyzeFrame(vframe);

                        LogMessage("Consumer: analysis took {0} ms", (DateTime.Now - startTime).Milliseconds);

                        OnNewResultAvailable(result);
                    }
                    catch(Exception ex)
                    {
                        LogMessage($"Exception in consumertaks:{ex.Message}");
                        //try to continue always
                    }
                }

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

            if (_consumerTask != null)
            {
                await _consumerTask;
                _consumerTask = null;
            }

            if (_mergeTask != null)
            {
                await _mergeTask;
                _mergeTask = null;
            }

            foreach (VideoStream vs in _streams)
            {
                await vs.StopProcessingAsync();
                vs.Dispose();
            }
            _streams.Clear();

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
                    //_frameGrabTimer?.Dispose();
                    //_timer?.Dispose();
                    //_timerMutex?.Dispose();
                    //_analysisTaskQueue?.Dispose();
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
