// Uncomment this to enable the LogMessage function, which can with debugging timing issues.
#define TRACE_GRABBER

using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading;
using System.Threading.Channels;
using System.Threading.Tasks;
using VideoFrameAnalyzeStd.VideoCapturing;
using System.Linq;
using System.Globalization;
using Microsoft.Extensions.Logging;

namespace VideoFrameAnalyzer
{
    /// <summary> A frame grabber. </summary>
    /// <typeparam name="TAnalysisResultType"> Type of the analysis result. This is the type that
    ///     the AnalysisFunction will return, when it calls some API on a video frame. </typeparam>
    public class MultiStreamBatchedFrameGrabber<TAnalysisResultType> : IDisposable
    {
        #region Types

        /// <summary> Additional information for new frame events. </summary>
        /// <seealso cref="System.EventArgs"/>
        public class NewFramesEventArgs : EventArgs
        {
            public NewFramesEventArgs(IList<VideoFrame> frame)
            {
                Frame = frame;
            }
            public IList<VideoFrame> Frame { get; }
        }

        /// <summary> Additional information for new result events, which occur when an API call
        ///     returns. </summary>
        /// <seealso cref="System.EventArgs"/>
        public class NewResultsEventArgs : EventArgs
        {
            public NewResultsEventArgs(IList<VideoFrame> frames)
            {
                Frames = frames;
            }
            public IList<VideoFrame> Frames { get; }
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
        public Func<IList<VideoFrame>, Task<TAnalysisResultType>> AnalysisFunction { get; set; } = null;

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

        #endregion Properties

        #region Fields

        private List<VideoStream> _streams = new List<VideoStream>();

        private bool _stopping = false;
        private Task _consumerTask = null;
        private Task _mergeTask = null;

        private bool disposedValue = false;
        private ILogger _logger;

        #endregion Fields

        #region Methods

        public MultiStreamBatchedFrameGrabber(ILogger<MultiFrameGrabber<TAnalysisResultType>> logger)
        {
            _logger = logger;
        }

        /// <summary> (Only available in TRACE_GRABBER builds) logs a message. </summary>
        /// <param name="format"> Describes the format to use. </param>
        /// <param name="args">   Event information. </param>
        [Conditional("TRACE_GRABBER")]
        protected void LogMessage(string format, params object[] args)
        {
            _logger.LogInformation(String.Format(CultureInfo.InvariantCulture, format, args));
        }

        [Conditional("TRACE_GRABBER")]
        protected void LogDebug(string format, params object[] args)
        {
            _logger.LogDebug(String.Format(CultureInfo.InvariantCulture, format, args));
        }

        [Conditional("TRACE_GRABBER")]
        protected void LogWarning(string format, params object[] args)
        {
            _logger.LogDebug(String.Format(CultureInfo.InvariantCulture, format, args));
        }

        protected async Task<NewResultsEventArgs> DoAnalyzeFrames(IList<VideoFrame> frames)
        {
            using (CancellationTokenSource source = new CancellationTokenSource())
            {
                // Make a local reference to the function, just in case someone sets
                // AnalysisFunction = null before we can call it.
                var fcn = AnalysisFunction;
                if (fcn != null)
                {
                    NewResultsEventArgs output = new NewResultsEventArgs(frames);
                    var task = fcn(frames);
                    LogDebug("DoAnalysis: started task {0}", task.Id);
                    try
                    {
                        if (task == await Task.WhenAny(task, Task.Delay(AnalysisTimeout, source.Token)))
                        {
                            output.Analysis = await task.ConfigureAwait(false);
                            source.Cancel();
                        }
                        else
                        {
                            LogWarning("DoAnalysis: Timeout from task {0}", task.Id);
                            output.TimedOut = true;
                        }
                    }
                    catch (Exception ae)
                    {
                        output.Exception = ae;
                        LogWarning("DoAnalysis: Exception from task {0}:{1}", task.Id, ae.Message);
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

        private static Channel<IList<VideoFrame>> CreateMergeChannel() =>
            Channel.CreateBounded<IList<VideoFrame>>(
        new BoundedChannelOptions(2)
        {
            AllowSynchronousContinuations = false,
            FullMode = BoundedChannelFullMode.DropNewest,
            SingleReader = true,
            SingleWriter = true
        });

        private List<Channel<VideoFrame>> _capturingChannels = new List<Channel<VideoFrame>>();


        public void StartProcessingFileAsync(string fileName, double overrideFPS = 0, bool isContinuousStream = true, RotateFlags? rotateFlags = null)
        {
            VideoStream vs = new VideoStream(_logger, "first", fileName, overrideFPS, isContinuousStream, rotateFlags);

            _streams.Add(vs);

            var newChannel = CreateCapturingChannel();
            _capturingChannels.Add(newChannel);

            vs.StartProcessingAsync(newChannel, TimeSpan.FromSeconds(3));
        }

        public Task MergeChannels<T>(
            IList<Channel<T>> inputChannels, Channel<IList<T>> outputChannel, TimeSpan mergeDelay)
        {
            var timeout = TimeSpan.FromMilliseconds(50);
            //            using var cts = new CancellationTokenSource();
            //            using var timeoutCts = new CancellationTokenSource();
            //timeoutCts.CancelAfter(150);

            return Task.Run(async () =>
            {
                var readers = inputChannels.Select(c => c.Reader).ToArray();
                var writer = outputChannel.Writer;

                while (!_stopping)
                {
                    List<T> results = new List<T>();
                    for (var index = 0; index < readers.Length; index++)
                    {
                        try
                        {
                            var result = await readers[index].ReadAsync().ConfigureAwait(false);
                            results.Add(result);
                        }
                        catch (Exception ex)
                        {
                            LogMessage($"Exception during channel merge:{ex.ToString()}");
                            //Just continue to next on errors or timeouts
                        }
                    }
                    await writer.WriteAsync(results).ConfigureAwait(false);

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
            var analysisChannel = CreateMergeChannel();
            _mergeTask = MergeChannels(_capturingChannels, analysisChannel, _analysisInterval);

            _consumerTask = Task.Run(async () =>
            {
                var reader = analysisChannel.Reader;

                while (!_stopping)
                {
                    LogMessage("Consumer: waiting for next result to arrive");

                    try
                    {
                        var vframes = await reader.ReadAsync();

                        var startTime = DateTime.Now;

                        var result = await DoAnalyzeFrames(vframes);

                        LogMessage("Consumer: analysis took {0} ms", (DateTime.Now - startTime).TotalMilliseconds);

                        OnNewResultAvailable(result);
                    }
                    catch (Exception ex)
                    {
                        LogMessage($"Exception in consumertask:{ex.Message}");
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
        }

        /// <summary> Raises the new frame provided event. </summary>
        /// <param name="frame"> The frame. </param>
        protected void OnNewFrameProvided(IList<VideoFrame> frames)
        {
            NewFramesProvided?.Invoke(this, new NewFramesEventArgs(frames));
        }

        /// <summary> Raises the new result event. </summary>
        /// <param name="args"> Event information to send to registered event handlers. </param>
        protected void OnNewResultAvailable(NewResultsEventArgs args)
        {
            NewResultsAvailable?.Invoke(this, args);
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

        public event EventHandler<NewFramesEventArgs> NewFramesProvided;
        public event EventHandler<NewResultsEventArgs> NewResultsAvailable;

        #endregion Events
    }
}
