using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Console;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using VideoFrameAnalyzer;
using VideoFrameAnalyzeStd.Detection;

namespace CameraWatcher
{

    class Program
    {
        public static IHostBuilder CreateHostBuilder(string[] args) =>
            Host.CreateDefaultBuilder(args)
            .ConfigureServices((hostContext, services) =>
            {
                //services.AddHostedService<CameraWatcherService>();
                //services.AddSingleton<IDnnDetector, Yolo3DnnDetector>();
                //services.AddSingleton<MultiFrameGrabber<DnnDetectedObject[]>,
                //    MultiFrameGrabber<DnnDetectedObject[]>>();
                //services.AddSingleton<MultiFrameGrabber<DnnDetectedObject[]>,
                //    MultiFrameGrabber<DnnDetectedObject[]>>();
                services.AddHostedService<BatchedCameraWatcherService>();
                services.AddSingleton<IBatchedDnnDetector, Yolo3BatchedDnnDetector>();
                services.AddSingleton<MultiStreamBatchedFrameGrabber<DnnDetectedObject[][]>,
                    MultiStreamBatchedFrameGrabber<DnnDetectedObject[][]>>();
            })
            .ConfigureLogging(logging =>
            {
                logging.AddConsole(c =>
                {
                    c.TimestampFormat = "[HH:mm:ss] ";
                    c.IncludeScopes = false;
                });
            });

        static async Task Main(string[] args)
        {
            await CreateHostBuilder(args).Build().RunAsync();
        }
    }
}
