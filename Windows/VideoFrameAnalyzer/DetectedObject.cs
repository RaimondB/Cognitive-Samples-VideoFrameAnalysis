using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace VideoFrameAnalyzer
{
    public class DnnDetectedObject
    {
        public string Label { get; set; }
        public float Probability { get; set; }
        public Rect2d BoundingBox { get; set; }
    }
}
