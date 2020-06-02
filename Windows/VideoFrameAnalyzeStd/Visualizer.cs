using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Text;

namespace VideoFrameAnalyzer
{
    public class Visualizer
    {
        public static Mat AnnotateImage(Mat orgImage, DnnDetectedObject[] detectedObjects)
        {
            Mat result = new Mat();
            Cv2.CopyTo(orgImage, result);
            foreach (var dObj in detectedObjects)
            {
                var x1 = dObj.BoundingBox.X;
                var y1 = dObj.BoundingBox.Y;
                var w = dObj.BoundingBox.Width;
                var h = dObj.BoundingBox.Height;
                var color = dObj.Color;

                var label = $"{dObj.Label} {dObj.Probability * 100:0.00}%";
                //var x1 = (centerX - width / 2) < 0 ? 0 : centerX - width / 2; //avoid left side over edge
                                                                              //draw result
                result.Rectangle(new Point(x1, y1), new Point(x1+w, y1+h), color, 2);
                
                var textSize = Cv2.GetTextSize(label, HersheyFonts.HersheyTriplex, 0.5, 1, out var baseline);
                Cv2.Rectangle(result, new Rect(new Point(x1, y1 - textSize.Height - baseline),
                        new Size(textSize.Width, textSize.Height + baseline)), color, Cv2.FILLED);
                Cv2.PutText(result, label, new Point(x1, y1 - baseline), 
                    HersheyFonts.HersheyTriplex, 0.5, Scalar.Black);
            }
            return result;
        }
    }
}
