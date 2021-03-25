using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using Newtonsoft.Json;

namespace ObjectDetection.RetinaParser
{
    public class CellDimensions : DimensionsBase { }

    public class RetinaOutputParser
    {
        private static readonly Color[] ClassColors = new Color[] { Color.Khaki, Color.Fuchsia, Color.Silver, Color.RoyalBlue, Color.Green, Color.DarkOrange, Color.Purple, Color.Gold, Color.Red, Color.Aquamarine, Color.Lime, Color.AliceBlue, Color.Sienna, Color.Orchid, Color.Tan, Color.LightPink, Color.Yellow, Color.HotPink, Color.OliveDrab, Color.SandyBrown, Color.DarkTurquoise };
        private static readonly string JSONLocation = @"C:\Users\imreb\source\repos\ObjectDetection\ObjectDetection\assets\Dataset\Classes.json";
        private readonly float[] Anchors = new float[] { 1.08F, 1.19F, 3.42F, 4.41F, 6.63F, 11.38F, 9.42F, 5.11F, 16.62F, 10.52F };
        private readonly List<string> Classes = new();

        public RetinaOutputParser(int cellHeight, int cellWidth, int colCount, int rowCount, int boxesPerCell = 5, int boxInfoFeatureCount = 5)
        {
            LoadJSON();
            Console.WriteLine($"Detecting objects with {ClassCount} {(ClassCount == 1 ? "class" : "classes")} available.");
            Console.WriteLine("");
            BoxesPerCell = boxesPerCell;
            BoxInfoFeatureCount = boxInfoFeatureCount;
            CellHeight = cellHeight;
            CellWidth = cellWidth;
            ColCount = colCount;
            RowCount = rowCount;
        }

        private int BoxesPerCell { set; get; }

        private int BoxInfoFeatureCount { set; get; }

        private int CellHeight { set; get; }

        private int CellWidth { set; get; }

        private int ChannelStride => ColCount * RowCount;

        private int ClassCount => Classes.Count;

        private int ColCount { set; get; }

        private int RowCount { set; get; }

        public static IList<RetinaBoundingBox> FilterBoundingBoxes(IList<RetinaBoundingBox> boxes, int limit, float threshold)
        {
            int activeCount = boxes.Count;
            bool[] isActiveBoxes = new bool[boxes.Count];

            // Mark all boxes ready for processing
            for (int i = 0; i < isActiveBoxes.Length; i++) isActiveBoxes[i] = true;

            // Sort the bounding boxes in descending order based on confidence
            var sortedBoxes = boxes.Select((b, i) => new { Box = b, Index = i })
                    .OrderByDescending(b => b.Box.Confidence)
                    .ToList();

            // Hold filtered results
            List<RetinaBoundingBox> results = new();

            for (int i = 0; i < boxes.Count; i++)
            {
                if (isActiveBoxes[i])
                {
                    RetinaBoundingBox boxA = sortedBoxes[i].Box;
                    results.Add(boxA);

                    // If
                    if (results.Count >= limit)
                        break;

                    for (int j = i + 1; j < boxes.Count; j++)
                    {
                        // If adjacent  box is ready to be processed
                        if (isActiveBoxes[j])
                        {
                            RetinaBoundingBox boxB = sortedBoxes[j].Box;

                            // Check whether the first box and the second box exceeds the threshold
                            if (IntersectionOverUnion(boxA.Rect, boxB.Rect) > threshold)
                            {
                                isActiveBoxes[j] = false;
                                activeCount--;

                                if (activeCount <= 0)
                                    break;
                            }
                        }
                    }
                    if (activeCount <= 0)
                        break;
                }
            }
            return results;
        }

        public float[] ExtractClasses(float[] modelOutput, int x, int y, int channel)
        {
            float[] predictedClasses = new float[ClassCount];
            int predictedClassOffset = channel + BoxInfoFeatureCount;
            for (int predictedClass = 0; predictedClass < ClassCount; predictedClass++)
            {
                predictedClasses[predictedClass] = modelOutput[GetOffset(x, y, predictedClass + predictedClassOffset)];
            }
            return Softmax(predictedClasses);
        }

        public IList<RetinaBoundingBox> ParseOutputs(float[] retinaModelOutputs, float threshold = .3F)
        {
            List<RetinaBoundingBox> boxes = new();

            for (int row = 0; row < RowCount; row++)
            {
                for (int column = 0; column < ColCount; column++)
                {
                    for (int box = 0; box < BoxesPerCell; box++)
                    {
                        // Calculate starting position of current box
                        int channel = box * (ClassCount + BoxInfoFeatureCount);

                        // Get x & y of current box
                        BoundingBoxDimensions boundingBoxDimensions = ExtractBoundingBoxDimensions(retinaModelOutputs, row, column, channel);

                        float confidence = GetConfidence(retinaModelOutputs, row, column, channel);

                        // Map current box to current cell
                        CellDimensions mappedBoundingBox = MapBoundingBoxToCell(row, column, box, boundingBoxDimensions);

                        if (confidence < threshold) continue;

                        // Get the probability distribution of the predicted classes in the current box
                        float[] predictedClasses = ExtractClasses(retinaModelOutputs, row, column, channel);

                        (int topResultIndex, float topResultScore) = GetTopResult(predictedClasses);
                        float topScore = topResultScore * confidence;

                        if (topScore < threshold) continue;

                        // If current box exceeds threshold, Add box to boxes List
                        boxes.Add(new RetinaBoundingBox()
                        {
                            Dimensions = new BoundingBoxDimensions
                            {
                                X = mappedBoundingBox.X - mappedBoundingBox.Width / 2,
                                Y = mappedBoundingBox.Y - mappedBoundingBox.Height / 2,
                                Width = mappedBoundingBox.Width,
                                Height = mappedBoundingBox.Height,
                            },
                            Confidence = topScore,
                            Label = Classes[topResultIndex],
                            BoxColor = ClassColors[topResultIndex]
                        });
                    }
                }
            }
            return boxes;
        }

        private static ValueTuple<int, float> GetTopResult(float[] predictedClasses)
        {
            return predictedClasses
                .Select((predictedClass, index) => (Index: index, Value: predictedClass))
                .OrderByDescending(result => result.Value)
                .First();
        }

        private static float IntersectionOverUnion(RectangleF boundingBoxA, RectangleF boundingBoxB)
        {
            float areaA = boundingBoxA.Width * boundingBoxA.Height;

            if (areaA <= 0)
                return 0;

            float areaB = boundingBoxB.Width * boundingBoxB.Height;

            if (areaB <= 0)
                return 0;

            float minX = Math.Max(boundingBoxA.Left, boundingBoxB.Left);
            float minY = Math.Max(boundingBoxA.Top, boundingBoxB.Top);
            float maxX = Math.Min(boundingBoxA.Right, boundingBoxB.Right);
            float maxY = Math.Min(boundingBoxA.Bottom, boundingBoxB.Bottom);

            float intersectionArea = Math.Max(maxY - minY, 0) * Math.Max(maxX - minX, 0);

            return intersectionArea / (areaA + areaB - intersectionArea);
        }

        private static float Sigmoid(float value)
        {
            float k = (float)Math.Exp(value);
            return k / (1.0f + k);
        }

        private static float[] Softmax(float[] values)
        {
            float maxVal = values.Max();
            IEnumerable<double> exp = values.Select(v => Math.Exp(v - maxVal));
            double sumExp = exp.Sum();

            return exp.Select(v => (float)(v / sumExp)).ToArray();
        }

        private BoundingBoxDimensions ExtractBoundingBoxDimensions(float[] modelOutput, int x, int y, int channel)
        {
            return new BoundingBoxDimensions
            {
                X = modelOutput[GetOffset(x, y, channel)],
                Y = modelOutput[GetOffset(x, y, channel + 1)],
                Width = modelOutput[GetOffset(x, y, channel + 2)],
                Height = modelOutput[GetOffset(x, y, channel + 3)]
            };
        }

        private float GetConfidence(float[] modelOutput, int x, int y, int channel)
        {
            return Sigmoid(modelOutput[GetOffset(x, y, channel + 4)]);
        }

        private int GetOffset(int x, int y, int channel)
        {
            // YOLO outputs a tensor that has a shape of 125x13x13, which
            // WinML flattens into a 1D array.  To access a specific channel
            // for a given (x,y) cell position, we need to calculate an offset
            // into the array
            return (channel * ChannelStride) + (y * ColCount) + x;
        }

        private void LoadJSON()
        {
            using StreamReader r = new(JSONLocation);
            string json = r.ReadToEnd();
            List<string> items = JsonConvert.DeserializeObject<List<string>>(json);
            foreach (string c in items)
            {
                Classes.Add(c);
            }
        }

        private CellDimensions MapBoundingBoxToCell(int x, int y, int box, BoundingBoxDimensions boxDimensions)
        {
            return new CellDimensions
            {
                X = (x + Sigmoid(boxDimensions.X)) * CellWidth,
                Y = (y + Sigmoid(boxDimensions.Y)) * CellHeight,
                Width = (float)Math.Exp(boxDimensions.Width) * CellWidth * Anchors[box * 2],
                Height = (float)Math.Exp(boxDimensions.Height) * CellHeight * Anchors[box * 2 + 1],
            };
        }
    }
}