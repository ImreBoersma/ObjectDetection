using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.OnnxRuntime.Tensors;
using ObjectDetection.DataStructures;
using ObjectDetection.RetinaParser;

namespace ObjectDetection
{
    public class Program
    {
        private static readonly string aassetsRelativePath = @"C:\Users\imreb\source\repos\ObjectDetection\ObjectDetection\assets\";
        private static readonly string assetsPath = GetAbsolutePath(aassetsRelativePath);
        private static readonly string imagesFolder = Path.Combine(assetsPath, "images");

        /// <summary>
        /// Mean to use for conversion see also: <see href="https://en.wikipedia.org/wiki/Mean"/>.
        /// This is specific to RetinaNET, other models might use a different mean!
        /// </summary>
        private static readonly float[] mean = new float[] { 0.485F, 0.456F, 0.406F };

        private static readonly string modelFilePath = Path.Combine(assetsPath, "Model", "retinanet-9.onnx");
        private static readonly string outputFolder = Path.Combine(imagesFolder, "output");

        /// <summary>
        /// Standard deviation to use for conversion see also: <see href="https://en.wikipedia.org/wiki/Standard_deviation"/>
        /// This is specific to RetinaNET, other models might use other std's!
        /// </summary>
        private static readonly float[] std = new float[] { 0.229F, 0.224F, 0.225F };

        /// <summary>
        /// Converts a <see cref="ImageNetData"/> image to a <see cref="Tensor{float}"/>.
        /// </summary>
        /// <param name="image">The image to be converted.</param>
        /// <returns>Tensor { N, C, W, H }</returns>
        public static Tensor<float> ConvertImageToFloatData(ImageNetData image)
        {
            // Boxing is used here to convert an ImageNetData to Bitmap
            Bitmap bitmapImage = new(image.ImagePath);
            Tensor<float> data = new DenseTensor<float>(new[] { 1, 3, bitmapImage.Width, bitmapImage.Height });

            // Iterate through all pixels in given image and convert it to RGB using the mean and std
            for (int x = 0; x < bitmapImage.Width; x++)
            {
                for (int y = 0; y < bitmapImage.Height; y++)
                {
                    Color color = bitmapImage.GetPixel(x, y);

                    float red = (color.R - mean[0] * 255) / (std[0] * 255);
                    float gre = (color.G - mean[1] * 255) / (std[1] * 255);
                    float blu = (color.B - mean[2] * 255) / (std[2] * 255);

                    data[0, 0, x, y] = red;
                    data[0, 1, x, y] = gre;
                    data[0, 2, x, y] = blu;
                }
            }
            return data;
        }

        /// <summary>
        /// Get the absolute path of given relativePath.
        /// </summary>
        /// <param name="relativePath">Relative path to convert to a absolute path.</param>
        /// <returns></returns>
        public static string GetAbsolutePath(string relativePath)
        {
            FileInfo _dataRoot = new(typeof(Program).Assembly.Location);
            string assemblyFolderPath = _dataRoot.Directory.FullName;

            string fullPath = Path.Combine(assemblyFolderPath, relativePath);

            return fullPath;
        }

        /// <summary>
        /// Draws the bounding box on the given image
        /// </summary>
        /// <param name="imageName">Which image to add the given bounding boxes to.</param>
        /// <param name="filteredBoundingBoxes">The bounding boxes to add to the given image</param>
        private static void DrawBoundingBox(string imageName, IList<RetinaBoundingBox> filteredBoundingBoxes)
        {
            Image image = Image.FromFile(Path.Combine(imagesFolder, imageName));

            int originalImageHeight = image.Height;
            int originalImageWidth = image.Width;

            foreach (RetinaBoundingBox box in filteredBoundingBoxes)
            {
                // Get Bounding Box Dimensions
                uint x = (uint)Math.Max(box.Dimensions.X, 0);
                uint y = (uint)Math.Max(box.Dimensions.Y, 0);
                uint width = (uint)Math.Min(originalImageWidth - x, box.Dimensions.Width);
                uint height = (uint)Math.Min(originalImageHeight - y, box.Dimensions.Height);

                // Resize To Image
                x = (uint)originalImageWidth * x / OnnxModelScorer.ImageNetSettings.imageWidth;
                y = (uint)originalImageHeight * y / OnnxModelScorer.ImageNetSettings.imageHeight;
                width = (uint)originalImageWidth * width / OnnxModelScorer.ImageNetSettings.imageWidth;
                height = (uint)originalImageHeight * height / OnnxModelScorer.ImageNetSettings.imageHeight;

                // Bounding Box Text
                string text = $"{box.Label} ({box.Confidence * 100:0}%)";

                using Graphics thumbnailGraphic = Graphics.FromImage(image);
                thumbnailGraphic.CompositingQuality = CompositingQuality.HighQuality;
                thumbnailGraphic.SmoothingMode = SmoothingMode.HighQuality;
                thumbnailGraphic.InterpolationMode = InterpolationMode.HighQualityBicubic;

                // Define Text Options
                Font drawFont = new("Arial", 12, FontStyle.Bold);
                SizeF size = thumbnailGraphic.MeasureString(text, drawFont);
                SolidBrush fontBrush = new(Color.Black);
                Point atPoint = new((int)x, (int)y - (int)size.Height - 1);

                // Define BoundingBox options
                Pen pen = new(box.BoxColor, 3.2f);
                SolidBrush colorBrush = new(box.BoxColor);

                // Draw text on image
                thumbnailGraphic.FillRectangle(colorBrush, (int)x, (int)(y - size.Height - 1), (int)size.Width, (int)size.Height);
                thumbnailGraphic.DrawString(text, drawFont, fontBrush, atPoint);

                // Draw bounding box on image
                thumbnailGraphic.DrawRectangle(pen, x, y, width, height);
            }

            if (!Directory.Exists(outputFolder))
            {
                Directory.CreateDirectory(outputFolder);
            }

            image.Save(Path.Combine(outputFolder, imageName));
        }

        /// <summary>
        /// Log the detected objects in given image to the console for debugging.
        /// </summary>
        /// <param name="imageName">Name of the image to log the bouding boxes for.</param>
        /// <param name="boundingBoxes">The bounding boxes detected in the given image.</param>
        private static void LogDetectedObjects(string imageName, IList<RetinaBoundingBox> boundingBoxes)
        {
            if (boundingBoxes.Count == 0)
            {
                Console.WriteLine($"There are no objects detected in the image {imageName}");
                Console.WriteLine("");
            }
            else
            {
                Console.WriteLine($".....The objects in the image {imageName} are detected as below....");

                foreach (RetinaBoundingBox box in boundingBoxes)
                {
                    Console.WriteLine($"{box.Label} and its Confidence score: {box.Confidence}");
                }
                Console.WriteLine("");
            }
        }

        private static void Main()
        {
            MLContext mlContext = new();

            try
            {
                // Load Data from image folder
                IEnumerable<ImageNetData> images = ImageNetData.ReadFromFile(imagesFolder);
                foreach (ImageNetData image in images)
                {
                    // Convert all images to float data
                    image.Tensor = ConvertImageToFloatData(image);
                }

                // Load the images into mlContext
                IDataView imageDataView = mlContext.Data.LoadFromEnumerable(images);

                OnnxModelScorer modelScorer = new(imagesFolder, modelFilePath, mlContext);

                // Use model to score data
                IEnumerable<float[]> probabilities = modelScorer.Score(imageDataView);

                // Post-process model output
                RetinaOutputParser parser = new(32, 32, 13, 13);

                // Iterates each object detection and parses it's output.
                IEnumerable<IList<RetinaBoundingBox>> boundingBoxes = probabilities
                    .Select(probability => parser.ParseOutputs(probability))
                    .Select(boxes => RetinaOutputParser.FilterBoundingBoxes(boxes, 5, .5F));

                // Draw bounding box for each detected object in each of the given image
                for (int i = 0; i < images.Count(); i++)
                {
                    string imageFileName = images.ElementAt(i).Label;
                    IList<RetinaBoundingBox> detectedObjects = boundingBoxes.ElementAt(i);
                    DrawBoundingBox(imageFileName, detectedObjects);
                    LogDetectedObjects(imageFileName, detectedObjects);
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.ToString());
            }
            finally
            {
                Console.WriteLine("========= End of Process..Hit any Key ========");
                Console.ReadKey();
            }
        }
    }
}