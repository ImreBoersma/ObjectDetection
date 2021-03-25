using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Onnx;
using ObjectDetection.DataStructures;
using static Microsoft.ML.DataViewSchema;

namespace ObjectDetection
{
    internal class OnnxModelScorer
    {
        private readonly string ImagesFolder;
        private readonly MLContext MLContext;
        private readonly string ModelLocation;

        /// <summary>
        /// Creates an instance of <see cref="OnnxModelScorer"/>.
        /// </summary>
        /// <param name="imagesFolder">Image folder to search for files for debugging purposes.</param>
        /// <param name="modelLocation">The location of the model to use for identifying objects.</param>
        /// <param name="mlContext"><see cref="Microsoft.ML.MLContext"/> to operate in.</param>
        public OnnxModelScorer(string imagesFolder, string modelLocation, MLContext mlContext)
        {
            ImagesFolder = imagesFolder;
            ModelLocation = modelLocation;
            MLContext = mlContext;
        }

        /// <summary>
        /// Using the given <see cref="IDataView"/> to identify objects in the using the given model
        /// </summary>
        /// <param name="data">All images in <see cref="IDataView"/> format.</param>
        /// <returns><see cref="float[]"/> which is the output of model.</returns>
        public IEnumerable<float[]> Score(IDataView data) => PredictDataUsingModel(data, LoadModel());

        /// <summary>
        /// Loads the model.
        /// </summary>
        /// <returns>Returns the loaded model</returns>
        private ITransformer LoadModel()
        {
            Console.WriteLine($"Model location: {ModelLocation}");
            Console.WriteLine($"Default parameters: image size=({ImageNetSettings.imageWidth},{ImageNetSettings.imageHeight})");

            // ML.NET pipelines need to know what data schema to operate on
            IDataView data = MLContext.Data.LoadFromEnumerable(new List<ImageNetData>());

            // .LoadImages loads images as Bitmaps
            EstimatorChain<OnnxTransformer> pipeline = MLContext.Transforms.LoadImages(outputColumnName: RetinaModelSettings.ModelInput, imageFolder: "", inputColumnName: nameof(ImageNetData.ImagePath))
                // .ResizeImages rescales the image to the size specified
                .Append(MLContext.Transforms.ResizeImages(outputColumnName: RetinaModelSettings.ModelInput, imageWidth: ImageNetSettings.imageWidth, imageHeight: ImageNetSettings.imageHeight))
                // .ExtractPixels Changes the pixel representation of the images from Bitmap to vector
                .Append(MLContext.Transforms.ExtractPixels(outputColumnName: RetinaModelSettings.ModelInput))
                // .ApplyOnnxModel loads the ONNX model and uses it to score on the data provided
                // TODO: Change the outputColumnNames
                .Append(MLContext.Transforms.ApplyOnnxModel(modelFile: ModelLocation, outputColumnNames: RetinaModelSettings.ModelOutput, inputColumnNames: new[] { RetinaModelSettings.ModelInput }));

            // Executing the above EstimatorChain
            TransformerChain<OnnxTransformer> model = pipeline.Fit(data);

            return model;
        }

        private IEnumerable<float[]> PredictDataUsingModel(IDataView testData, ITransformer model)
        {
            Console.WriteLine($"Images location: {ImagesFolder}");
            Console.WriteLine("");
            Console.WriteLine("===== Identify the objects in the images =====");
            Console.WriteLine("");

            // Score the data with Transform method
            IDataView scoredData = model.Transform(testData);

            DataViewSchema scoredDataSchema = scoredData.Schema;

            // Iterate through all rows in the IDataView
            foreach (Column item in scoredDataSchema)
            {
                // Do something with the bounding boxes.
                if (RetinaModelSettings.ModelBoxHeads.Contains(item.Name))
                {
                    Console.WriteLine($"BoxHead {item.Name} is Type: {item.Type}");
                }
                // Do something with the detected classes.
                else if (RetinaModelSettings.ModelClassHeads.Contains(item.Name))
                {
                    Console.WriteLine($"ClassHead {item.Name} is Type: {item.Type}");
                }
            }

            //for (int i = 0; i < RetinaModelSettings.ModelOutputs.Length; i++)
            //{
            //    (string classHead, string boxHead) = RetinaModelSettings.ModelOutputs.ElementAt(i);
            //    Console.WriteLine($"Classhead: {classHead}");
            //    Console.WriteLine($"BoxHead: {boxHead}");
            //    Console.WriteLine($"ModelOutput: {RetinaModelSettings.ModelOutputs.GetValue(i)}");
            //}

            // TODO: Iterate over outputs to get probabilites.
            // Extract predicted probabilities from IDataView scoredData
            IEnumerable<float[]> probabilities = scoredData.GetColumn<float[]>(RetinaModelSettings.ModelOutput[0]);
            return probabilities;
        }

        public struct ImageNetSettings
        {
            public const int imageHeight = 480;
            public const int imageWidth = 640;
        }

        public struct RetinaModelSettings
        {
            public const string ModelInput = "input";
            public readonly static string[] ModelBoxHeads = new string[] { "output6", "output7", "output8", "output9", "output10" };
            public readonly static string[] ModelClassHeads = new string[] { "output1", "output2", "output3", "output4", "output5" };
            public readonly static string[] ModelOutput = { "output1", "output2", "output3", "output4", "output5", "output6", "output7", "output8", "output9", "output10" };
            public readonly static (string clsHead, string boxHead)[] ModelOutputs = ModelClassHeads.Zip(ModelBoxHeads).ToArray();
        }
    }
}