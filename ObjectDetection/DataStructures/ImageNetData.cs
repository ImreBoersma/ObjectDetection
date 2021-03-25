using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace ObjectDetection.DataStructures
{
    /// <summary>
    /// This is the input model of the ONNX-model.
    /// An instance of this class needs to be converted to the correct input format for the ONNX-model.
    /// </summary>
    public class ImageNetData
    {
        /// <summary>
        /// Path to the image file.
        /// </summary>
        [LoadColumn(0)]
        public string ImagePath;

        /// <summary>
        /// The name of the image.
        /// </summary>
        [LoadColumn(1)]
        public string Label;

        [NoColumn]
        public Tensor<float> Tensor;

        /// <summary>
        /// Gets all files in given directory.
        /// </summary>
        /// <param name="imageFolder">The directory to search in.</param>
        /// <returns><see cref="IEnumerable{ImageNetData}"/>  of the input model.</returns>
        public static IEnumerable<ImageNetData> ReadFromFile(string imageFolder) => Directory
            .GetFiles(imageFolder)
            .Select(filePath => new ImageNetData { ImagePath = filePath, Label = Path.GetFileName(filePath) });
    }
}