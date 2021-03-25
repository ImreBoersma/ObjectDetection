using System.Drawing;

namespace ObjectDetection.RetinaParser
{
    public class BoundingBoxDimensions : DimensionsBase { }

    /// <summary>
    /// This is the class for the rendered bounding boxes.
    /// </summary>
    public class RetinaBoundingBox
    {
        /// <summary>
        /// Bounding box color.
        /// </summary>
        public Color BoxColor { get; set; }

        /// <summary>
        /// The confidence of an object within the bounding box.
        /// </summary>
        public float Confidence { get; set; }

        /// <summary>
        /// The dimensions of the bounding box.
        /// </summary>
        public BoundingBoxDimensions Dimensions { get; set; }

        /// <summary>
        /// The label of the identified object within the bounding box.
        /// </summary>
        public string Label { get; set; }

        /// <summary>
        /// The actual bounding box.
        /// </summary>
        public RectangleF Rect => new(Dimensions.X, Dimensions.Y, Dimensions.Width, Dimensions.Height);
    }
}