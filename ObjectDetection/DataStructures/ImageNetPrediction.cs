using Microsoft.ML.Data;

namespace ObjectDetection.DataStructures
{
    /// <summary>
    /// This is the model for the output of RetinaNet
    /// </summary>
    public class ImageNetPrediction
    {
        [ColumnName("output1")]
        public float[] Output1;

        [ColumnName("output10")]
        public float[] Output10;

        [ColumnName("output2")]
        public float[] Output2;

        [ColumnName("output3")]
        public float[] Output3;

        [ColumnName("output4")]
        public float[] Output4;

        [ColumnName("output5")]
        public float[] Output5;

        [ColumnName("output6")]
        public float[] Output6;

        [ColumnName("output7")]
        public float[] Output7;

        [ColumnName("output8")]
        public float[] Output8;

        [ColumnName("output9")]
        public float[] Output9;
    }
}