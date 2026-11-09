using System;
using System.IO;
using UnityEngine;

namespace MotionBricks
{
    /// <summary>
    /// Finds the SOMA model files. The model lives in <c>Assets/StreamingAssets/MotionBricks</c>, so it
    /// is not compressed in a build and is read from disk at runtime.
    /// </summary>
    public static class SomaModelFiles
    {
        /// <summary>Folder of the model files, relative to StreamingAssets.</summary>
        public const string StreamingAssetsFolder = "MotionBricks";

        /// <summary>The half-precision model file. It is much smaller and fast enough on a GPU.</summary>
        public const string Fp16ModelFileName = "mmm_v6_soma_fp16.onnx";

        /// <summary>The full-precision model file. It is the reference for parity checks.</summary>
        public const string Fp32ModelFileName = "mmm_v6_soma_fp32.onnx";

        /// <summary>The model manifest, which documents the tensor contract.</summary>
        public const string ManifestFileName = "soma_n1.json";

        /// <summary>Returns the absolute path of the StreamingAssets folder of the application.</summary>
        public static string StreamingAssetsPath => Application.streamingAssetsPath;

        /// <summary>Returns the absolute path of a model file inside StreamingAssets.</summary>
        public static string GetStreamingAssetsPath(string fileName)
        {
            return Path.Combine(StreamingAssetsPath, StreamingAssetsFolder, fileName);
        }

        /// <summary>
        /// Returns the path of the model file to use, preferring the half-precision model. It returns
        /// null when no model file is present.
        /// </summary>
        public static string FindModelPath(bool preferFp16 = true)
        {
            var candidates = preferFp16
                ? new[] { Fp16ModelFileName, Fp32ModelFileName }
                : new[] { Fp32ModelFileName, Fp16ModelFileName };

            foreach (var candidate in candidates)
            {
                var path = GetStreamingAssetsPath(candidate);
                if (File.Exists(path))
                {
                    return path;
                }
            }

            return null;
        }
    }
}
