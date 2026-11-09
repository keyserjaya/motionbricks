using System.IO;
using UnityEditor;
using UnityEngine;

namespace MotionBricks.Editor
{
    /// <summary>
    /// Checks that the model files are in place and offers to copy them from the MotionBricks
    /// repository. The model is large, so it stays out of the version control of the host project.
    /// </summary>
    public static class SomaModelSetup
    {
        private const string MenuRoot = "Tools/MotionBricks/";

        [MenuItem(MenuRoot + "Check Model Files")]
        public static void CheckModelFiles()
        {
            var found = false;
            foreach (var fileName in new[] { SomaModelFiles.Fp16ModelFileName, SomaModelFiles.Fp32ModelFileName })
            {
                var path = SomaModelFiles.GetStreamingAssetsPath(fileName);
                if (!File.Exists(path))
                {
                    continue;
                }

                found = true;
                var size = new FileInfo(path).Length / (1024f * 1024f);
                Debug.Log($"[MotionBricks] {fileName}: {size:0} MB at {path}");
            }

            var manifest = SomaModelFiles.GetStreamingAssetsPath(SomaModelFiles.ManifestFileName);
            if (File.Exists(manifest))
            {
                Debug.Log($"[MotionBricks] Manifest: {manifest}");
            }

            if (!found)
            {
                Debug.LogWarning(
                    "[MotionBricks] No model file was found in " +
                    $"{SomaModelFiles.GetStreamingAssetsPath(string.Empty)}. " +
                    "Copy mmm_v6_soma_fp16.onnx from the MotionBricks repository (motionbricks/assets/onnx/soma) " +
                    "into that folder, or use Tools/MotionBricks/Copy Model From Repository.");
            }
        }

        [MenuItem(MenuRoot + "Copy Model From Repository")]
        public static void CopyModelFromRepository()
        {
            var source = EditorUtility.OpenFolderPanel(
                "Select the MotionBricks ONNX folder", "", "");
            if (string.IsNullOrEmpty(source))
            {
                return;
            }

            var destination = SomaModelFiles.GetStreamingAssetsPath(string.Empty);
            Directory.CreateDirectory(destination);

            var copied = 0;
            foreach (var fileName in new[] { SomaModelFiles.Fp16ModelFileName, SomaModelFiles.ManifestFileName })
            {
                var sourceFile = Path.Combine(source, fileName);
                if (!File.Exists(sourceFile))
                {
                    continue;
                }

                File.Copy(sourceFile, Path.Combine(destination, fileName), true);
                copied++;
            }

            AssetDatabase.Refresh();
            Debug.Log($"[MotionBricks] Copied {copied} file(s) to {destination}.");
        }
    }
}
