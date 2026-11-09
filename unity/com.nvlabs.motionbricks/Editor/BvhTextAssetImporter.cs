using System.IO;
using UnityEditor.AssetImporters;

namespace MotionBricks.Editor
{
    /// <summary>
    /// Imports BVH motion files as <see cref="BvhTextAsset"/>. The custom <c>.mbvh</c> extension keeps
    /// the files away from other BVH importers in the host project, so clip references work in scenes
    /// and builds like other assets.
    /// </summary>
    [ScriptedImporter(1, "mbvh")]
    public sealed class BvhTextAssetImporter : ScriptedImporter
    {
        public override void OnImportAsset(AssetImportContext context)
        {
            var asset = UnityEngine.ScriptableObject.CreateInstance<BvhTextAsset>();
            asset.Text = File.ReadAllText(context.assetPath);
            asset.name = System.IO.Path.GetFileNameWithoutExtension(context.assetPath);
            context.AddObjectToAsset("main", asset);
            context.SetMainObject(asset);
        }
    }
}
