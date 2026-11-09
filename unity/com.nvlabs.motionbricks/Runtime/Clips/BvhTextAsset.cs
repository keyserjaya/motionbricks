using UnityEngine;

namespace MotionBricks
{
    /// <summary>
    /// Holds the text of a BVH motion file. The package's editor assembly imports <c>.bvh</c> files
    /// into this asset type, so clips can be referenced from scenes and builds.
    /// </summary>
    public sealed class BvhTextAsset : ScriptableObject
    {
        [TextArea(4, 12)]
        public string Text;
    }
}
