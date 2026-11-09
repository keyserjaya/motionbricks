using UnityEngine;

namespace MotionBricks
{
    /// <summary>
    /// The affine map between SOMA motion space and the Unity scene that the retargeter applies:
    /// <c>scene = anchor + rotation * (soma - somaAnchor) * scale</c>. The rotation aligns the
    /// generated trajectory to the avatar's placement and the scale applies to root travel.
    /// </summary>
    /// <remarks>
    /// A seed at reference hip height, with root scale one, maps the SOMA ground onto the avatar's
    /// ground height. Other seed heights or root travel scales can shift the mapped ground plane.
    /// </remarks>
    public readonly struct SomaSceneMap
    {
        /// <summary>Maps identity: scene point = soma point.</summary>
        public static SomaSceneMap Identity { get; } = new SomaSceneMap(Vector3.zero, Vector3.zero, 1f);

        /// <summary>Uniform scale between the two spaces, in scene meters per SOMA meter.</summary>
        public float Scale { get; }

        /// <summary>The scene point at which the SOMA space is anchored.</summary>
        public Vector3 Anchor { get; }

        /// <summary>The SOMA point at which the SOMA space is anchored.</summary>
        public Vector3 AnchorSoma { get; }

        /// <summary>Rotation from SOMA motion space into scene space.</summary>
        public Quaternion Rotation { get; }

        private readonly float _inverseScale;

        /// <summary>
        /// Creates the map. <paramref name="anchor"/> is the scene point where
        /// <paramref name="anchorSoma"/> sits; <paramref name="scale"/> must be positive.
        /// </summary>
        public SomaSceneMap(Vector3 anchor, Vector3 anchorSoma, float scale)
            : this(anchor, anchorSoma, scale, Quaternion.identity)
        {
        }

        /// <summary>Creates a scene map with an explicit source-to-scene rotation.</summary>
        public SomaSceneMap(Vector3 anchor, Vector3 anchorSoma, float scale, Quaternion rotation)
        {
            if (scale <= 1e-6f || float.IsNaN(scale) || float.IsInfinity(scale))
            {
                throw new System.ArgumentOutOfRangeException(
                    nameof(scale), scale, "The scene map scale must be finite and positive.");
            }

            var lengthSquared = rotation.x * rotation.x + rotation.y * rotation.y +
                                rotation.z * rotation.z + rotation.w * rotation.w;
            if (float.IsNaN(lengthSquared) || float.IsInfinity(lengthSquared) || lengthSquared < 1e-8f)
                throw new System.ArgumentException("The scene rotation must be finite and nonzero.", nameof(rotation));

            Anchor = anchor;
            AnchorSoma = anchorSoma;
            Scale = scale;
            Rotation = Quaternion.Normalize(rotation);
            _inverseScale = 1f / scale;
        }

        /// <summary>Maps a SOMA motion-space point into the scene.</summary>
        public Vector3 SomaPointToScene(Vector3 somaPoint)
        {
            return Anchor + Rotation * (somaPoint - AnchorSoma) * Scale;
        }

        /// <summary>Maps a scene point into SOMA motion space.</summary>
        public Vector3 ScenePointToSoma(Vector3 scenePoint)
        {
            return AnchorSoma + Quaternion.Inverse(Rotation) * (scenePoint - Anchor) * _inverseScale;
        }

        /// <summary>Rotates a direction into the scene without applying translation or scale.</summary>
        public Vector3 SomaDirectionToScene(Vector3 direction) => Rotation * direction;

        /// <summary>Rotates a scene direction back into SOMA space.</summary>
        public Vector3 SceneDirectionToSoma(Vector3 direction) => Quaternion.Inverse(Rotation) * direction;

        /// <summary>Maps a joint relative to a moving root, independently of root travel scale.</summary>
        public Vector3 SomaJointToScene(Vector3 somaRoot, Vector3 somaJoint, float bodyScale)
        {
            return SomaPointToScene(somaRoot) + SomaDirectionToScene(somaJoint - somaRoot) * bodyScale;
        }

        /// <summary>Maps a scene contact into the SOMA skeleton around the specified root.</summary>
        public Vector3 ScenePointToSomaJoint(Vector3 somaRoot, Vector3 scenePoint, float bodyScale)
        {
            if (bodyScale <= 1e-6f || float.IsNaN(bodyScale) || float.IsInfinity(bodyScale))
                throw new System.ArgumentOutOfRangeException(nameof(bodyScale));
            return somaRoot + SceneDirectionToSoma(scenePoint - SomaPointToScene(somaRoot)) / bodyScale;
        }
    }
}
