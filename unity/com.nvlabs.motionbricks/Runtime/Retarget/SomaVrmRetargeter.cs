using System;
using System.Collections.Generic;
using UnityEngine;

namespace MotionBricks
{
    /// <summary>
    /// One humanoid bone of the avatar and the SOMA joints that drive it.
    /// </summary>
    public struct SomaBoneBinding
    {
        /// <summary>The Unity humanoid bone.</summary>
        public HumanBodyBones Bone;

        /// <summary>
        /// The SOMA joints of the chain, from the parent's chain end to this bone's chain end. The last
        /// joint carries the bone's rotation.
        /// </summary>
        public int[] SomaJoints;
    }

    /// <summary>
    /// Maps the SOMA model's joints onto the humanoid bones of a Unity avatar. This covers VRM
    /// avatars in their VRM 0.x and VRM 1.0 forms, because UniVRM imports both as Unity humanoids.
    /// The two forms differ in their spine: VRM 0.x has no upper chest bone, so the mapping merges
    /// the SOMA chest into the chest bone when the avatar lacks one.
    /// </summary>
    public static class SomaHumanoidMap
    {
        /// <summary>The SOMA joints that drive each humanoid bone, in the order the bones are written.</summary>
        public static IEnumerable<SomaBoneBinding> CreateBindings(SomaSkeleton skeleton)
        {
            var bindings = new List<SomaBoneBinding>
            {
                Bind(skeleton, HumanBodyBones.Hips, "Hips"),
                Bind(skeleton, HumanBodyBones.Spine, "Spine1"),
                Bind(skeleton, HumanBodyBones.Chest, "Spine2"),
                Bind(skeleton, HumanBodyBones.UpperChest, "Chest"),
                Bind(skeleton, HumanBodyBones.Neck, "Neck1"),
                Bind(skeleton, HumanBodyBones.Head, "Neck2", "Head"),
                Bind(skeleton, HumanBodyBones.LeftShoulder, "LeftShoulder"),
                Bind(skeleton, HumanBodyBones.LeftUpperArm, "LeftArm"),
                Bind(skeleton, HumanBodyBones.LeftLowerArm, "LeftForeArm"),
                Bind(skeleton, HumanBodyBones.LeftHand, "LeftHand"),
                Bind(skeleton, HumanBodyBones.RightShoulder, "RightShoulder"),
                Bind(skeleton, HumanBodyBones.RightUpperArm, "RightArm"),
                Bind(skeleton, HumanBodyBones.RightLowerArm, "RightForeArm"),
                Bind(skeleton, HumanBodyBones.RightHand, "RightHand"),
                Bind(skeleton, HumanBodyBones.LeftUpperLeg, "LeftLeg"),
                Bind(skeleton, HumanBodyBones.LeftLowerLeg, "LeftShin"),
                Bind(skeleton, HumanBodyBones.LeftFoot, "LeftFoot"),
                Bind(skeleton, HumanBodyBones.LeftToes, "LeftToeBase"),
                Bind(skeleton, HumanBodyBones.RightUpperLeg, "RightLeg"),
                Bind(skeleton, HumanBodyBones.RightLowerLeg, "RightShin"),
                Bind(skeleton, HumanBodyBones.RightFoot, "RightFoot"),
                Bind(skeleton, HumanBodyBones.RightToes, "RightToeBase"),
            };

            return bindings;
        }

        private static SomaBoneBinding Bind(SomaSkeleton skeleton, HumanBodyBones bone, params string[] jointNames)
        {
            var joints = new int[jointNames.Length];
            for (var index = 0; index < jointNames.Length; index++)
            {
                var joint = skeleton.JointIndex(jointNames[index]);
                if (joint < 0)
                {
                    throw new ArgumentException($"The skeleton is missing the joint '{jointNames[index]}'.");
                }

                joints[index] = joint;
            }

            return new SomaBoneBinding { Bone = bone, SomaJoints = joints };
        }

        /// <summary>
        /// Keeps only the bindings whose bone the avatar has, and merges the SOMA joints of a missing
        /// bone into the next bone that exists.
        /// </summary>
        public static List<SomaBoneBinding> ResolveForAvatar(
            SomaSkeleton skeleton,
            Animator avatar,
            out List<HumanBodyBones> bones,
            out List<int> chainEnds)
        {
            bones = new List<HumanBodyBones>();
            chainEnds = new List<int>();
            var resolved = new List<SomaBoneBinding>();
            var pending = new List<int>();

            foreach (var binding in CreateBindings(skeleton))
            {
                if (avatar.GetBoneTransform(binding.Bone) == null)
                {
                    // The avatar has no such bone, so its SOMA joints drive the next bone instead.
                    pending.AddRange(binding.SomaJoints);
                    continue;
                }

                var chain = new List<int>(pending);
                chain.AddRange(binding.SomaJoints);
                pending.Clear();

                resolved.Add(new SomaBoneBinding { Bone = binding.Bone, SomaJoints = chain.ToArray() });
                bones.Add(binding.Bone);
                chainEnds.Add(chain[chain.Count - 1]);
            }

            return resolved;
        }
    }

    /// <summary>
    /// Drives a Unity humanoid avatar with SOMA motion. The generated SOMA poses carry local joint
    /// transforms; this component maps them onto the avatar's humanoid bones with a per-bone
    /// alignment measured once from the avatar's rest pose and the SOMA reference pose.
    /// </summary>
    /// <remarks>
    /// The avatar must hold its rest pose when the retargeter initializes, so the alignment is
    /// measured correctly. A VRM avatar without an Animator Controller, or with one that is disabled,
    /// keeps its rest pose.
    /// </remarks>
    [AddComponentMenu("MotionBricks/SOMA VRM Retargeter")]
    [RequireComponent(typeof(Animator))]
    [DefaultExecutionOrder(10000)]
    public sealed class SomaVrmRetargeter : MonoBehaviour
    {
        [Tooltip("The avatar to drive. Its Animator must have a humanoid avatar.")]
        [SerializeField] private Animator _avatar;

        [Tooltip("Scale of the root motion. 1 matches the SOMA skeleton's size.")]
        [SerializeField] private float _rootScale = 1f;

        [Tooltip("Draw the SOMA joint positions while playing.")]
        [SerializeField] private bool _drawDebugJoints;

        private SomaSkeleton _skeleton;
        private Transform[] _boneTransforms = Array.Empty<Transform>();
        private Quaternion[] _alignment = Array.Empty<Quaternion>();
        private int[] _chainEnd = Array.Empty<int>();
        private Pose[] _somaWorld = Array.Empty<Pose>();
        private Vector3 _anchorPosition;
        private Vector3 _anchorSomaPosition;
        private float _heightScale = 1f;
        private bool _initialized;
        private Pose[] _lastSomaPose;

        /// <summary>The SOMA skeleton this retargeter uses.</summary>
        public SomaSkeleton Skeleton => _skeleton;

        /// <summary>True once the retargeter measured its alignment and can drive the avatar.</summary>
        public bool IsInitialized => _initialized;

        private void Awake()
        {
            Initialize();
        }

        /// <summary>
        /// Prepares the retargeter. It measures the alignment between the avatar's rest pose and the
        /// SOMA reference pose, so call it while the avatar still holds its rest pose.
        /// </summary>
        public void Initialize()
        {
            if (_initialized)
            {
                return;
            }

            if (_avatar == null)
            {
                _avatar = GetComponent<Animator>();
            }

            if (_avatar == null)
            {
                Debug.LogError("[MotionBricks] The retargeter needs an Animator.", this);
                return;
            }

            if (_avatar.avatar == null || !_avatar.avatar.isValid || !_avatar.avatar.isHuman)
            {
                Debug.LogError(
                    "[MotionBricks] The retargeter needs a humanoid avatar. VRM avatars are imported as humanoids by UniVRM.",
                    this);
                return;
            }

            _skeleton = SomaSkeleton.CreateDefault();
            var numJoints = _skeleton.NumJoints;
            _somaWorld = new Pose[numJoints];

            var somaRestWorld = new Pose[numJoints];
            _skeleton.ComputeReferenceWorld(somaRestWorld);

            var bindings = SomaHumanoidMap.ResolveForAvatar(
                _skeleton, _avatar, out var bones, out var chainEnds);

            var count = bindings.Count;
            _boneTransforms = new Transform[count];
            _alignment = new Quaternion[count];
            _chainEnd = chainEnds.ToArray();

            for (var index = 0; index < count; index++)
            {
                var transform = _avatar.GetBoneTransform(bones[index]);
                _boneTransforms[index] = transform;

                // The alignment maps the SOMA world rotation onto the avatar's world rotation, measured
                // from the two rest poses.
                _alignment[index] = MbPose.Normalize(
                    transform.rotation * MbPose.Conjugate(somaRestWorld[chainEnds[index]].rotation));
            }

            // The avatar's hips height sets the scale of the root motion.
            var hips = _avatar.GetBoneTransform(HumanBodyBones.Hips);
            var somaHips = somaRestWorld[_skeleton.JointIndex("Hips")].position;
            if (hips != null && Mathf.Abs(somaHips.y) > 1e-4f)
            {
                _heightScale = Mathf.Abs(hips.position.y) / Mathf.Abs(somaHips.y);
            }

            DisableAnimationRigging();
            DisableSkinnedMeshCulling();
            _initialized = true;
        }

        /// <summary>
        /// Keeps skinned meshes rendered wherever the generated motion takes the avatar. Skinned
        /// meshes otherwise cull against their import-time bounds, which go stale once the root
        /// motion moves the character far from its spawn point.
        /// </summary>
        private void DisableSkinnedMeshCulling()
        {
            foreach (var renderer in _avatar.GetComponentsInChildren<SkinnedMeshRenderer>(true))
            {
                renderer.updateWhenOffscreen = true;
            }
        }

        /// <summary>
        /// Disables Animation Rigging evaluation on the avatar. Rig builders run after Update and
        /// overwrite the humanoid bones every frame, which fights a directly driven skeleton. The
        /// component is found by name so this package does not depend on the rigging package.
        /// </summary>
        private void DisableAnimationRigging()
        {
            foreach (var behaviour in _avatar.GetComponentsInChildren<MonoBehaviour>(true))
            {
                if (behaviour == null || !behaviour.enabled)
                {
                    continue;
                }

                if (behaviour.GetType().FullName == "UnityEngine.Animations.Rigging.RigBuilder")
                {
                    behaviour.enabled = false;
                    Debug.Log(
                        $"[MotionBricks] Disabled Animation Rigging on '{behaviour.name}' so the retargeter " +
                        "keeps control of the humanoid bones. Re-enable it if you want rig constraints on " +
                        "top of the generated motion.", behaviour);
                }
            }
        }

        private void LateUpdate()
        {
            // Re-apply after every other LateUpdate (Animator, spring bones) so the generated pose is
            // what renders.
            if (_initialized && _lastSomaPose != null)
            {
                ApplyPose(_lastSomaPose);
            }
        }

        /// <summary>
        /// Sets the point where the avatar's hips sit and the SOMA root position that belongs there.
        /// Call it when the character is placed, so the generated motion starts from the avatar's own
        /// position.
        /// </summary>
        public void SetAnchor(Vector3 avatarWorldPosition, Vector3 somaRootPosition)
        {
            Initialize();
            _anchorPosition = avatarWorldPosition;
            _anchorSomaPosition = somaRootPosition;
        }

        /// <summary>
        /// Applies one SOMA pose to the avatar. The pose holds the local joint transforms of the SOMA
        /// skeleton, with the root at index zero.
        /// </summary>
        public void ApplyPose(Pose[] somaLocalPose)
        {
            if (!_initialized)
            {
                Initialize();
                if (!_initialized)
                {
                    return;
                }
            }

            var numJoints = _skeleton.NumJoints;
            MbPose.ComposeWorldPoses(
                somaLocalPose, _skeleton.ParentIndices, _somaWorld, new Pose(Vector3.zero, Quaternion.identity));
            _lastSomaPose = somaLocalPose;

            // The root moves the avatar. Its rotation comes from the SOMA root, aligned to the avatar's
            // rest pose.
            var rootWorld = _somaWorld[0];
            var rootRotation = MbPose.Normalize(_alignment[0] * rootWorld.rotation);
            var hips = _boneTransforms[0];
            var parent = hips.parent;
            var parentRotation = parent != null ? parent.rotation : Quaternion.identity;
            var rootPosition = _anchorPosition +
                               (rootWorld.position - _anchorSomaPosition) * (_heightScale * _rootScale);
            hips.SetPositionAndRotation(rootPosition, MbPose.Normalize(MbPose.Conjugate(parentRotation) * rootRotation));

            // The other bones follow the hierarchy, so each bone's local rotation is measured against
            // the world rotation its parent received.
            for (var index = 1; index < _boneTransforms.Length; index++)
            {
                var transform = _boneTransforms[index];
                var world = MbPose.Normalize(_alignment[index] * _somaWorld[_chainEnd[index]].rotation);

                var boneParent = transform.parent;
                var boneParentRotation = boneParent != null ? boneParent.rotation : Quaternion.identity;
                transform.localRotation = MbPose.Normalize(MbPose.Conjugate(boneParentRotation) * world);
            }
        }

        /// <summary>
        /// Returns true when the model predicts the joint, so it keeps its reference pose otherwise.
        /// </summary>
        public bool IsModelJoint(int somaJoint)
        {
            return _skeleton != null && _skeleton.PoseToModel[somaJoint] >= 0;
        }

        private void OnDrawGizmosSelected()
        {
            if (!_drawDebugJoints || !_initialized)
            {
                return;
            }

            Gizmos.color = new Color(0.2f, 0.9f, 1f, 0.9f);
            for (var joint = 0; joint < _somaWorld.Length; joint++)
            {
                if (_skeleton.PoseToModel[joint] < 0)
                {
                    continue;
                }

                var position = _somaWorld[joint].position;
                Gizmos.DrawWireSphere(position, 0.02f);
                var parent = _skeleton.JointParent(joint);
                if (parent >= 0)
                {
                    Gizmos.DrawLine(position, _somaWorld[parent].position);
                }
            }
        }
    }
}
