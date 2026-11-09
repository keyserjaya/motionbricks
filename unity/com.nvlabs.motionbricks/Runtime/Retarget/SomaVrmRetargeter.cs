using System;
using System.Collections.Generic;
using System.Text;
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
    /// </summary>
    /// <remarks>
    /// When the avatar is missing a bone (a VRM 0.x torso has no upper chest), their SOMA joint is
    /// carried onto the next existing binding of the chain. Because each bone is driven by a single
    /// chain-end rotation — a world rotation that already includes every ancestor — the carried
    /// joint is NOT applied anywhere on its own. It only folds the missing joint's rotation into the
    /// world rotation of the next chain end. So a missing upper chest means the SOMA chest rotation
    /// reaches the avatar folded into its neck/head rotation, while the chest bone itself keeps the
    /// spine's rotation.
    /// </remarks>
    public static class SomaHumanoidMap
    {
        /// <summary>The SOMA joints that drive each humanoid bone, in the order the bones are written.</summary>
        public static IEnumerable<SomaBoneBinding> CreateBindings(SomaSkeleton skeleton)
        {
            return CreateBindings(skeleton, false);
        }

        /// <summary>
        /// The SOMA joints that drive each humanoid bone, optionally cross-binding the sides. The
        /// default SOMA skeleton's left/right names are mirrored relative to its geometry (its toes
        /// face +Z while its "Left" joints sit at +X, the +Z-facing character's right), so the
        /// avatar's left bones must read the skeleton's "Right" joints and vice versa.
        /// </summary>
        public static IEnumerable<SomaBoneBinding> CreateBindings(SomaSkeleton skeleton, bool swapLeftRight)
        {
            var bindings = new List<SomaBoneBinding>
            {
                Bind(skeleton, HumanBodyBones.Hips, "Hips"),
                Bind(skeleton, HumanBodyBones.Spine, "Spine1"),
                Bind(skeleton, HumanBodyBones.Chest, "Spine2"),
                Bind(skeleton, HumanBodyBones.UpperChest, "Chest"),
                Bind(skeleton, HumanBodyBones.Neck, "Neck1"),
                Bind(skeleton, HumanBodyBones.Head, "Neck2", "Head"),
                Bind(skeleton, HumanBodyBones.LeftShoulder, Side("LeftShoulder", swapLeftRight)),
                Bind(skeleton, HumanBodyBones.LeftUpperArm, Side("LeftArm", swapLeftRight)),
                Bind(skeleton, HumanBodyBones.LeftLowerArm, Side("LeftForeArm", swapLeftRight)),
                Bind(skeleton, HumanBodyBones.LeftHand, Side("LeftHand", swapLeftRight)),
                Bind(skeleton, HumanBodyBones.RightShoulder, Side("RightShoulder", swapLeftRight)),
                Bind(skeleton, HumanBodyBones.RightUpperArm, Side("RightArm", swapLeftRight)),
                Bind(skeleton, HumanBodyBones.RightLowerArm, Side("RightForeArm", swapLeftRight)),
                Bind(skeleton, HumanBodyBones.RightHand, Side("RightHand", swapLeftRight)),
                Bind(skeleton, HumanBodyBones.LeftUpperLeg, Side("LeftLeg", swapLeftRight)),
                Bind(skeleton, HumanBodyBones.LeftLowerLeg, Side("LeftShin", swapLeftRight)),
                Bind(skeleton, HumanBodyBones.LeftFoot, Side("LeftFoot", swapLeftRight)),
                Bind(skeleton, HumanBodyBones.LeftToes, Side("LeftToeBase", swapLeftRight)),
                Bind(skeleton, HumanBodyBones.RightUpperLeg, Side("RightLeg", swapLeftRight)),
                Bind(skeleton, HumanBodyBones.RightLowerLeg, Side("RightShin", swapLeftRight)),
                Bind(skeleton, HumanBodyBones.RightFoot, Side("RightFoot", swapLeftRight)),
                Bind(skeleton, HumanBodyBones.RightToes, Side("RightToeBase", swapLeftRight)),
            };

            return bindings;
        }

        /// <summary>Swaps the Left/Right prefix of a joint name when cross-binding.</summary>
        private static string Side(string jointName, bool swapLeftRight)
        {
            if (!swapLeftRight)
            {
                return jointName;
            }

            if (jointName.StartsWith("Left"))
            {
                return "Right" + jointName.Substring(4);
            }

            if (jointName.StartsWith("Right"))
            {
                return "Left" + jointName.Substring(5);
            }

            return jointName;
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
            return ResolveForAvatar(skeleton, avatar, false, out bones, out chainEnds, out _);
        }

        /// <summary>Resolves bindings and reports humanoid bones not present on the avatar.</summary>
        public static List<SomaBoneBinding> ResolveForAvatar(
            SomaSkeleton skeleton,
            Animator avatar,
            out List<HumanBodyBones> bones,
            out List<int> chainEnds,
            out List<HumanBodyBones> missingBones)
        {
            return ResolveForAvatar(skeleton, avatar, false, out bones, out chainEnds, out missingBones);
        }

        /// <summary>
        /// Resolves bindings with optional left/right cross-binding and reports humanoid bones not
        /// present on the avatar.
        /// </summary>
        public static List<SomaBoneBinding> ResolveForAvatar(
            SomaSkeleton skeleton,
            Animator avatar,
            bool swapLeftRight,
            out List<HumanBodyBones> bones,
            out List<int> chainEnds,
            out List<HumanBodyBones> missingBones)
        {
            bones = new List<HumanBodyBones>();
            chainEnds = new List<int>();
            missingBones = new List<HumanBodyBones>();
            var resolved = new List<SomaBoneBinding>();
            var pending = new List<int>();

            foreach (var binding in CreateBindings(skeleton, swapLeftRight))
            {
                if (avatar.GetBoneTransform(binding.Bone) == null)
                {
                    // The avatar has no such bone, so its SOMA joints drive the next bone instead.
                    missingBones.Add(binding.Bone);
                    pending.AddRange(binding.SomaJoints);
                    continue;
                }

                var chain = new List<int>();
                foreach (var joint in pending)
                {
                    for (var ancestor = skeleton.JointParent(binding.SomaJoints[0]); ancestor >= 0;
                         ancestor = skeleton.JointParent(ancestor))
                    {
                        if (ancestor != joint) continue;
                        chain.Add(joint);
                        break;
                    }
                }
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
    /// Debug draw space for the SOMA joint gizmos.
    /// </summary>
    public enum SomaDebugSpace
    {
        /// <summary>
        /// Draw the SOMA joints where the avatar is: anchored at the retargeter's anchor and scaled
        /// like the root motion. The SOMA hips then sit on the avatar's hips.
        /// </summary>
        AvatarAnchored = 0,

        /// <summary>Draw the raw SOMA model-space positions without anchoring or scaling.</summary>
        Raw = 1,
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

        [Min(0.0001f)]
        [Tooltip("Scale of the root motion. 1 matches the SOMA skeleton's size.")]
        [SerializeField] private float _rootScale = 1f;

        [Tooltip("Draw the SOMA joint positions while playing.")]
        [SerializeField] private bool _drawDebugJoints;

        [Tooltip("Where the debug joints are drawn: on the avatar (anchored and scaled) or in raw SOMA model space.")]
        [SerializeField] private SomaDebugSpace _debugDrawSpace = SomaDebugSpace.AvatarAnchored;

        [Tooltip("Draw the actual mapped humanoid bones for comparison with the source joints.")]
        [SerializeField] private bool _drawAvatarBones;

        [Tooltip("Draw lines between corresponding source joints and avatar bones.")]
        [SerializeField] private bool _drawBoneErrorLines;

        [Tooltip("Align SOMA motion to the avatar root's starting yaw. Changes take effect when initialized.")]
        [SerializeField] private bool _alignMotionToAvatarYaw = true;

        [Tooltip(
            "Cross-bind left/right: drive the avatar's left bones from the source's 'Right' joints and " +
            "vice versa. The default SOMA skeleton's names are mirrored relative to its geometry " +
            "(toes face +Z while its 'Left' joints sit at +X, the +Z-facing character's right side).")]
        [SerializeField] private bool _swapLeftRightBones = true;

        [Tooltip("Extra yaw in degrees on the source-to-scene placement. Adjust if the overlay is rotated relative to the avatar.")]
        [SerializeField, Range(-180f, 180f)] private float _sourceFacingOffsetDeg;

        [Tooltip(
            "Solve the avatar's own limb chains onto the scene-space pins set by SetAvatarLimbPins, " +
            "after the SOMA rotations are applied. Turn off to see the pure SOMA-rotation result.")]
        [SerializeField] private bool _avatarLimbIk = true;

        [Tooltip("Temporarily disable Animation Rigging while this component is enabled. Applied on enable.")]
        [SerializeField] private bool _disableAnimationRigging = true;

        [Tooltip("Keep skinned meshes updating offscreen while this component is enabled. Applied on enable.")]
        [SerializeField] private bool _updateMeshesOffscreen = true;

        private SomaSkeleton _skeleton;
        private Transform[] _boneTransforms = Array.Empty<Transform>();
        private Quaternion[] _somaRestInv = Array.Empty<Quaternion>();
        private Quaternion[] _restBone = Array.Empty<Quaternion>();
        private int[] _chainEnd = Array.Empty<int>();
        private int[] _jointSide = Array.Empty<int>();
        private Pose[] _somaWorld = Array.Empty<Pose>();
        private Vector3 _anchorPosition;
        private Vector3 _anchorSomaPosition;
        private float _heightScale = 1f;
        private bool _initialized;
        private Pose[] _lastSomaPose;
        private string _missingBonesReport;
        private readonly List<MonoBehaviour> _disabledRigBuilders = new List<MonoBehaviour>();
        private readonly List<SkinnedMeshRenderer> _changedRenderers = new List<SkinnedMeshRenderer>();

        // Avatar limb pins, indexed by the bit position of MbTwoBoneIk.IkPin.
        private readonly bool[] _limbPinUsed = new bool[4];
        private readonly Vector3[] _limbPinTarget = new Vector3[4];
        private readonly bool[] _limbPinUnreachableLogged = new bool[4];
        private float _limbPinWeight;

        /// <summary>The SOMA skeleton this retargeter uses.</summary>
        public SomaSkeleton Skeleton => _skeleton;

        /// <summary>True once the retargeter measured its alignment and can drive the avatar.</summary>
        public bool IsInitialized => _initialized;

        /// <summary>Scale of the root motion. 1 matches the SOMA skeleton's size.</summary>
        public float RootScale
        {
            get => _rootScale;
            set => _rootScale = float.IsNaN(value) || float.IsInfinity(value) ? 1f : Mathf.Clamp(value, 1e-4f, 1e4f);
        }

        /// <summary>
        /// Avatar height per SOMA height, measured from the hips height above the avatar's ground at
        /// initialization. 1 when the avatar is as tall as the SOMA skeleton.
        /// </summary>
        public float HeightScale => _heightScale;

        /// <summary>The scale that root-motion displacements go through (HeightScale * RootScale).</summary>
        public float EffectiveRootScale => Mathf.Max(1e-4f, _heightScale *
            (float.IsNaN(_rootScale) || float.IsInfinity(_rootScale) ? 1f : Mathf.Max(1e-4f, _rootScale)));

        /// <summary>The scene point where the SOMA space is anchored (the avatar's seeded hips).</summary>
        public Vector3 AnchorPosition => _anchorPosition;

        /// <summary>The SOMA point that sits on the anchor (the seeded SOMA root position).</summary>
        public Vector3 AnchorSomaPosition => _anchorSomaPosition;

        /// <summary>True when left/right joints are cross-bound to the avatar's opposite side.</summary>
        public bool SwapLeftRightBones => _swapLeftRightBones;

        /// <summary>
        /// The source-to-scene rotation currently applied: the avatar's yaw plus the manual facing
        /// offset. Computed live, so inspector changes take effect on the next applied pose.
        /// </summary>
        public Quaternion PlacementRotation
        {
            get
            {
                var yaw = _alignMotionToAvatarYaw && _avatar != null ? _avatar.transform.eulerAngles.y : 0f;
                return Quaternion.Euler(0f, yaw + _sourceFacingOffsetDeg, 0f);
            }
        }

        /// <summary>The current map between SOMA motion space and the Unity scene.</summary>
        public SomaSceneMap CurrentSceneMap =>
            new SomaSceneMap(_anchorPosition, _anchorSomaPosition, EffectiveRootScale, PlacementRotation);

        /// <summary>Maps a scene point into SOMA motion space through <see cref="CurrentSceneMap"/>.</summary>
        public Vector3 ScenePointToSoma(Vector3 scenePoint)
        {
            return CurrentSceneMap.ScenePointToSoma(scenePoint);
        }

        /// <summary>Maps a SOMA motion-space point into the scene through <see cref="CurrentSceneMap"/>.</summary>
        public Vector3 SomaPointToScene(Vector3 somaPoint)
        {
            return CurrentSceneMap.SomaPointToScene(somaPoint);
        }

        /// <summary>
        /// The largest distance of a solved avatar limb tip from its pin after the last
        /// <see cref="ApplyPose"/>, in meters. Zero when the avatar limb IK is off or has no pins.
        /// </summary>
        public float AvatarLimbPinError { get; private set; }

        /// <summary>True while avatar-space IK is enabled and at least one pin has nonzero weight.</summary>
        public bool AvatarLimbPinActive => _avatarLimbIk && _limbPinWeight > 0f &&
            (_limbPinUsed[0] || _limbPinUsed[1] || _limbPinUsed[2] || _limbPinUsed[3]);

        /// <summary>How many humanoid bones the retargeter drives.</summary>
        public int BoundBoneCount => _boneTransforms.Length;

        /// <summary>A report of the humanoid bones the avatar lacks, or an empty string.</summary>
        public string MissingBonesReport => _missingBonesReport ?? "";

        private void Awake()
        {
            Initialize();
        }

        private void OnEnable()
        {
            if (_initialized)
            {
                DisableAnimationRigging();
                DisableSkinnedMeshCulling();
            }
        }

        private void OnDisable()
        {
            foreach (var rig in _disabledRigBuilders)
                if (rig != null) rig.enabled = true;
            _disabledRigBuilders.Clear();
            foreach (var renderer in _changedRenderers)
                if (renderer != null) renderer.updateWhenOffscreen = false;
            _changedRenderers.Clear();
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

            if (_avatar.GetBoneTransform(HumanBodyBones.Hips) == null)
            {
                Debug.LogError("[MotionBricks] The humanoid avatar has no hips bone.", this);
                return;
            }

            _skeleton = SomaSkeleton.CreateDefault();
            var numJoints = _skeleton.NumJoints;
            _somaWorld = new Pose[numJoints];

            var somaRestWorld = new Pose[numJoints];
            _skeleton.ComputeReferenceWorld(somaRestWorld);
            if (float.IsNaN(_rootScale) || float.IsInfinity(_rootScale) || _rootScale <= 0f)
            {
                Debug.LogWarning("[MotionBricks] Invalid root scale; using 1.", this);
                _rootScale = 1f;
            }

            var bindings = SomaHumanoidMap.ResolveForAvatar(
                _skeleton, _avatar, _swapLeftRightBones, out var bones, out var chainEnds, out var missingBones);
            _missingBonesReport = missingBones.Count == 0
                ? ""
                : FormatBones("without its own bone", missingBones);
            if (missingBones.Count > 0)
            {
                Debug.LogWarning(
                    $"[MotionBricks] Avatar {_avatar.name} {_missingBonesReport}. " +
                    "Those SOMA rotations fold into the next chain end instead of driving a bone of " +
                    "their own, which can lean the next bone out of shape.", this);
            }

            var count = bindings.Count;
            if (count == 0)
            {
                Debug.LogError("[MotionBricks] The avatar has no humanoid bones the retargeter can drive.", this);
                return;
            }

            _boneTransforms = new Transform[count];
            _somaRestInv = new Quaternion[count];
            _restBone = new Quaternion[count];
            _chainEnd = chainEnds.ToArray();

            // Color debug joints by the avatar side they drive (green = avatar left), so the overlay
            // stays comparable even when the source's names are cross-bound.
            _jointSide = new int[numJoints];
            for (var joint = 0; joint < numJoints; joint++)
            {
                _jointSide[joint] = 2;
            }

            for (var index = 0; index < count; index++)
            {
                var transform = _avatar.GetBoneTransform(bones[index]);
                _boneTransforms[index] = transform;

                // The calibration stores the source's rest rotation and the avatar's rest rotation
                // separately; the placement rotation is applied live in ApplyPose. The transfer is
                // world = placement * (sourceWorld * sourceRest^-1) * placement^-1 * avatarRest,
                // i.e. the source's world rotation delta carried into scene axes.
                _somaRestInv[index] = MbPose.Conjugate(somaRestWorld[chainEnds[index]].rotation);
                _restBone[index] = transform.rotation;

                var boneName = bones[index].ToString();
                var side = boneName.StartsWith("Left") ? 0 : boneName.StartsWith("Right") ? 1 : 2;
                foreach (var joint in bindings[index].SomaJoints)
                {
                    _jointSide[joint] = side;
                }
            }

            // The avatar's hips height above its own origin sets the scale of the root motion. The
            // avatar root marks the ground, so a raised or lowered spawn does not change the scale.
            var hips = _avatar.GetBoneTransform(HumanBodyBones.Hips);
            var somaHips = somaRestWorld[_skeleton.JointIndex("Hips")].position;
            if (hips == null || Mathf.Abs(somaHips.y) <= 1e-4f)
            {
                Debug.LogWarning(
                    "[MotionBricks] The avatar has no hips bone, so root motion keeps a scale of 1.", this);
            }
            else
            {
                var hipsHeight = hips.position.y - _avatar.transform.position.y;
                if (hipsHeight <= 1e-4f || float.IsNaN(hipsHeight) || float.IsInfinity(hipsHeight))
                {
                    Debug.LogWarning(
                        "[MotionBricks] The avatar's hips sit at its root height, so root motion keeps a " +
                        "scale of 1. Initialize while the avatar holds its rest pose.", this);
                }
                else
                {
                    _heightScale = hipsHeight / Mathf.Abs(somaHips.y);
                }
            }

            DisableAnimationRigging();
            DisableSkinnedMeshCulling();
            _initialized = true;
        }

        private static string FormatBones(string suffix, List<HumanBodyBones> bones)
        {
            var report = new StringBuilder();
            foreach (var bone in bones)
            {
                if (report.Length > 0)
                {
                    report.Append(", ");
                }

                report.Append(bone);
            }

            return report.Insert(0, $"humanoid bones {suffix}: ").ToString();
        }

        /// <summary>
        /// Throws away the bindings and alignment and measures them again from the current poses, so
        /// inspector toggles such as Swap Left Right Bones take effect without leaving Play mode.
        /// </summary>
        public void Reinitialize()
        {
            _initialized = false;
            _missingBonesReport = "";
            ClearAvatarLimbPins();
            Initialize();
        }

        /// <summary>
        /// Rebases the current source pose onto the current avatar pose without changing height scale.
        /// </summary>
        public void Recalibrate()
        {
            if (!_initialized || _lastSomaPose == null)
            {
                Debug.LogWarning("[MotionBricks] A source pose is required before recalibration.", this);
                return;
            }

            MbPose.ComposeWorldPoses(_lastSomaPose, _skeleton.ParentIndices, _somaWorld,
                new Pose(Vector3.zero, Quaternion.identity));
            for (var i = 0; i < _boneTransforms.Length; i++)
            {
                _somaRestInv[i] = MbPose.Conjugate(_somaWorld[_chainEnd[i]].rotation);
                _restBone[i] = _boneTransforms[i].rotation;
            }
            _anchorPosition = _boneTransforms[0].position;
            _anchorSomaPosition = _somaWorld[0].position;
            ClearAvatarLimbPins();
        }
        private void DisableSkinnedMeshCulling()
        {
            if (!_updateMeshesOffscreen || !isActiveAndEnabled) return;
            foreach (var renderer in _avatar.GetComponentsInChildren<SkinnedMeshRenderer>(true))
            {
                if (renderer.updateWhenOffscreen) continue;
                _changedRenderers.Add(renderer);
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
            if (!_disableAnimationRigging || !isActiveAndEnabled) return;
            foreach (var behaviour in _avatar.GetComponentsInChildren<MonoBehaviour>(true))
            {
                if (behaviour == null || !behaviour.enabled)
                {
                    continue;
                }

                if (behaviour.GetType().FullName == "UnityEngine.Animations.Rigging.RigBuilder")
                {
                    _disabledRigBuilders.Add(behaviour);
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
        /// Pins the avatar's own limbs (hands, feet) onto a scene-space point. The limb chains are
        /// solved with two-bone IK after the SOMA rotations are applied, so the tips land on the
        /// given scene points as long as they are within reach. Both the SOMA-space IK pins of
        /// <c>TargetPoseIkCorrector</c> and these avatar-space pins can be used together; the SOMA
        /// pins steer the generated motion, the avatar pins straighten what the differently
        /// proportioned avatar shows.
        /// </summary>
        /// <param name="pinMask">The limbs to pin. Every set bit shares the same target.</param>
        /// <param name="target">The scene point the limb tips should reach.</param>
        /// <param name="weight">Blend from the retargeted pose (0) to the solved pose (1).</param>
        public void SetAvatarLimbPins(MbTwoBoneIk.IkPin pinMask, Vector3 target, float weight)
        {
            Initialize();
            _limbPinWeight = Mathf.Clamp01(weight);
            for (var bit = 0; bit < _limbPinUsed.Length; bit++)
            {
                var pin = (MbTwoBoneIk.IkPin)(1 << bit);
                var used = (pinMask & pin) != 0;
                if (used)
                {
                    // The "cannot reach" warning logs once per target, not per frame.
                    if (!_limbPinUsed[bit] || _limbPinTarget[bit] != target)
                        _limbPinUnreachableLogged[bit] = false;
                    _limbPinTarget[bit] = target;
                }

                _limbPinUsed[bit] = used;
            }
        }

        /// <summary>Removes all avatar limb pins.</summary>
        public void ClearAvatarLimbPins()
        {
            _limbPinWeight = 0f;
            for (var bit = 0; bit < _limbPinUsed.Length; bit++)
            {
                _limbPinUsed[bit] = false;
                _limbPinUnreachableLogged[bit] = false;
            }
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

            var placement = PlacementRotation;
            var inversePlacement = Quaternion.Inverse(placement);

            // The root moves the avatar. Its rotation comes from the SOMA root, aligned to the avatar's
            // rest pose. SetPositionAndRotation takes world values, so the world rotation is passed
            // directly.
            var rootWorld = _somaWorld[0];
            var rootRotation = MbPose.Normalize(
                placement * (rootWorld.rotation * _somaRestInv[0]) * inversePlacement * _restBone[0]);
            var hips = _boneTransforms[0];
            var rootPosition = CurrentSceneMap.SomaPointToScene(rootWorld.position);
            hips.SetPositionAndRotation(rootPosition, rootRotation);

            // The other bones follow the hierarchy, so each bone's local rotation is measured against
            // the world rotation its parent received.
            for (var index = 1; index < _boneTransforms.Length; index++)
            {
                var transform = _boneTransforms[index];
                var world = MbPose.Normalize(
                    placement * (_somaWorld[_chainEnd[index]].rotation * _somaRestInv[index]) *
                    inversePlacement * _restBone[index]);

                var boneParent = transform.parent;
                var boneParentRotation = boneParent != null ? boneParent.rotation : Quaternion.identity;
                transform.localRotation = MbPose.Normalize(MbPose.Conjugate(boneParentRotation) * world);
            }

            AvatarLimbPinError = 0f;
            if (_avatarLimbIk && _limbPinWeight > 0f)
            {
                ApplyAvatarLimbPins();
            }
        }

        /// <summary>Solves the avatar's own limb chains onto the current pins, in scene space.</summary>
        private void ApplyAvatarLimbPins()
        {
            for (var bit = 0; bit < _limbPinUsed.Length; bit++)
            {
                if (!_limbPinUsed[bit])
                {
                    continue;
                }

                var pin = (MbTwoBoneIk.IkPin)(1 << bit);
                if (!TryGetLimbChain(pin, out var a, out var b, out var c, out var aParent))
                {
                    if (!_limbPinUnreachableLogged[bit])
                    {
                        _limbPinUnreachableLogged[bit] = true;
                        Debug.LogWarning($"[MotionBricks] Avatar limb {pin} has missing or unrelated bones; IK skipped.", this);
                    }
                    AvatarLimbPinError = float.NaN;
                    continue;
                }

                var aParentWorld = new Pose(aParent.position, aParent.rotation);
                var aWorld = new Pose(a.position, a.rotation);
                var bWorld = new Pose(b.position, b.rotation);
                var cWorld = new Pose(c.position, c.rotation);
                var target = _limbPinTarget[bit];

                // Reach check: report a target the limb cannot reach instead of silently clamping.
                var chainLength = Vector3.Distance(aWorld.position, bWorld.position) +
                                  Vector3.Distance(bWorld.position, cWorld.position);
                if (Vector3.Distance(aWorld.position, bWorld.position) < 1e-5f ||
                    Vector3.Distance(bWorld.position, cWorld.position) < 1e-5f)
                {
                    AvatarLimbPinError = Mathf.Max(AvatarLimbPinError, Vector3.Distance(c.position, target));
                    continue;
                }
                if (_limbPinWeight >= 0.999f && Vector3.Distance(aWorld.position, target) > chainLength * 1.02f &&
                    !_limbPinUnreachableLogged[bit])
                {
                    _limbPinUnreachableLogged[bit] = true;
                    Debug.LogWarning(
                        $"[MotionBricks] The avatar limb {pin} cannot reach its pin at {target:0.000} " +
                        $"(chain reach {chainLength:0.000} m). The IK clamps and the limb stops short.", a);
                }

                // The bend hint follows the current elbow/knee so the retargeted pose keeps its bend
                // direction instead of snapping the joint to a canonical side.
                var hint = bWorld.position;
                var axis = cWorld.position - aWorld.position;
                if (Vector3.Cross(axis, hint - aWorld.position).sqrMagnitude < 1e-8f)
                {
                    var bend = Vector3.ProjectOnPlane(_avatar.transform.forward, axis);
                    if (bend.sqrMagnitude < 1e-8f)
                        bend = Vector3.ProjectOnPlane(_avatar.transform.up, axis);
                    hint += bend.normalized * chainLength;
                }
                MbTwoBoneIk.SolveTwoBoneIk(
                    _limbPinWeight,
                    aParentWorld,
                    aWorld,
                    bWorld,
                    cWorld,
                    new Pose(target, cWorld.rotation),
                    useHint: true,
                    hint,
                    out var solvedA,
                    out var solvedB,
                    out var solvedC);

                a.rotation = solvedA.rotation;
                b.rotation = solvedB.rotation;
                c.rotation = solvedC.rotation;

                AvatarLimbPinError = Mathf.Max(AvatarLimbPinError, Vector3.Distance(c.position, target));
            }
        }

        /// <summary>Resolves the avatar's two-bone chain for one limb pin.</summary>
        private bool TryGetLimbChain(
            MbTwoBoneIk.IkPin pin,
            out Transform a,
            out Transform b,
            out Transform c,
            out Transform aParent)
        {
            a = b = c = aParent = null;
            HumanBodyBones aBone;
            HumanBodyBones bBone;
            HumanBodyBones cBone;
            switch (pin)
            {
                case MbTwoBoneIk.IkPin.LeftHand:
                    aBone = HumanBodyBones.LeftUpperArm;
                    bBone = HumanBodyBones.LeftLowerArm;
                    cBone = HumanBodyBones.LeftHand;
                    break;
                case MbTwoBoneIk.IkPin.RightHand:
                    aBone = HumanBodyBones.RightUpperArm;
                    bBone = HumanBodyBones.RightLowerArm;
                    cBone = HumanBodyBones.RightHand;
                    break;
                case MbTwoBoneIk.IkPin.LeftFoot:
                    aBone = HumanBodyBones.LeftUpperLeg;
                    bBone = HumanBodyBones.LeftLowerLeg;
                    cBone = HumanBodyBones.LeftFoot;
                    break;
                case MbTwoBoneIk.IkPin.RightFoot:
                    aBone = HumanBodyBones.RightUpperLeg;
                    bBone = HumanBodyBones.RightLowerLeg;
                    cBone = HumanBodyBones.RightFoot;
                    break;
                default:
                    return false;
            }

            a = _avatar.GetBoneTransform(aBone);
            b = _avatar.GetBoneTransform(bBone);
            c = _avatar.GetBoneTransform(cBone);
            if (a == null || b == null || c == null)
            {
                return false;
            }

            aParent = a.parent;
            return aParent != null && b.IsChildOf(a) && c.IsChildOf(b);
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
            if ((!_drawDebugJoints && !_drawAvatarBones && !_drawBoneErrorLines) ||
                !_initialized || _lastSomaPose == null || _somaWorld.Length == 0)
            {
                return;
            }

            // AvatarAnchored draws the skeleton where the avatar actually is: hips on hips. Raw shows
            // the model-space motion, which sits wherever the SOMA root travels.
            var map = _debugDrawSpace == SomaDebugSpace.AvatarAnchored
                ? CurrentSceneMap
                : SomaSceneMap.Identity;

            Vector3 SourcePosition(int joint)
            {
                var point = _somaWorld[joint].position;
                return _debugDrawSpace == SomaDebugSpace.Raw ? point :
                    map.SomaJointToScene(_somaWorld[0].position, point, _heightScale);
            }

            for (var joint = 0; _drawDebugJoints && joint < _somaWorld.Length; joint++)
            {
                if (_skeleton.PoseToModel[joint] < 0)
                {
                    continue;
                }

                var position = SourcePosition(joint);
                var parent = _skeleton.JointParent(joint);
                Gizmos.color = JointColor(joint);
                Gizmos.DrawWireSphere(position, 0.02f);
                if (parent >= 0 && _skeleton.PoseToModel[parent] >= 0)
                {
                    Gizmos.DrawLine(position, SourcePosition(parent));
                }
            }

            // The body's forward direction is the root's local +Y (the root frame is tilted), so show
            // that axis to debug headings.
            if (_drawDebugJoints)
            {
                var root = SourcePosition(0);
                Gizmos.color = Color.yellow;
                Gizmos.DrawLine(root, root + map.SomaDirectionToScene(_somaWorld[0].rotation * Vector3.up) * 0.3f);
            }

            for (var i = 0; i < _boneTransforms.Length; i++)
            {
                var bone = _boneTransforms[i];
                if (_drawAvatarBones)
                {
                    var parent = bone.parent;
                    while (parent != null && Array.IndexOf(_boneTransforms, parent) < 0) parent = parent.parent;
                    Gizmos.color = Color.magenta;
                    Gizmos.DrawWireSphere(bone.position, 0.015f);
                    if (parent != null) Gizmos.DrawLine(bone.position, parent.position);
                }
                if (_drawBoneErrorLines)
                {
                    Gizmos.color = Color.white;
                    Gizmos.DrawLine(SourcePosition(_chainEnd[i]), bone.position);
                }
            }
        }

        private Color JointColor(int joint)
        {
            // Side by which avatar limb the joint drives, not by the source joint's name, so green
            // always marks the chain that moves the avatar's left side.
            if (_jointSide != null && joint < _jointSide.Length && _jointSide[joint] == 0)
            {
                return new Color(0.3f, 0.95f, 0.5f, 0.9f);
            }

            if (_jointSide != null && joint < _jointSide.Length && _jointSide[joint] == 1)
            {
                return new Color(1f, 0.6f, 0.2f, 0.9f);
            }

            return new Color(0.2f, 0.9f, 1f, 0.9f);
        }
    }
}
