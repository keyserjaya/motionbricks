using System;
using UnityEngine;

namespace MotionBricks
{
    /// <summary>
    /// Pins a target pose's hands and feet, and eases a played pose's chains onto those pins with
    /// two-bone IK. A port of <c>TargetPoseIKCorrector</c> from
    /// <c>motionbricks/runtime/target_pose.py</c> and its chain configs from
    /// <c>create_soma_target_pose_ik_rig</c>.
    /// </summary>
    public sealed class TargetPoseIkCorrector
    {
        /// <summary>One pinned limb: the chain start, middle, and tip joint indices, plus the bend
        /// hint offset in the middle joint's local space.</summary>
        public readonly struct IkChain
        {
            public readonly MbTwoBoneIk.IkPin Pin;
            public readonly int RootIndex;
            public readonly int MidIndex;
            public readonly int TipIndex;
            public readonly Vector3 HintOffsetLocal;

            public IkChain(MbTwoBoneIk.IkPin pin, int rootIndex, int midIndex, int tipIndex, Vector3 hintOffsetLocal)
            {
                Pin = pin;
                RootIndex = rootIndex;
                MidIndex = midIndex;
                TipIndex = tipIndex;
                HintOffsetLocal = hintOffsetLocal;
            }
        }

        /// <summary>The SOMA hand and foot IK chains, resolved on a skeleton.</summary>
        public static IkChain[] CreateSomaIkRig(SomaSkeleton skeleton)
        {
            return new[]
            {
                new IkChain(MbTwoBoneIk.IkPin.LeftHand,
                    skeleton.JointIndex("LeftArm"), skeleton.JointIndex("LeftForeArm"),
                    skeleton.JointIndex("LeftHand"), new Vector3(0f, -0.25f, 0f)),
                new IkChain(MbTwoBoneIk.IkPin.RightHand,
                    skeleton.JointIndex("RightArm"), skeleton.JointIndex("RightForeArm"),
                    skeleton.JointIndex("RightHand"), new Vector3(0f, 0.25f, 0f)),
                new IkChain(MbTwoBoneIk.IkPin.LeftFoot,
                    skeleton.JointIndex("LeftLeg"), skeleton.JointIndex("LeftShin"),
                    skeleton.JointIndex("LeftFoot"), new Vector3(0f, -0.25f, 0f)),
                new IkChain(MbTwoBoneIk.IkPin.RightFoot,
                    skeleton.JointIndex("RightLeg"), skeleton.JointIndex("RightShin"),
                    skeleton.JointIndex("RightFoot"), new Vector3(0f, 0.25f, 0f)),
            };
        }

        private readonly SomaSkeleton _skeleton;
        private readonly IkChain[] _chains;
        private readonly Pose[] _pinnedTransforms;
        private readonly Pose[] _worldTransforms;
        private readonly Pose[] _localScratch;
        private MbTwoBoneIk.IkPin _pinMask = MbTwoBoneIk.IkPin.None;

        public TargetPoseIkCorrector(SomaSkeleton skeleton, IkChain[] chains)
        {
            _skeleton = skeleton ?? throw new ArgumentNullException(nameof(skeleton));
            _chains = chains ?? throw new ArgumentNullException(nameof(chains));
            if (_chains.Length == 0)
            {
                throw new ArgumentException("Target pose IK chains cannot be empty.", nameof(chains));
            }

            foreach (var chain in _chains)
            {
                if (skeleton.JointParent(chain.MidIndex) != chain.RootIndex ||
                    skeleton.JointParent(chain.TipIndex) != chain.MidIndex)
                {
                    throw new ArgumentException(
                        "An IK chain's mid joint must be a child of its chain start, and its tip a child of the mid joint.");
                }
            }

            _pinnedTransforms = new Pose[_chains.Length];
            _worldTransforms = new Pose[skeleton.NumJoints];
            _localScratch = new Pose[skeleton.NumJoints];
        }

        /// <summary>The pins captured from the current target pose.</summary>
        public MbTwoBoneIk.IkPin PinMask => _pinMask;

        /// <summary>
        /// The reference IK weight ramp: <c>smoothstep(0.5, 1, progress)</c>. The weight stays zero
        /// for the first half of a segment and reaches 1 at the transition point. At progress 0,
        /// 0.25, 0.5, 0.75 and 1 the weights are 0, 0, 0, 0.5 and 1, respectively.
        /// </summary>
        /// <param name="progress">Playback progress through the segment, 0..1.</param>
        public static float LateRampWeight(float progress)
        {
            return Mathf.SmoothStep(0f, 1f, Mathf.InverseLerp(0.5f, 1f, Mathf.Clamp01(progress)));
        }

        /// <summary>
        /// Captures the pinned hand and foot world transforms from a target pose's local transforms.
        /// A mask of None clears the pins.
        /// </summary>
        public void CapturePins(Pose[] targetPose, MbTwoBoneIk.IkPin pinMask)
        {
            if (pinMask == MbTwoBoneIk.IkPin.None)
            {
                ClearPins();
                return;
            }

            ValidatePose(targetPose);
            MbPose.ComposeWorldPoses(targetPose, _skeleton.ParentIndices, _worldTransforms,
                new Pose(Vector3.zero, Quaternion.identity));
            for (var chain = 0; chain < _chains.Length; chain++)
            {
                _pinnedTransforms[chain] = _worldTransforms[_chains[chain].TipIndex];
            }

            _pinMask = pinMask;
        }

        /// <summary>
        /// Corrects the pinned chains of a sampled local pose in place. Nothing changes without
        /// captured pins or with a weight of zero or less. All chains solve from the pose's original
        /// world transforms, matching the reference kernel.
        /// </summary>
        public void Apply(Pose[] currentPose, float weight)
        {
            if (_pinMask == MbTwoBoneIk.IkPin.None || weight <= 0f)
            {
                return;
            }

            ValidatePose(currentPose);
            MbPose.ComposeWorldPoses(currentPose, _skeleton.ParentIndices, _worldTransforms,
                new Pose(Vector3.zero, Quaternion.identity));

            for (var chainIndex = 0; chainIndex < _chains.Length; chainIndex++)
            {
                var chain = _chains[chainIndex];
                if ((_pinMask & chain.Pin) == 0)
                {
                    continue;
                }

                var rootParentIndex = _skeleton.JointParent(chain.RootIndex);
                var rootParentWorld = rootParentIndex >= 0
                    ? _worldTransforms[rootParentIndex]
                    : new Pose(Vector3.zero, Quaternion.identity);
                var midWorld = _worldTransforms[chain.MidIndex];
                var hintWorld = midWorld.position + midWorld.rotation * chain.HintOffsetLocal;

                MbTwoBoneIk.SolveTwoBoneIk(
                    weight,
                    rootParentWorld,
                    _worldTransforms[chain.RootIndex],
                    _worldTransforms[chain.MidIndex],
                    _worldTransforms[chain.TipIndex],
                    _pinnedTransforms[chainIndex],
                    true,
                    hintWorld,
                    out var solvedRoot,
                    out var solvedMid,
                    out var solvedTip);

                // Write the solved chain back as local transforms, matching the kernel's outputs.
                currentPose[chain.RootIndex] = rootParentIndex >= 0
                    ? MbPose.Mul(MbPose.Inverse(rootParentWorld), solvedRoot)
                    : solvedRoot;
                currentPose[chain.MidIndex] = MbPose.Mul(MbPose.Inverse(solvedRoot), solvedMid);
                currentPose[chain.TipIndex] = MbPose.Mul(MbPose.Inverse(solvedMid), solvedTip);
            }
        }

        /// <summary>Disables all pins without reallocating storage.</summary>
        public void ClearPins()
        {
            _pinMask = MbTwoBoneIk.IkPin.None;
        }

        private void ValidatePose(Pose[] pose)
        {
            if (pose == null || pose.Length != _skeleton.NumJoints)
            {
                throw new ArgumentException(
                    $"The pose must contain one transform per skeleton joint ({_skeleton.NumJoints}).");
            }
        }
    }
}
