using System;
using System.Collections.Generic;
using UnityEngine;

namespace MotionBricks
{
    /// <summary>
    /// The SOMA joint hierarchy and its reference pose, ported from
    /// <c>motionbricks/animation/skeleton.py</c>. The bundled USD skeleton holds the model's 30
    /// joints plus the finger chains, which the model does not predict.
    /// </summary>
    public sealed class SomaSkeleton
    {
        private readonly Dictionary<string, int> _indexByName;

        /// <summary>Name of each joint.</summary>
        public string[] JointNames { get; }

        /// <summary>Parent index of each joint, or -1 for the root.</summary>
        public int[] ParentIndices { get; }

        /// <summary>Reference pose, one local transform per joint.</summary>
        public Pose[] ReferenceLocal { get; }

        /// <summary>Maps a model joint index to the skeleton joint that holds it.</summary>
        public int[] ModelToPose { get; }

        /// <summary>Maps a skeleton joint to its model joint index, or -1 when the model does not predict it.</summary>
        public int[] PoseToModel { get; }

        /// <summary>Index of the right leg joint, whose position gives the pose heading.</summary>
        public int RightLegIndex { get; }

        /// <summary>Index of the left leg joint, whose position gives the pose heading.</summary>
        public int LeftLegIndex { get; }

        public int NumJoints => JointNames.Length;

        private SomaSkeleton(string[] jointNames, int[] parentIndices, Pose[] referenceLocal)
        {
            JointNames = jointNames;
            ParentIndices = parentIndices;
            ReferenceLocal = referenceLocal;
            _indexByName = new Dictionary<string, int>(jointNames.Length, StringComparer.Ordinal);
            for (var joint = 0; joint < jointNames.Length; joint++)
            {
                _indexByName[jointNames[joint]] = joint;
            }

            ModelToPose = new int[SomaModelContract.NumModelJoints];
            for (var modelJoint = 0; modelJoint < SomaModelContract.NumModelJoints; modelJoint++)
            {
                var index = JointIndex(SomaModelContract.JointNames[modelJoint]);
                if (index < 0)
                {
                    throw new ArgumentException(
                        $"The skeleton is missing the model joint '{SomaModelContract.JointNames[modelJoint]}'.");
                }

                ModelToPose[modelJoint] = index;
            }

            if (ModelToPose[0] != 0)
            {
                throw new ArgumentException("The skeleton must store the root joint at index zero.");
            }

            // Model joints outside the connected skeleton hierarchy keep their local transforms from
            // the reference pose during decoding.
            PoseToModel = new int[JointNames.Length];
            for (var joint = 0; joint < JointNames.Length; joint++)
            {
                PoseToModel[joint] = -1;
            }

            for (var modelJoint = 0; modelJoint < SomaModelContract.NumModelJoints; modelJoint++)
            {
                var poseJoint = ModelToPose[modelJoint];
                var parent = ParentIndices[poseJoint];
                if (parent == -1 || PoseToModel[parent] >= 0)
                {
                    PoseToModel[poseJoint] = modelJoint;
                }
            }

            RightLegIndex = JointIndex(SomaModelContract.RightHeadingJoint);
            LeftLegIndex = JointIndex(SomaModelContract.LeftHeadingJoint);
            if (RightLegIndex < 0 || LeftLegIndex < 0)
            {
                throw new ArgumentException("The skeleton is missing the heading joints.");
            }
        }

        /// <summary>Returns the bundled SOMA skeleton that ships with the MotionBricks repository.</summary>
        public static SomaSkeleton CreateDefault()
        {
            var names = SomaSkeletonDefault.JointNames;
            var parents = SomaSkeletonDefault.ParentIndices;
            var values = SomaSkeletonDefault.ReferenceLocal;
            var reference = new Pose[names.Length];
            for (var joint = 0; joint < names.Length; joint++)
            {
                var baseIndex = joint * 7;
                reference[joint] = new Pose(
                    new Vector3(values[baseIndex], values[baseIndex + 1], values[baseIndex + 2]),
                    new Quaternion(values[baseIndex + 3], values[baseIndex + 4], values[baseIndex + 5], values[baseIndex + 6]));
            }

            return new SomaSkeleton(names, parents, reference);
        }

        /// <summary>Returns the index of a joint, or -1 when no joint has that name.</summary>
        public int JointIndex(string jointName)
        {
            return _indexByName.TryGetValue(jointName, out var index) ? index : -1;
        }

        /// <summary>Returns the parent index of a joint, or -1 for the root.</summary>
        public int JointParent(int index) => ParentIndices[index];

        /// <summary>Copies the reference pose into a caller-provided buffer.</summary>
        public void CopyReferencePose(Pose[] destination)
        {
            Array.Copy(ReferenceLocal, destination, ReferenceLocal.Length);
        }

        /// <summary>Composes the reference pose into world transforms.</summary>
        public void ComputeReferenceWorld(Pose[] destination)
        {
            MbPose.ComposeWorldPoses(ReferenceLocal, ParentIndices, destination, new Pose(Vector3.zero, Quaternion.identity));
        }
    }
}
