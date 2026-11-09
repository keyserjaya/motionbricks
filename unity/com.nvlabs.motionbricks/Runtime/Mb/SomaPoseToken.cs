using System;
using UnityEngine;

namespace MotionBricks
{
    /// <summary>
    /// Holds up to four poses for the model context or a target, ported from
    /// <c>motionbricks/model/soma/buffers.py</c>. The poses are local joint transforms laid out as
    /// [pose, joint]. Padding repeats the last pose in the unused space.
    /// </summary>
    public sealed class SomaPoseToken
    {
        private readonly SomaSkeleton _skeleton;

        /// <summary>Local joint transforms of all four poses, laid out as [pose, joint].</summary>
        public Pose[] Poses { get; }

        /// <summary>Number of supplied poses, from 1 to 4.</summary>
        public int NumPoses { get; private set; }

        public int NumJoints => _skeleton.NumJoints;

        public SomaPoseToken(SomaSkeleton skeleton)
        {
            _skeleton = skeleton;
            Poses = new Pose[MbContract.PosesPerToken * skeleton.NumJoints];
            NumPoses = 0;
        }

        /// <summary>
        /// Copies one to four poses and pads the unused space with the last pose. The poses are in
        /// time order at the model's sample rate of 30 Hz.
        /// </summary>
        public void UpdateFromPoses(Pose[] localPoses, int numJoints, int numPoses)
        {
            if (numPoses < 1 || numPoses > MbContract.PosesPerToken)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(numPoses), numPoses, $"A context or target token requires 1 to {MbContract.PosesPerToken} poses.");
            }

            if (numJoints != NumJoints)
            {
                throw new ArgumentException(
                    $"The pose holds {numJoints} joints, the skeleton holds {NumJoints}.", nameof(numJoints));
            }

            for (var pose = 0; pose < MbContract.PosesPerToken; pose++)
            {
                var source = Math.Min(pose, numPoses - 1);
                Array.Copy(
                    localPoses,
                    source * numJoints,
                    Poses,
                    pose * NumJoints,
                    numJoints);
            }

            NumPoses = numPoses;
        }

        /// <summary>Copies one pose into the token.</summary>
        public void UpdateFromPose(Pose[] localPoses, int numJoints)
        {
            UpdateFromPoses(localPoses, numJoints, 1);
        }

        /// <summary>
        /// Samples four poses at 30 Hz from the recorded history, ending at the latest recorded pose.
        /// Ported from <c>PoseToken.update_from_recorder</c> and <c>history_sample_indices</c>.
        /// </summary>
        public void UpdateFromRecorder(SomaMotionRecorder recorder)
        {
            var numJoints = NumJoints;
            SomaMotionRecorder.HistorySampleIndices(
                recorder.DeltaTimes,
                MbContract.PosesPerToken,
                MbContract.SampleRate,
                out var lowerFrames,
                out var upperFrames,
                out var blends);

            for (var pose = 0; pose < MbContract.PosesPerToken; pose++)
            {
                var lower = lowerFrames[pose] * numJoints;
                var upper = upperFrames[pose] * numJoints;
                var blend = blends[pose];
                for (var joint = 0; joint < numJoints; joint++)
                {
                    Poses[pose * numJoints + joint] = MbPose.Blend(
                        recorder.Poses[lower + joint],
                        recorder.Poses[upper + joint],
                        blend);
                }
            }

            NumPoses = MbContract.PosesPerToken;
        }

        /// <summary>Copies the pose rows into a caller-provided buffer of [pose, joint] layout.</summary>
        public void CopyPoses(Pose[] destination, int numPoses)
        {
            var numJoints = NumJoints;
            for (var pose = 0; pose < numPoses; pose++)
            {
                Array.Copy(Poses, pose * numJoints, destination, pose * numJoints, numJoints);
            }
        }
    }
}
