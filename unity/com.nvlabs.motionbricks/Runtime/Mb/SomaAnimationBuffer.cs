using System;
using UnityEngine;

namespace MotionBricks
{
    /// <summary>
    /// Stores animation frames as local joint transforms at a fixed sample rate, ported from
    /// <c>motionbricks/animation/animation_buffer.py</c>. Foot contacts are predicted for generated
    /// motion and absent for clips.
    /// </summary>
    public sealed class SomaAnimationBuffer
    {
        private readonly SomaSkeleton _skeleton;

        /// <summary>Local joint transforms of every frame, laid out as [frame, joint].</summary>
        public Pose[] Poses { get; }

        /// <summary>Predicted foot contacts, laid out as [frame, contact].</summary>
        public float[] FootContacts { get; }

        /// <summary>Number of frames in use, from 0 to the capacity.</summary>
        public int NumFrames { get; set; }

        public int NumJoints => _skeleton.NumJoints;

        public int Capacity => Poses.Length / NumJoints;

        public float SampleRate { get; }

        /// <summary>Time from the first to the last frame in use, in seconds.</summary>
        public float Duration => Math.Max(0, NumFrames - 1) / SampleRate;

        public SomaAnimationBuffer(SomaSkeleton skeleton, int capacity = MbContract.MaxValidFrames, float sampleRate = MbContract.SampleRate)
        {
            _skeleton = skeleton;
            SampleRate = sampleRate;
            Poses = new Pose[capacity * skeleton.NumJoints];
            FootContacts = new float[capacity * MbContract.FootContactCount];
            NumFrames = 0;
        }

        /// <summary>
        /// Writes the pose sampled at <paramref name="time"/> into <paramref name="destination"/>. The
        /// two nearest frames are blended. Times outside the animation are clamped.
        /// </summary>
        public void Sample(float time, Pose[] destination, Pose rootOffset)
        {
            if (NumFrames == 0)
            {
                throw new InvalidOperationException("Cannot sample an empty animation buffer.");
            }

            MbPose.FrameIndexFromTime(time, SampleRate, NumFrames, out var frame, out var nextFrame, out var blend);
            var numJoints = NumJoints;
            for (var joint = 0; joint < numJoints; joint++)
            {
                var pose = MbPose.Blend(
                    Poses[frame * numJoints + joint],
                    Poses[nextFrame * numJoints + joint],
                    blend);
                if (joint == 0)
                {
                    pose = MbPose.Mul(rootOffset, pose);
                }

                destination[joint] = pose;
            }
        }

        /// <summary>Writes the four foot contacts sampled at <paramref name="time"/> into the output.</summary>
        public void SampleFootContacts(float time, float[] destination)
        {
            if (NumFrames == 0)
            {
                throw new InvalidOperationException("Cannot sample an empty animation buffer.");
            }

            MbPose.FrameIndexFromTime(time, SampleRate, NumFrames, out var frame, out var nextFrame, out var blend);
            for (var contact = 0; contact < MbContract.FootContactCount; contact++)
            {
                var a = FootContacts[frame * MbContract.FootContactCount + contact];
                var b = FootContacts[nextFrame * MbContract.FootContactCount + contact];
                destination[contact] = a + (b - a) * blend;
            }
        }

        /// <summary>Returns the root position of each frame in use.</summary>
        public Vector3[] RootPositions()
        {
            var positions = new Vector3[NumFrames];
            for (var frame = 0; frame < NumFrames; frame++)
            {
                positions[frame] = Poses[frame * NumJoints].position;
            }

            return positions;
        }

        /// <summary>Copies the frames in use and their foot contacts into a new buffer.</summary>
        public SomaAnimationBuffer Copy()
        {
            var copy = new SomaAnimationBuffer(_skeleton, Math.Max(1, NumFrames), SampleRate);
            Array.Copy(Poses, copy.Poses, NumFrames * NumJoints);
            Array.Copy(FootContacts, copy.FootContacts, NumFrames * MbContract.FootContactCount);
            copy.NumFrames = NumFrames;
            return copy;
        }
    }
}
