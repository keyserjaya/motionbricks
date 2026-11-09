using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using UnityEngine;

namespace MotionBricks
{
    /// <summary>
    /// An animation clip stored as local joint transforms at a fixed sample rate, ported from
    /// <c>motionbricks/animation/animation_buffer.py</c> for clips loaded from BVH files.
    /// </summary>
    public sealed class SomaClip
    {
        private readonly SomaSkeleton _skeleton;

        /// <summary>Local joint transforms of every frame, laid out as [frame, joint].</summary>
        public Pose[] Frames { get; }

        public float SampleRate { get; }

        public int NumFrames { get; }

        public int NumJoints => _skeleton.NumJoints;

        public float Duration => Math.Max(0, NumFrames - 1) / SampleRate;

        internal SomaClip(SomaSkeleton skeleton, Pose[] frames, int numFrames, float sampleRate)
        {
            _skeleton = skeleton;
            Frames = frames;
            NumFrames = numFrames;
            SampleRate = sampleRate;
        }

        /// <summary>
        /// Writes the pose sampled at <paramref name="time"/> into <paramref name="destination"/>.
        /// The two nearest frames are blended. Times outside the clip are clamped.
        /// </summary>
        public void Sample(float time, Pose[] destination, Pose rootOffset)
        {
            if (NumFrames == 0)
            {
                throw new InvalidOperationException("Cannot sample an empty clip.");
            }

            MbPose.FrameIndexFromTime(time, SampleRate, NumFrames, out var frame, out var nextFrame, out var blend);
            var numJoints = NumJoints;
            for (var joint = 0; joint < numJoints; joint++)
            {
                var pose = MbPose.Blend(
                    Frames[frame * numJoints + joint],
                    Frames[nextFrame * numJoints + joint],
                    blend);
                if (joint == 0)
                {
                    pose = MbPose.Mul(rootOffset, pose);
                }

                destination[joint] = pose;
            }
        }

        /// <summary>Returns the root position of each frame.</summary>
        public Vector3[] RootPositions()
        {
            var positions = new Vector3[NumFrames];
            for (var frame = 0; frame < NumFrames; frame++)
            {
                positions[frame] = Frames[frame * NumJoints].position;
            }

            return positions;
        }
    }

    /// <summary>
    /// Loads animation clips from BVH files, ported from <c>motionbricks/io/bvh.py</c>. Positions are
    /// converted from centimeters to meters, and rotation angles are read in degrees and applied in the
    /// order listed for each joint.
    /// </summary>
    public static class SomaBvhLoader
    {
        /// <summary>
        /// Loads a BVH file and binds it to <paramref name="targetSkeleton"/> by exact joint name.
        /// Joints that the file lacks keep the skeleton's reference transforms.
        /// </summary>
        public static SomaClip LoadBvh(string path, SomaSkeleton targetSkeleton)
        {
            if (string.IsNullOrEmpty(path))
            {
                throw new ArgumentException("path cannot be empty.", nameof(path));
            }

            if (!File.Exists(path))
            {
                throw new FileNotFoundException($"BVH file not found: {path}", path);
            }

            var text = File.ReadAllText(path, Encoding.UTF8);
            return ParseBvh(text, targetSkeleton);
        }

        /// <summary>Loads a BVH file from a text asset, for example one streamed with the app.</summary>
        public static SomaClip LoadBvh(TextAsset asset, SomaSkeleton targetSkeleton)
        {
            if (asset == null)
            {
                throw new ArgumentNullException(nameof(asset));
            }

            return ParseBvh(asset.text, targetSkeleton);
        }

        /// <summary>
        /// Parses BVH text into a clip bound to <paramref name="targetSkeleton"/>.
        /// </summary>
        public static SomaClip ParseBvh(string text, SomaSkeleton targetSkeleton)
        {
            var names = new List<string>();
            var parents = new List<int>();
            var offsets = new List<Vector3>();
            var jointChannels = new List<List<string>>();

            var lines = text.Split('\n');
            var index = 0;
            var jointStack = new List<int>();
            var inEndSite = false;
            var frameCount = 0;
            float sampleRate = 0f;

            // Hierarchy.
            for (; index < lines.Length; index++)
            {
                var tokens = SplitTokens(lines[index]);
                if (tokens.Count == 0)
                {
                    continue;
                }

                if (tokens[0] == "MOTION")
                {
                    break;
                }

                if (tokens[0] == "ROOT" || tokens[0] == "JOINT")
                {
                    var name = tokens[1];
                    var separator = name.LastIndexOf(':');
                    if (separator >= 0 && separator < name.Length - 1)
                    {
                        name = name.Substring(separator + 1);
                    }

                    names.Add(name);
                    parents.Add(jointStack.Count > 0 ? jointStack[jointStack.Count - 1] : -1);
                    jointStack.Add(names.Count - 1);
                    offsets.Add(Vector3.zero);
                    jointChannels.Add(new List<string>());
                }
                else if (tokens[0] == "End" && tokens.Count > 1 && tokens[1] == "Site")
                {
                    inEndSite = true;
                }
                else if (tokens[0] == "OFFSET" && !inEndSite && jointStack.Count > 0)
                {
                    offsets[jointStack[jointStack.Count - 1]] = new Vector3(
                        ParseFloat(tokens[1]), ParseFloat(tokens[2]), ParseFloat(tokens[3])) * 0.01f;
                }
                else if (tokens[0] == "CHANNELS" && jointStack.Count > 0)
                {
                    var declaredChannels = int.Parse(tokens[1], CultureInfo.InvariantCulture);
                    var channels = new List<string>();
                    for (var channel = 0; channel < declaredChannels; channel++)
                    {
                        channels.Add(tokens[2 + channel]);
                    }

                    jointChannels[jointStack[jointStack.Count - 1]] = channels;
                }
                else if (tokens[0] == "}")
                {
                    if (inEndSite)
                    {
                        inEndSite = false;
                    }
                    else if (jointStack.Count > 0)
                    {
                        jointStack.RemoveAt(jointStack.Count - 1);
                    }
                }
            }

            // Motion header.
            for (; index < lines.Length; index++)
            {
                var tokens = SplitTokens(lines[index]);
                if (tokens.Count == 0)
                {
                    continue;
                }

                if (tokens[0] == "Frames:")
                {
                    frameCount = int.Parse(tokens[1], CultureInfo.InvariantCulture);
                }
                else if (tokens[0] == "Frame" && tokens.Count > 1 && tokens[1] == "Time:")
                {
                    var frameTime = ParseFloat(tokens[2]);
                    sampleRate = frameTime > 0f ? Mathf.Round(1f / frameTime * 100f) / 100f : 30f;
                    index++;
                    break;
                }
            }

            // Motion values.
            var values = new List<float>();
            for (; index < lines.Length; index++)
            {
                var tokens = SplitTokens(lines[index]);
                if (tokens.Count == 0)
                {
                    continue;
                }

                foreach (var token in tokens)
                {
                    values.Add(ParseFloat(token));
                }
            }

            var channelCount = 0;
            foreach (var channels in jointChannels)
            {
                channelCount += channels.Count;
            }

            if (values.Count < frameCount * channelCount)
            {
                throw new InvalidOperationException(
                    $"The BVH motion holds {values.Count} values, but {frameCount} frames of {channelCount} channels need " +
                    $"{frameCount * channelCount}.");
            }

            // Bind the file's joints to the target skeleton by name.
            var numJoints = targetSkeleton.NumJoints;
            var sourceJointOfTarget = new int[numJoints];
            var hasSource = new bool[numJoints];
            for (var joint = 0; joint < numJoints; joint++)
            {
                sourceJointOfTarget[joint] = -1;
                var source = names.IndexOf(targetSkeleton.JointNames[joint]);
                if (source >= 0)
                {
                    sourceJointOfTarget[joint] = source;
                    hasSource[joint] = true;
                }
            }

            // Channel layout per source joint.
            var hasPositions = new bool[names.Count];
            var hasRotations = new bool[names.Count];
            foreach (var joint in System.Linq.Enumerable.Range(0, names.Count))
            {
                foreach (var channel in jointChannels[joint])
                {
                    var lower = channel.ToLowerInvariant();
                    if (lower.EndsWith("position", StringComparison.Ordinal))
                    {
                        hasPositions[joint] = true;
                    }
                    else if (lower.EndsWith("rotation", StringComparison.Ordinal))
                    {
                        hasRotations[joint] = true;
                    }
                }
            }

            var frames = new Pose[frameCount * numJoints];
            var framePositions = new Vector3[names.Count];
            var frameRotations = new Quaternion[names.Count];
            for (var frame = 0; frame < frameCount; frame++)
            {
                // Read the file's channels in file order. A joint with position channels starts from
                // zero and fills the listed axes; a joint without them keeps the file's offset. A joint
                // without rotation channels keeps an identity rotation.
                var columnCursor = frame * channelCount;
                for (var source = 0; source < names.Count; source++)
                {
                    var position = hasPositions[source] ? Vector3.zero : offsets[source];
                    var rotation = Quaternion.identity;

                    foreach (var channel in jointChannels[source])
                    {
                        var lower = channel.ToLowerInvariant();
                        var value = values[columnCursor++];
                        if (lower.EndsWith("position", StringComparison.Ordinal))
                        {
                            var axis = "xyz".IndexOf(lower[0]);
                            position = axis == 0 ? new Vector3(value * 0.01f, position.y, position.z)
                                : axis == 1 ? new Vector3(position.x, value * 0.01f, position.z)
                                : new Vector3(position.x, position.y, value * 0.01f);
                        }
                        else if (lower.EndsWith("rotation", StringComparison.Ordinal))
                        {
                            var axis = "xyz".IndexOf(lower[0]);
                            var unit = axis == 0 ? Vector3.right : axis == 1 ? Vector3.up : Vector3.forward;
                            // The file lists the rotation axes in order, and the composition follows it.
                            rotation = rotation * Quaternion.AngleAxis(value, unit);
                        }
                    }

                    framePositions[source] = position;
                    frameRotations[source] = MbPose.Normalize(rotation);
                }

                // Bind the file's joints to the target skeleton by name. A joint the file lacks keeps
                // the target skeleton's reference transform.
                for (var joint = 0; joint < numJoints; joint++)
                {
                    if (!hasSource[joint])
                    {
                        frames[frame * numJoints + joint] = targetSkeleton.ReferenceLocal[joint];
                        continue;
                    }

                    var source = sourceJointOfTarget[joint];
                    var position = hasPositions[source] ? framePositions[source] : offsets[source];
                    var rotation = hasRotations[source] ? frameRotations[source] : Quaternion.identity;
                    frames[frame * numJoints + joint] = new Pose(position, rotation);
                }
            }

            return new SomaClip(targetSkeleton, frames, frameCount, sampleRate);
        }

        private static List<string> SplitTokens(string line)
        {
            var tokens = new List<string>();
            foreach (var token in line.Split(new[] { ' ', '\t', '\r' }, StringSplitOptions.RemoveEmptyEntries))
            {
                tokens.Add(token);
            }

            return tokens;
        }

        private static float ParseFloat(string token)
        {
            return float.Parse(token, CultureInfo.InvariantCulture);
        }
    }
}
