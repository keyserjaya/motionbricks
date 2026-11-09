using System;
using System.Collections.Generic;
using NUnit.Framework;
using UnityEditor;
using UnityEngine;
using Object = UnityEngine.Object;

namespace MotionBricks.Tests
{
    public sealed class SomaVrmRetargeterTests
    {
        private const string FixturePath = "Assets/Res/VRMS/Prefabs/Character-VRM1-Takagi Katsumi.prefab";
        private const float PositionTolerance = 1e-4f;
        private const float RotationToleranceDegrees = 0.1f;
        private readonly List<GameObject> _clones = new List<GameObject>();
        private GameObject _prefab;

        [SetUp]
        public void SetUp()
        {
            foreach (var guid in AssetDatabase.FindAssets("Takagi t:Prefab", new[] { "Assets" }))
            {
                var path = AssetDatabase.GUIDToAssetPath(guid);
                if (!string.Equals(path, FixturePath, StringComparison.OrdinalIgnoreCase)) continue;
                _prefab = AssetDatabase.LoadAssetAtPath<GameObject>(path);
                break;
            }

            if (_prefab == null)
                Assert.Ignore($"Optional humanoid fixture not found at {FixturePath}.");
        }

        [TearDown]
        public void TearDown()
        {
            foreach (var clone in _clones)
                if (clone != null) Object.DestroyImmediate(clone);
            _clones.Clear();
            _prefab = null;
        }

        [Test]
        public void ReferencePosePreservesRestAcrossTranslatedElevatedYawedSpawns()
        {
            var positions = new[] { Vector3.zero, new Vector3(3f, 4f, -2f), new Vector3(-7f, -2f, 5f) };
            var yaws = new[] { 0f, 67f, -112f };
            var firstScale = 0f;
            for (var i = 0; i < positions.Length; i++)
            {
                var rig = CreateRig(positions[i], yaws[i]);
                var bones = GetBoundBones(rig);
                var rest = Capture(bones);
                var reference = ReferencePose(rig);
                var somaHipsHeight = reference[0].position.y;
                var expectedScale = (rig.Animator.GetBoneTransform(HumanBodyBones.Hips).position.y -
                                     rig.Animator.transform.position.y) / Mathf.Abs(somaHipsHeight);
                Assert.Greater(expectedScale, 0.1f, "The fixture must have a meaningful hips height.");
                Assert.AreEqual(expectedScale, rig.Retargeter.HeightScale, 1e-4f, "Height is relative to the avatar root.");
                if (i == 0) firstScale = rig.Retargeter.HeightScale;
                else Assert.AreEqual(firstScale, rig.Retargeter.HeightScale, 1e-4f, "Spawn height and yaw must not change scale.");

                rig.Retargeter.SetAnchor(rest[0].position, reference[0].position);
                rig.Retargeter.ApplyPose(reference);
                AssertPoseEquals(rest, bones, $"reference pose at spawn {i}");
                AssertVector(rest[0].position, rig.Retargeter.SomaPointToScene(reference[0].position), "anchor mapping");

                var offset = new Vector3(0.4f, 0.2f, -0.6f);
                var placement = rig.Retargeter.PlacementRotation;
                var expected = rest[0].position + placement * offset * rig.Retargeter.EffectiveRootScale;
                AssertVector(expected, rig.Retargeter.SomaPointToScene(reference[0].position + offset), "translated source point");
                AssertVector(reference[0].position + offset, rig.Retargeter.ScenePointToSoma(expected), "inverse map");
            }
        }

        [Test]
        public void SourceYaw90RotatesAvatarInWorldAndMovesAlongSpawnHeading()
        {
            var rig = CreateRig(new Vector3(2.3f, 3.1f, -1.4f), 53f);
            var bones = GetBoundBones(rig);
            var rest = Capture(bones);
            var pose = ReferencePose(rig);
            var placement = Quaternion.Euler(0f, rig.Animator.transform.eulerAngles.y, 0f);
            var sourceYaw = Quaternion.Euler(0f, 90f, 0f);
            var sourceOffset = new Vector3(0.7f, 0.1f, -0.2f);
            rig.Retargeter.SetAnchor(rest[0].position, pose[0].position);
            pose[0] = new Pose(pose[0].position + sourceOffset, sourceYaw * pose[0].rotation);

            rig.Retargeter.ApplyPose(pose);

            var sceneYaw = placement * sourceYaw * Quaternion.Inverse(placement);
            AssertVector(rest[0].position + placement * sourceOffset * rig.Retargeter.EffectiveRootScale,
                bones[0].position, "yawed root translation");
            for (var i = 0; i < bones.Count; i++)
                AssertRotation(sceneYaw * rest[i].rotation, bones[i].rotation, $"world yaw on {bones[i].name}");
        }

        [Test]
        public void RecalibrationPreservesNonRestPoseAndTransfersAsymmetricSourceRotations()
        {
            var rig = CreateRig(new Vector3(-2f, 2.5f, 4f), 38f);
            var bones = GetBoundBones(rig);
            var bindings = SomaHumanoidMap.ResolveForAvatar(rig.Retargeter.Skeleton, rig.Animator,
                rig.Retargeter.SwapLeftRightBones, out var mappedBones, out var chainEnds, out _);
            int JointDriving(HumanBodyBones bone) => chainEnds[mappedBones.IndexOf(bone)];
            var pose = ReferencePose(rig);
            var hips = rig.Animator.GetBoneTransform(HumanBodyBones.Hips);
            var leftArm = rig.Animator.GetBoneTransform(HumanBodyBones.LeftUpperArm);
            var rightArm = rig.Animator.GetBoneTransform(HumanBodyBones.RightLowerArm);
            Assert.IsNotNull(leftArm);
            Assert.IsNotNull(rightArm);
            rig.Retargeter.SetAnchor(hips.position, pose[0].position);

            pose[0] = new Pose(pose[0].position + new Vector3(0.5f, 0.15f, -0.3f),
                Quaternion.Euler(0f, 28f, 0f) * pose[0].rotation);
            RotateJointAt(pose, JointDriving(HumanBodyBones.LeftUpperArm), Quaternion.Euler(16f, -21f, 32f));
            rig.Retargeter.ApplyPose(pose);
            hips.position += new Vector3(0.3f, 0.12f, -0.22f);
            hips.rotation = Quaternion.Euler(0f, 13f, 0f) * hips.rotation;
            leftArm.rotation = Quaternion.Euler(18f, -12f, 8f) * leftArm.rotation;
            rightArm.rotation = Quaternion.Euler(-9f, 15f, -20f) * rightArm.rotation;

            var before = Capture(bones);
            var heightScale = rig.Retargeter.HeightScale;
            var calibratedSource = WorldPoses(rig.Retargeter.Skeleton, pose);
            rig.Retargeter.SetAvatarLimbPins(MbTwoBoneIk.IkPin.LeftHand, leftArm.position, 1f);
            rig.Retargeter.Recalibrate();
            AssertPoseEquals(before, bones, "recalibration itself");
            Assert.IsFalse(rig.Retargeter.AvatarLimbPinActive, "Recalibration clears stale avatar pins.");
            Assert.AreEqual(heightScale, rig.Retargeter.HeightScale, 1e-6f);
            AssertVector(before[0].position, rig.Retargeter.AnchorPosition, "recalibrated avatar anchor");
            AssertVector(calibratedSource[0].position, rig.Retargeter.AnchorSomaPosition, "recalibrated source anchor");
            rig.Retargeter.ApplyPose(pose);
            AssertPoseEquals(before, bones, "reapplying the calibrated non-rest pose");

            var next = (Pose[])pose.Clone();
            RotateJointAt(next, JointDriving(HumanBodyBones.LeftUpperArm), Quaternion.Euler(37f, -24f, 19f));
            RotateJointAt(next, JointDriving(HumanBodyBones.RightLowerArm), Quaternion.Euler(-31f, 12f, 26f));
            var nextSource = WorldPoses(rig.Retargeter.Skeleton, next);
            Assert.AreEqual(bones.Count, bindings.Count);
            rig.Retargeter.ApplyPose(next);
            var placement = Quaternion.Euler(0f, rig.Animator.transform.eulerAngles.y, 0f);
            for (var i = 0; i < bones.Count; i++)
            {
                Assert.AreSame(bones[i], rig.Animator.GetBoneTransform(mappedBones[i]));
                var joint = chainEnds[i];
                var delta = placement * nextSource[joint].rotation *
                            Quaternion.Inverse(calibratedSource[joint].rotation) * Quaternion.Inverse(placement);
                AssertRotation(delta * before[i].rotation, bones[i].rotation, $"calibrated source delta on {mappedBones[i]}");
            }
            Assert.Greater(Quaternion.Angle(before[bones.IndexOf(leftArm)].rotation, leftArm.rotation), 20f);
            Assert.Greater(Quaternion.Angle(before[bones.IndexOf(rightArm)].rotation, rightArm.rotation), 20f);
        }

        [TestCase(35f, false, TestName = "AvatarHandIkReachesTargetFromBentLimb")]
        [TestCase(0f, true, TestName = "AvatarHandIkReachesTargetFromStraightLimb")]
        [TestCase(0.01f, true, TestName = "AvatarHandIkReachesTargetFromNearStraightLimb")]
        public void AvatarHandIkMovesActualHandAndReportsActualError(float elbowDegrees, bool expectStraightFallback)
        {
            var rig = CreateRig(new Vector3(1f, 1.4f, -3f), 41f);
            var pose = ReferencePose(rig);
            var hips = rig.Animator.GetBoneTransform(HumanBodyBones.Hips);
            var upper = rig.Animator.GetBoneTransform(HumanBodyBones.LeftUpperArm);
            var lower = rig.Animator.GetBoneTransform(HumanBodyBones.LeftLowerArm);
            var hand = rig.Animator.GetBoneTransform(HumanBodyBones.LeftHand);
            Assert.IsNotNull(upper);
            Assert.IsNotNull(lower);
            Assert.IsNotNull(hand);
            Assert.IsTrue(lower.IsChildOf(upper) && hand.IsChildOf(lower));
            rig.Retargeter.SetAnchor(hips.position, pose[0].position);
            rig.Retargeter.ApplyPose(pose);

            var upperDirection = (lower.position - upper.position).normalized;
            lower.rotation = Quaternion.FromToRotation(hand.position - lower.position, upperDirection) * lower.rotation;
            var bendAxis = Vector3.Cross(upperDirection, rig.Animator.transform.forward).normalized;
            if (bendAxis.sqrMagnitude < 1e-6f)
                bendAxis = Vector3.Cross(upperDirection, rig.Animator.transform.up).normalized;
            lower.rotation = Quaternion.AngleAxis(elbowDegrees, bendAxis) * lower.rotation;
            var chainLength = Vector3.Distance(upper.position, lower.position) + Vector3.Distance(lower.position, hand.position);
            Assert.Greater(chainLength, 0.05f, "The fixture's arm must be nondegenerate.");
            if (expectStraightFallback)
                Assert.Less(Vector3.Cross(hand.position - upper.position, lower.position - upper.position).sqrMagnitude,
                    1e-8f, "The setup must exercise the straight-limb hint fallback.");

            // Rebase the edited avatar arm so applying the unchanged source pose retains the setup.
            rig.Retargeter.Recalibrate();
            var direction = (hand.position - upper.position).normalized;
            var side = Vector3.Cross(direction, rig.Animator.transform.up).normalized;
            if (side.sqrMagnitude < 1e-6f) side = Vector3.Cross(direction, rig.Animator.transform.forward).normalized;
            var target = upper.position + chainLength * (0.75f * direction + 0.12f * side);
            var beforeDistance = Vector3.Distance(hand.position, target);
            Assert.Greater(beforeDistance, 0.02f, "The hand must actually need to move.");
            Assert.Less(Vector3.Distance(upper.position, target), chainLength - 0.01f, "Target must be within reach.");
            Assert.Greater(Vector3.Distance(upper.position, target),
                Mathf.Abs(Vector3.Distance(upper.position, lower.position) - Vector3.Distance(lower.position, hand.position)) + 0.01f,
                "Target must be outside the chain's inner unreachable radius.");

            rig.Retargeter.SetAvatarLimbPins(MbTwoBoneIk.IkPin.LeftHand, target, 1f);
            Assert.IsTrue(rig.Retargeter.AvatarLimbPinActive);
            rig.Retargeter.ApplyPose(pose);
            var actualError = Vector3.Distance(hand.position, target);
            Assert.AreEqual(actualError, rig.Retargeter.AvatarLimbPinError, 1e-5f, "Public error must measure the actual hand.");
            Assert.Less(actualError, beforeDistance, "IK must move the actual hand toward the target.");
            Assert.Less(actualError, 0.005f, "A reachable target should be reached within 5 mm.");
            rig.Retargeter.ClearAvatarLimbPins();
            rig.Retargeter.ApplyPose(pose);
            Assert.IsFalse(rig.Retargeter.AvatarLimbPinActive);
            Assert.AreEqual(0f, rig.Retargeter.AvatarLimbPinError, 1e-6f);
        }

        private Rig CreateRig(Vector3 position, float yaw)
        {
            var clone = Object.Instantiate(_prefab);
            _clones.Add(clone);
            clone.SetActive(false);
            clone.hideFlags = HideFlags.HideAndDontSave;
            clone.transform.SetPositionAndRotation(position, Quaternion.Euler(0f, yaw, 0f));
            var animator = clone.GetComponentInChildren<Animator>(true);
            Assert.IsNotNull(animator, "The fixture must contain an Animator.");
            Assert.IsNotNull(animator.avatar, "The fixture must have an avatar.");
            Assert.IsTrue(animator.avatar.isValid && animator.avatar.isHuman, "The fixture must be a valid humanoid.");
            animator.enabled = false;
            var retargeter = animator.gameObject.AddComponent<SomaVrmRetargeter>();
            retargeter.Initialize();
            Assert.IsTrue(retargeter.IsInitialized);
            return new Rig { Animator = animator, Retargeter = retargeter };
        }

        private static List<Transform> GetBoundBones(Rig rig)
        {
            SomaHumanoidMap.ResolveForAvatar(rig.Retargeter.Skeleton, rig.Animator,
                rig.Retargeter.SwapLeftRightBones, out var mappedBones, out _, out _);
            var bones = new List<Transform>();
            foreach (var bone in mappedBones) bones.Add(rig.Animator.GetBoneTransform(bone));
            Assert.Greater(bones.Count, 10);
            Assert.AreSame(rig.Animator.GetBoneTransform(HumanBodyBones.Hips), bones[0]);
            return bones;
        }

        private static Pose[] ReferencePose(Rig rig)
        {
            var skeleton = rig.Retargeter.Skeleton;
            var pose = new Pose[skeleton.NumJoints];
            skeleton.CopyReferencePose(pose);
            return pose;
        }

        private static Pose[] WorldPoses(SomaSkeleton skeleton, Pose[] local)
        {
            var world = new Pose[skeleton.NumJoints];
            MbPose.ComposeWorldPoses(local, skeleton.ParentIndices, world, new Pose(Vector3.zero, Quaternion.identity));
            return world;
        }

        private static void RotateJointAt(Pose[] pose, int joint, Quaternion delta)
        {
            Assert.GreaterOrEqual(joint, 0, "Missing source joint binding.");
            pose[joint] = new Pose(pose[joint].position, pose[joint].rotation * delta);
        }

        private static Pose[] Capture(List<Transform> bones)
        {
            var poses = new Pose[bones.Count];
            for (var i = 0; i < bones.Count; i++) poses[i] = new Pose(bones[i].position, bones[i].rotation);
            return poses;
        }

        private static void AssertPoseEquals(Pose[] expected, List<Transform> bones, string context)
        {
            for (var i = 0; i < bones.Count; i++)
            {
                AssertVector(expected[i].position, bones[i].position, $"{context}: {bones[i].name} position");
                AssertRotation(expected[i].rotation, bones[i].rotation, $"{context}: {bones[i].name} rotation");
            }
        }

        private static void AssertVector(Vector3 expected, Vector3 actual, string context)
        {
            Assert.LessOrEqual(Vector3.Distance(expected, actual), PositionTolerance,
                $"{context}: expected {expected}, got {actual}.");
        }

        private static void AssertRotation(Quaternion expected, Quaternion actual, string context)
        {
            Assert.LessOrEqual(Quaternion.Angle(MbPose.Normalize(expected), MbPose.Normalize(actual)),
                RotationToleranceDegrees, context);
        }

        private sealed class Rig
        {
            public Animator Animator;
            public SomaVrmRetargeter Retargeter;
        }
    }
}
